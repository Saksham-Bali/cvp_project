"""Mid-term figure: "blur is geometric" (Lee & Lee 2013) shown on RealX3D.

1. PREDICTED motion-blur length per pixel from the laser depth:  b(u) = f * L / Z(u)
   (camera translating sideways by L during the exposure: 2 cm mild, 6 cm strong per the paper).
   NOTE: measured blur on Ujikintoki (~30 px strong, ~10 px mild) is larger than this translation-only
   prediction and depends only weakly on depth (1/Z spans just 2x here, so the fit a + c/Z is weakly
   constrained). A camera rotation during the exposure would produce such a depth-independent smear;
   that is a HYPOTHESIS, not shown. The one-sided line kernel fits better than a centred one on most
   patches, consistent with the paper's "path preceding the target pose" (sharp image = end pose).
2. MEASURED blur per image patch: for textured patches at different depths, find the line kernel
   (motion) or disc kernel (defocus) that best turns the sharp reference patch into the blurred
   patch. Plotted against 1/depth, motion blur should grow ~linearly with 1/Z (near = more blur),
   while defocus blur should be roughly constant (focus plane at 0.4-0.6 m, scene at 2-5 m).
3. A relative defocus circle-of-confusion map  |1/Z_f - 1/Z| / median  (near-flat).

Usage:  python scripts/blur_maps.py --data data --scene Ujikintoki [--view 0011.JPG]
Outputs (outputs/<Scene>/blur/): fig_blur_maps.png, fig_blur_vs_depth.png, blur_stats.json
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.data import Scene  # noqa: E402
from cvp.stage1 import save_json  # noqa: E402

PATH_LEN = {"motion_mild": 0.02, "motion_strong": 0.06}
FOCUS = {"defocus_mild": 0.6, "defocus_strong": 0.4}


def line_kernel(length: float, theta: float) -> np.ndarray:
    if length < 0.5:
        return np.ones((1, 1), np.float32)
    r = int(np.ceil(length)) + 1
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    n = max(int(length * 4), 2)
    for s in np.linspace(0, length, n):
        x, y = r + s * np.cos(theta), r + s * np.sin(theta)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= y0 + dy < k.shape[0] and 0 <= x0 + dx < k.shape[1]:
                k[y0 + dy, x0 + dx] += w
    return k / k.sum()


def disc_kernel(radius: float) -> np.ndarray:
    if radius < 0.5:
        return np.ones((1, 1), np.float32)
    r = int(np.ceil(radius)) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    k = np.clip(radius + 0.5 - d, 0, 1).astype(np.float32)
    return k / k.sum()


def best_kernel(sharp_reg, blur_patch, margin, kind, lengths, thetas):
    """Return (best_param, best_theta, mse_ratio) over a grid."""
    P = blur_patch.shape[0]
    base = None
    best = (np.inf, 0.0, 0.0)
    for L in lengths:
        for th in (thetas if kind == "motion" and L > 0 else [0.0]):
            k = line_kernel(L, th) if kind == "motion" else disc_kernel(L)
            out = cv2.filter2D(sharp_reg, -1, k, borderType=cv2.BORDER_REFLECT)
            crop = out[margin:margin + P, margin:margin + P]
            mse = float(((crop - blur_patch) ** 2).mean())
            if L == 0:
                base = mse
            if mse < best[0]:
                best = (mse, L, th)
    return best[1], best[2], best[0] / max(base, 1e-12)


def measure(sc_sharp, sc_blur, name, gt_depth, kind, max_len, n_patches=40, P=48, margin=70, seed=0):
    sharp = cv2.cvtColor((sc_sharp.load_image(name, "sharp") * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    blur = cv2.cvtColor((sc_blur.load_image(name, "blurred") * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    H, W = sharp.shape
    gx = cv2.Sobel(sharp, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(sharp, cv2.CV_32F, 0, 1)
    energy = cv2.boxFilter(gx ** 2 + gy ** 2, -1, (P, P))
    cands = []
    for y in range(margin, H - margin - P, P // 2):
        for x in range(margin, W - margin - P, P // 2):
            Z = gt_depth[y:y + P, x:x + P]
            valid = Z > 0
            if valid.mean() < 0.95:
                continue
            z = np.median(Z[valid])
            if np.std(Z[valid]) / z > 0.04:
                continue
            cands.append((energy[y + P // 2, x + P // 2], y, x, z))
    if not cands:
        return []
    # spread patches evenly over inverse depth: bin by 1/Z, take the most textured patch per bin
    inv = np.array([1 / c[3] for c in cands])
    edges = np.linspace(inv.min(), inv.max() + 1e-9, n_patches + 1)
    chosen = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        inbin = [c for c, iv in zip(cands, inv) if b0 <= iv < b1]
        if inbin:
            chosen.append(max(inbin, key=lambda c: c[0]))
    lengths = np.arange(0, max_len + 1, 2 if kind == "motion" else 1)
    thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    out = []
    for _, y, x, z in chosen:
        reg = sharp[y - margin:y + P + margin, x - margin:x + P + margin]
        L, th, ratio = best_kernel(reg, blur[y:y + P, x:x + P], margin, kind, lengths, thetas)
        out.append({"y": int(y), "x": int(x), "depth_m": float(z), "blur_px": float(L),
                    "theta_deg": float(np.degrees(th)), "mse_ratio": float(ratio)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--view", default=None, help="training view name, e.g. 0011.JPG (default: middle view)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--patches", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [c for c in ["motion_mild", "motion_strong", "defocus_mild", "defocus_strong"]
            if os.path.isdir(os.path.join(a.data, "data_4", c, a.scene))]
    if not have:
        sys.exit("No data_4 conditions found for this scene.")
    scenes = {c: Scene(a.data, a.scene, c) for c in have}
    sc0 = scenes[have[0]]
    name = a.view or sc0.splits["train"][len(sc0.splits["train"]) // 2]
    cam = sc0.camera(name)
    Z = sc0.gt_depth(name)
    f = 0.5 * (cam.K[0, 0] + cam.K[1, 1])
    out_dir = os.path.join(a.out, a.scene, "blur")
    os.makedirs(out_dir, exist_ok=True)
    stats = {"view": name, "focal_px": f, "depth_percentiles_m": np.percentile(Z[Z > 0], [5, 50, 95]).tolist()}

    # predicted maps -----------------------------------------------------------
    valid = Z > 0
    pred = {}
    for c, L in PATH_LEN.items():
        b = np.where(valid, f * L / np.maximum(Z, 1e-3), np.nan)
        pred[c] = b
        stats[f"pred_motion_blur_px_{c}"] = np.nanpercentile(b, [5, 50, 95]).tolist()
    coc = {}
    for c, zf in FOCUS.items():
        r = np.where(valid, np.abs(1 / zf - 1 / np.maximum(Z, 1e-3)), np.nan)
        r = r / np.nanmedian(r)
        coc[c] = r
        stats[f"relative_coc_{c}_p5_p95"] = np.nanpercentile(r, [5, 95]).tolist()

    sharp = sc0.load_image(name, "sharp")
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.4))
    axes[0, 0].imshow(sharp); axes[0, 0].set_title(f"sharp reference ({name})")
    if "motion_strong" in scenes:
        axes[0, 1].imshow(scenes["motion_strong"].load_image(name, "blurred")); axes[0, 1].set_title("motion blur, strong (6 cm path)")
    im = axes[0, 2].imshow(pred["motion_strong"], cmap="magma", vmin=0, vmax=np.nanpercentile(pred["motion_strong"], 99))
    axes[0, 2].set_title("predicted motion-blur length f·L/Z (px), from laser depth")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.03)
    im = axes[1, 0].imshow(np.where(valid, Z, np.nan), cmap="viridis"); axes[1, 0].set_title("laser depth Z (m)")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.03)
    if "defocus_strong" in scenes:
        axes[1, 1].imshow(scenes["defocus_strong"].load_image(name, "blurred")); axes[1, 1].set_title("defocus, strong (focus at 0.4 m)")
    im = axes[1, 2].imshow(coc["defocus_strong"], cmap="magma", vmin=0, vmax=2)
    axes[1, 2].set_title("relative defocus blur size |1/Zf − 1/Z| (median = 1)")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.03)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_blur_maps.png"), dpi=130)
    plt.close(fig)

    # measured blur vs depth ------------------------------------------------------
    meas = {}
    tr = sc0.splits["train"]
    views = [tr[len(tr) // 2], tr[len(tr) // 5], tr[(4 * len(tr)) // 5]]
    for c in have:
        kind = "motion" if c.startswith("motion") else "defocus"
        max_len = {"motion_mild": 40, "motion_strong": 110, "defocus_mild": 30, "defocus_strong": 40}[c]
        meas[c] = []
        for vn in views:
            meas[c] += measure(scenes[c], scenes[c], vn, sc0.gt_depth(vn), kind, max_len, n_patches=a.patches)
        print(f"  measured {len(meas[c])} patches for {c}", flush=True)
    stats["measured"] = meas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    zz = np.linspace(max(np.nanmin(Z[valid]), 0.3), np.nanmax(Z[valid]), 100)
    fits = {}
    for c, col in (("motion_mild", "tab:blue"), ("motion_strong", "tab:red")):
        if c in meas and meas[c]:
            d = np.array([[m["depth_m"], m["blur_px"]] for m in meas[c] if m["mse_ratio"] < 0.8])
            if len(d):
                axes[0].scatter(1 / d[:, 0], d[:, 1], c=col, s=18, label=f"{c} measured")
            axes[0].plot(1 / zz, f * PATH_LEN[c] / zz, c=col, ls=":", label=f"translation only f·L/Z, L={PATH_LEN[c] * 100:.0f} cm")
            if len(d) > 3:
                A = np.stack([np.ones(len(d)), 1 / d[:, 0]], 1)
                cf = np.linalg.lstsq(A, d[:, 1], rcond=None)[0]
                fits[c] = {"const_px": float(cf[0]), "slope_px_m": float(cf[1]), "L_eff_cm": float(cf[1] / f * 100)}
                axes[0].plot(1 / zz, cf[0] + cf[1] / zz, c=col, ls="-", lw=1,
                             label=f"least-squares fit {cf[0]:.0f}px + {cf[1]:.0f}/Z (weakly constrained)")
    axes[0].set_xlabel("1 / depth  (1/m)   ← far      near →")
    axes[0].set_ylabel("blur length (px, data_4 resolution)")
    axes[0].set_title("Motion blur vs 1/depth: mostly constant, weak depth trend")
    axes[0].legend(fontsize=8)
    for c, col in (("defocus_mild", "tab:blue"), ("defocus_strong", "tab:red")):
        if c in meas and meas[c]:
            d = np.array([[m["depth_m"], m["blur_px"]] for m in meas[c] if m["mse_ratio"] < 0.8])
            if len(d):
                axes[1].scatter(1 / d[:, 0], d[:, 1], c=col, s=18, label=f"{c} measured (disc radius)")
    axes[1].set_xlabel("1 / depth  (1/m)")
    axes[1].set_ylabel("defocus disc radius (px)")
    axes[1].set_title("Defocus blur is almost depth-independent here")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_blur_vs_depth.png"), dpi=140)
    plt.close(fig)
    for c in meas:
        d = np.array([[m["depth_m"], m["blur_px"]] for m in meas[c] if m["mse_ratio"] < 0.8])
        if len(d) > 3:
            r = np.corrcoef(1 / d[:, 0], d[:, 1])[0, 1]
            stats[f"corr_blur_vs_invdepth_{c}"] = float(r)
            print(f"  {c}: corr(blur, 1/Z) = {r:.2f} over {len(d)} patches; median blur {np.median(d[:, 1]):.1f} px")
    stats["motion_fit_const_plus_slope_over_Z"] = fits
    for c, v in fits.items():
        print(f"  {c}: least-squares blur ~ {v['const_px']:.1f} px + {v['slope_px_m']:.1f}/Z (weakly constrained)")
    save_json(stats, os.path.join(out_dir, "blur_stats.json"))
    print(f"Saved figures to {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
