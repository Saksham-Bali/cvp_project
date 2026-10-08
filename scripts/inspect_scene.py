"""Mid-term "understand the data" script: facts + figures for one RealX3D scene.

Produces (outputs/<Scene>/inspect/):
  facts.json            counts, intrinsics, principal-point offset, frames, sharpness tables
  fig_pairs.png         one view: sharp reference vs motion/defocus x mild/strong, full + zoomed crop
  fig_cameras.png       top-down map: laser points + training/test camera positions and directions
  fig_sharpness.png     per-view sharpness (variance of Laplacian) of blurred inputs vs sharp references
  laser_preview.png     laser scan rendered from one training camera, coloured by height

Usage:  python scripts/inspect_scene.py --data data --scene Ujikintoki [--view 0016.JPG]
"""
import argparse
import hashlib
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.data import CONDITIONS, Scene, voxel_downsample  # noqa: E402
from cvp.geom import sample_map  # noqa: E402
from cvp.stage1 import save_json  # noqa: E402
from cvp.viz import colormap, splat  # noqa: E402


def lap_var(img01):
    g = cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--view", default=None)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [c for c in CONDITIONS if os.path.isdir(os.path.join(a.data, "data_4", c, a.scene))]
    if not have:
        sys.exit(f"No data for {a.scene} under {a.data}/data_4. Run scripts/download_realx3d.py first.")
    scs = {c: Scene(a.data, a.scene, c) for c in have}
    sc = scs[have[0]]
    out = os.path.join(a.out, a.scene, "inspect")
    os.makedirs(out, exist_ok=True)
    tr, te = sc.splits["train"], sc.splits["test"]
    name = a.view or tr[len(tr) // 2]
    cam = sc.camera(name)
    facts = {
        "scene": a.scene, "conditions_present": have,
        "n_train": len(tr), "n_val_sharp_refs": len(sc.splits["val"]), "n_test": len(te),
        "image_size": [cam.W, cam.H],
        "fx_fy_cx_cy": [round(float(v), 2) for v in (cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2])],
        "principal_point_offset_px": [round(float(cam.K[0, 2] - cam.W / 2), 2), round(float(cam.K[1, 2] - cam.H / 2), 2)],
        "fy_over_fx": round(float(cam.K[1, 1] / cam.K[0, 0]), 4),
        "colmap_to_metres_scale": sc.colmap_scale,
        "n_sfm_points": int(len(sc.sfm_points)),
        "note_splits": "train/ = blurred inputs, val/ = sharp references of the same views, test/ = sharp held-out",
    }
    # are sharp references and poses identical across conditions?
    if len(have) > 1:
        same_val = all(md5(scs[c].image_path(name, "sharp")) == md5(sc.image_path(name, "sharp")) for c in have)
        same_pose = all(np.allclose(scs[c].cams[name].viewmat, sc.cams[name].viewmat) for c in have)
        facts["sharp_refs_identical_across_conditions"] = bool(same_val)
        facts["poses_identical_across_conditions"] = bool(same_pose)

    # sharpness ---------------------------------------------------------------
    table = {"sharp_ref": [lap_var(sc.load_image(n, "sharp")) for n in tr]}
    for c in have:
        table[c] = [lap_var(scs[c].load_image(n, "blurred")) for n in tr]
    json_sharp = [sc.json_sharpness.get(n) for n in tr]
    facts["sharpness_var_laplacian_median"] = {k: float(np.median(v)) for k, v in table.items()}
    facts["sharpness_within_condition_cv"] = {k: float(np.std(v) / np.mean(v)) for k, v in table.items()}
    ref = np.array(table["sharp_ref"])
    facts["blur_ratio_blurred_over_sharp"] = {
        c: {"median": float(np.median(np.array(table[c]) / ref)),
            "cv_across_views": float(np.std(np.array(table[c]) / ref) / np.mean(np.array(table[c]) / ref)),
            "min": float(np.min(np.array(table[c]) / ref)), "max": float(np.max(np.array(table[c]) / ref))}
        for c in have}
    facts["json_sharpness_field_median"] = float(np.median([s for s in json_sharp if s is not None]))
    facts["json_sharpness_note"] = ("transforms_*.json 'sharpness' is identical in every condition: it was "
                                    "computed on the sharp images (instant-ngp colmap2nerf: variance of "
                                    "Laplacian at full resolution), so it is NOT a blur measure")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k, v in table.items():
        ax.plot(range(1, len(v) + 1), v, marker="o", ms=3, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("training view")
    ax.set_ylabel("variance of Laplacian (log)")
    ax.set_title("Image sharpness per view: blur lowers it ~uniformly within a condition")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_sharpness.png"), dpi=140)
    plt.close(fig)

    # pairs figure ------------------------------------------------------------
    sharp = sc.load_image(name, "sharp")
    g = cv2.cvtColor((sharp * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (201, 201))
    y, x = np.unravel_index(np.argmax(e[100:-100, 100:-100]), e[100:-100, 100:-100].shape)
    y, x = y + 100, x + 100
    cs = 160
    y0, x0 = max(0, y - cs), max(0, x - cs)
    panels = [("sharp reference", sharp)] + [(c.replace("_", " "), scs[c].load_image(name, "blurred")) for c in have]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 5.2))
    for j, (t, im) in enumerate(panels):
        axes[0, j].imshow(im)
        axes[0, j].add_patch(plt.Rectangle((x0, y0), 2 * cs, 2 * cs, fill=False, ec="w", lw=1))
        axes[0, j].set_title(t, fontsize=10)
        axes[1, j].imshow(im[y0:y0 + 2 * cs, x0:x0 + 2 * cs])
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"{a.scene}, view {name}: sharp reference vs blurred inputs (same camera pose)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_pairs.png"), dpi=130)
    plt.close(fig)

    # cameras + laser top-down -----------------------------------------------------
    if sc.has_gt:
        P = voxel_downsample(sc.gt_points(), 0.03)
        up = np.median(np.stack([-sc.cams[n].R[1] for n in tr]), 0)   # camera "up" (−y) averaged
        up /= np.linalg.norm(up)
        e1 = np.cross(up, [1, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        h = P @ up
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(P @ e1, P @ e2, s=0.2, c=h, cmap="viridis", alpha=0.5)
        for names, col, lab in ((tr, "tab:red", "train (blurred input)"), (te, "tab:blue", "test (held out)")):
            C = np.stack([sc.cams[n].center for n in names])
            F = np.stack([sc.cams[n].R[2] for n in names])
            ax.scatter(C @ e1, C @ e2, c=col, s=20, label=lab, zorder=3)
            ax.quiver(C @ e1, C @ e2, F @ e1, F @ e2, color=col, scale=15, width=0.003, zorder=3)
        ax.set_aspect("equal")
        ax.set_title(f"{a.scene}: laser scan (top view) and camera poses (metres)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_cameras.png"), dpi=140)
        plt.close(fig)
        # laser preview from the camera + frame check
        D = sc.gt_depth(name)
        Pf = voxel_downsample(sc.gt_points(), 0.005)
        hcol = colormap(Pf @ up, np.percentile(Pf @ up, 2), np.percentile(Pf @ up, 98))
        img, mask = splat(cam, Pf, hcol, cam.H, cam.W, radius=1, depth_ref=D)
        both = np.concatenate([sharp, np.where(mask[..., None], img, 0)], 1)
        cv2.imwrite(os.path.join(out, "laser_preview.png"), cv2.cvtColor((both * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        d, z, inside = sample_map(cam, Pf, D)
        ok = inside & (d > 0) & (np.abs(z - d) < 0.03)
        facts["frame_check_median_abs_mm"] = float(np.median(np.abs(z[ok] - d[ok])) * 1000)
        facts["laser_points"] = "see laser_preview.png (laser points drawn from the same camera)"
        dv = D[D > 0]
        facts["depth_m_p5_p50_p95"] = np.percentile(dv, [5, 50, 95]).tolist()
    save_json(facts, os.path.join(out, "facts.json"))
    for k, v in facts.items():
        print(f"  {k}: {v}")
    print(f"Saved to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
