"""Collect all of OUR trained runs for a scene into one table and the Stage-1 "blur damage" figures.

Reads runs/<Scene>/<condition>/<tag>/s0/{metrics.json, geometry.json} (from train_gsplat.py + eval_run.py).
The sharp-control run (tag sharp_control, sharp reference images as input) is the reference for every
blur condition: poses and sharp images are identical in all four conditions.

Our runs are compared with each other using the NATIVE metric protocol (our depth is already metric in
the laser frame; alpha>0.5 and depth-std filter; no fitting). The published-protocol numbers (per-view
2-parameter fit, as needed for RealX3D's published depth) are also listed, for comparison with them.

Usage:  python scripts/summarize_runs.py --runs runs --scene Ujikintoki
Writes: outputs/<Scene>/ours/runs_table.csv, fig_blur_damage.png (F@5 per stratum: sharp vs mild vs
        strong), fig_delta_vs_sharp.png (F(blurred) - F(sharp) per stratum), printed table.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

GROUPS = [("all", "all")] + [("curvature", s) for s in ("low", "mid", "high")] + \
         [("proximity", s) for s in ("tight", "near", "open")]
LABELS = ["overall", "curv\nlow", "curv\nmid", "curv\nhigh", "prox\ntight", "prox\nnear", "prox\nopen"]


def fval(geo, ax, s, key="F@5cm"):
    return geo["overall"][key] if ax == "all" else geo["strata"][ax][s][key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, geos = [], {}
    for mp in sorted(glob.glob(os.path.join(a.runs, a.scene, "*", "*", "s*", "metrics.json"))):
        d = os.path.dirname(mp)
        gp = os.path.join(d, "geometry.json")
        m = json.load(open(mp))
        cfg = m["config"]
        key = "sharp" if cfg["images"] == "sharp" else cfg["condition"]
        if cfg["images"] not in ("sharp", "blurred"):
            key = f"{cfg['condition']}|{cfg['images']}"
        key = f"{key}|{cfg['steps'] // 1000}k" if cfg["steps"] != 7000 else key
        r = {"run": os.path.relpath(d, a.runs), "input": key, "steps": cfg["steps"],
             "n_gaussians": m["n_gaussians"], "train_min": round(m.get("train_minutes", 0), 1),
             "PSNR_train_vs_sharp": round(m["train_vs_sharp_ref"]["psnr"], 2),
             "PSNR_test": round(m["test"]["psnr"], 2),
             "LPIPS_test": round(m["test"].get("lpips", float("nan")), 3)}
        pub = m.get("realx3d_published_3dgs", {})
        if pub:
            r["RealX3D_3DGS_PSNR_train"] = round(pub.get("train", {}).get("psnr", float("nan")), 2)
        if os.path.exists(gp):
            g = json.load(open(gp))
            geos[key] = g
            for (ax, s), lab in zip(GROUPS, LABELS):
                r["F5_" + lab.replace("\n", "_")] = round(fval(g, ax, s), 3)
            r["P5"] = round(g["overall"]["P@5cm"], 3)
            r["R5"] = round(g["overall"]["R@5cm"], 3)
            r["acc_median_cm"] = round(g["overall"]["acc_median_cm"], 2)
            if "published_protocol" in g:
                r["F5_published_protocol"] = round(g["published_protocol"]["overall"]["F@5cm"], 3)
        rows.append(r)
    if not rows:
        raise SystemExit(f"No runs found under {a.runs}/{a.scene}")
    out = os.path.join(a.out, a.scene, "ours")
    os.makedirs(out, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "runs_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{'input':22s} {'steps':>6s} {'#gauss':>8s} {'PSNRtr':>7s} {'PSNRte':>7s} {'F5':>6s} {'F5pub':>6s}")
    for r in rows:
        print(f"{r['input']:22s} {r['steps']:6d} {r['n_gaussians']:8d} {r['PSNR_train_vs_sharp']:7.2f} "
              f"{r['PSNR_test']:7.2f} {r.get('F5_overall', float('nan')):6.3f} "
              f"{r.get('F5_published_protocol', float('nan')):6.3f}")

    # figures -------------------------------------------------------------------------------
    blur_types = [("motion", ["motion_mild", "motion_strong"]), ("defocus", ["defocus_mild", "defocus_strong"])]
    colors = {"sharp": "#2b8a3e", "mild": "#f08c00", "strong": "#c92a2a"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    x = np.arange(len(GROUPS))
    for axp, (bt, conds) in zip(axes, blur_types):
        series = [("sharp", "sharp")] + [(c.split("_")[1], c) for c in conds]
        series = [(n, k) for n, k in series if k in geos]
        w = 0.8 / max(len(series), 1)
        for i, (name, k) in enumerate(series):
            vals = [fval(geos[k], ax, s) for ax, s in GROUPS]
            axp.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, color=colors[name],
                    label=f"{name} input" if name != "sharp" else "sharp control")
        axp.set_xticks(x)
        axp.set_xticklabels(LABELS, fontsize=9)
        axp.axvline(0.5, color="k", lw=0.6)
        axp.axvline(3.5, color="k", lw=0.6, ls=":")
        axp.set_title(f"{bt} blur: our 3DGS (gsplat) geometry vs laser scan")
        axp.set_ylim(0, 1)
        axp.legend(fontsize=8)
    axes[0].set_ylabel("F@5cm (higher is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_blur_damage.png"), dpi=150)
    plt.close(fig)

    if "sharp" in geos:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ks = [k for k in ("motion_mild", "motion_strong", "defocus_mild", "defocus_strong") if k in geos]
        M = np.array([[fval(geos[k], g0, g1) - fval(geos["sharp"], g0, g1) for g0, g1 in GROUPS] for k in ks])
        lim = max(0.05, np.abs(M).max())
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_title("Blur damage: F@5cm(blurred input) − F@5cm(sharp control), per structure type")
        plt.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_delta_vs_sharp.png"), dpi=150)
        plt.close(fig)
        with open(os.path.join(out, "delta_vs_sharp.json"), "w") as f:
            json.dump({k: dict(zip([l.replace("\n", "_") for l in LABELS], map(float, M[i])))
                       for i, k in enumerate(ks)}, f, indent=2)
    print(f"\nSaved table and figures to {out}")


if __name__ == "__main__":
    main()
