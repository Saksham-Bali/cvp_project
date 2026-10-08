"""Stage 1 preview WITHOUT training: per-stratum geometry of RealX3D's six published methods.

For each blur condition and method, take RealX3D's published training-view depth maps, align each
view to metric depth (robust 2-parameter affine fit to the laser depth, see cvp/published.py),
fuse them, and score accuracy / completeness / F-score overall and per curvature / proximity
stratum. Also collects the published PSNR/SSIM/LPIPS so the image ranking can be compared with
the geometry ranking (our question Q1, on one scene).

Needs:  data/data_4/<cond>/<Scene>, data/pointclouds/<Scene>, data/baseline_results/<cond>/<Scene>
Usage:  python scripts/eval_published.py --data data --scene Ujikintoki
        python scripts/eval_published.py --conditions motion_strong defocus_strong motion_mild defocus_mild
Outputs (outputs/<Scene>/published/): geometry_<cond>.csv, summary.csv, fig_strata_<cond>.png,
        fig_ranking.png, results.json
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.data import Scene  # noqa: E402
from cvp.geom import flat_table  # noqa: E402
from cvp.published import DEPTH_UNRELIABLE, PRETTY, available_methods, load_metric_depths, read_eval  # noqa: E402
from cvp.stage1 import LABEL_NAMES, evaluate_depth_maps, prepare_gt, save_json  # noqa: E402


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bar_figure(results, cond, path, metric="F@5cm"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(results)
    groups = [("all", "all")] + [(ax, s) for ax in ("curvature", "proximity") for s in LABEL_NAMES[ax]]
    labels = ["overall"] + [f"{'curv' if ax == 'curvature' else 'prox'}:\n{s}" for ax, s in groups[1:]]
    x = np.arange(len(groups))
    wdt = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    cols = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        r = results[m]
        vals = [r["overall"][metric] if a == "all" else r["strata"][a][s][metric] for a, s in groups]
        ax.bar(x + (i - (len(methods) - 1) / 2) * wdt, vals, wdt, label=PRETTY.get(m, m), color=cols(i))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axvline(0.5, color="k", lw=0.6)
    ax.axvline(3.5, color="k", lw=0.6, ls=":")
    ax.set_ylabel(metric + " (higher is better)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{cond}: geometry of published reconstructions vs laser scan, by structure type")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ranking_figure(summary, path):
    """Rank of each method by test PSNR vs by F@5cm in each stratum (1 = best)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = sorted({r["condition"] for r in summary})
    fig, axes = plt.subplots(1, len(conds), figsize=(6.5 * len(conds), 4.2), squeeze=False)
    cols = ["PSNR_train", "F5_overall", "F5_curv_high", "F5_curv_low", "F5_prox_tight", "F5_prox_open"]
    for ax, c in zip(axes[0], conds):
        rows = [r for r in summary if r["condition"] == c and "F5_overall" in r]
        M = np.array([[r[k] for k in cols] for r in rows], dtype=float)
        ranks = np.zeros_like(M)
        for j in range(M.shape[1]):
            ranks[np.argsort(-M[:, j]), j] = np.arange(1, len(rows) + 1)
        im = ax.imshow(ranks, cmap="RdYlGn_r", vmin=1, vmax=len(rows))
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, f"{int(ranks[i, j])}", ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([k.replace("_", "\n") for k in cols], fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([PRETTY.get(r["method"], r["method"]) for r in rows])
        ax.set_title(f"{c}: rank (1 = best)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--conditions", nargs="+", default=["motion_strong", "defocus_strong"])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--include-unreliable", action="store_true",
                    help="also score methods whose published depth is known to be broken (BAD-Gaussians)")
    ap.add_argument("--stride", type=int, default=4, help="pixel stride for fusion (4 ~ 1 cm at 3.5 m; 1 = all pixels)")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.join(a.out, a.scene, "published")
    os.makedirs(out_dir, exist_ok=True)
    all_res, summary = {}, []
    gt = None
    for cond in a.conditions:
        methods = a.methods or available_methods(a.data, cond, a.scene)
        if not methods:
            print(f"[{cond}] no baseline_results found; download with:\n"
                  f"  python scripts/download_realx3d.py --what baselines --scenes {a.scene} --conditions {cond}")
            continue
        sc = Scene(a.data, a.scene, cond)
        if gt is None:
            print("Preparing ground truth + strata (cached after the first run) ...")
            gt = prepare_gt(sc, a.out)
        all_res[cond] = {}
        rows = []
        for m in methods:
            t1 = time.time()
            if m in DEPTH_UNRELIABLE and not a.include_unreliable:
                ev = read_eval(a.data, cond, a.scene, m)
                summary.append({"condition": cond, "method": m,
                                "PSNR_train": ev.get("train", {}).get("psnr"), "PSNR_test": ev.get("test", {}).get("psnr"),
                                "SSIM_test": ev.get("test", {}).get("ssim"), "LPIPS_test": ev.get("test", {}).get("lpips"),
                                "geometry_skipped": DEPTH_UNRELIABLE[m]})
                print(f"[{cond}] {PRETTY.get(m, m):16s} geometry SKIPPED: {DEPTH_UNRELIABLE[m]}")
                continue
            depths, fits = load_metric_depths(sc, a.data, cond, m)
            res, _ = evaluate_depth_maps(sc, gt, depths, stride=a.stride)
            ev = read_eval(a.data, cond, a.scene, m)
            res["published_photometric"] = ev
            res["depth_encoding"] = fits["choice"]
            res["per_view_fit_median_abs_err_cm"] = float(np.median([f["median_abs_err_cm"] for f in fits["views"].values()]))
            all_res[cond][m] = res
            rows += flat_table(res, m)
            st = res["strata"]
            summary.append({
                "condition": cond, "method": m,
                "PSNR_train": ev.get("train", {}).get("psnr"), "PSNR_test": ev.get("test", {}).get("psnr"),
                "SSIM_test": ev.get("test", {}).get("ssim"), "LPIPS_test": ev.get("test", {}).get("lpips"),
                "F5_overall": res["overall"]["F@5cm"], "P5_overall": res["overall"]["P@5cm"],
                "R5_overall": res["overall"]["R@5cm"],
                "F5_curv_low": st["curvature"]["low"]["F@5cm"], "F5_curv_mid": st["curvature"]["mid"]["F@5cm"],
                "F5_curv_high": st["curvature"]["high"]["F@5cm"],
                "F5_prox_tight": st["proximity"]["tight"]["F@5cm"], "F5_prox_near": st["proximity"]["near"]["F@5cm"],
                "F5_prox_open": st["proximity"]["open"]["F@5cm"],
                "acc_median_cm": res["overall"]["acc_median_cm"], "comp_median_cm": res["overall"]["comp_median_cm"],
                "outlier_frac_unassigned": res["outlier_frac_unassigned"],
                "depth_encoding": fits["choice"]["encoding"], "fit_err_cm": res["per_view_fit_median_abs_err_cm"],
                "realx3d_depth_L1_train": ev.get("train", {}).get("depth_L1"),
            })
            s = summary[-1]
            print(f"[{cond}] {PRETTY.get(m, m):16s} PSNR(train) {s['PSNR_train'] or float('nan'):6.2f}  "
                  f"F@5 all {s['F5_overall']:.3f} | curv low/mid/high {s['F5_curv_low']:.3f}/"
                  f"{s['F5_curv_mid']:.3f}/{s['F5_curv_high']:.3f} | prox tight/near/open "
                  f"{s['F5_prox_tight']:.3f}/{s['F5_prox_near']:.3f}/{s['F5_prox_open']:.3f}  "
                  f"[{s['depth_encoding']}-depth, fit err {s['fit_err_cm']:.1f} cm] "
                  f"({time.time() - t1:.0f}s)", flush=True)
        write_csv(rows, os.path.join(out_dir, f"geometry_{cond}.csv"))
        bar_figure(all_res[cond], cond, os.path.join(out_dir, f"fig_strata_{cond}.png"))
    if summary:
        write_csv(summary, os.path.join(out_dir, "summary.csv"))
        ranking_figure(summary, os.path.join(out_dir, "fig_ranking.png"))
        save_json(all_res, os.path.join(out_dir, "results.json"))
        print(f"\nSaved to {out_dir}  ({time.time() - t0:.0f}s)")
        print("Note: published depth maps are min-max normalised per image; each view was aligned to\n"
              "metric depth with a 2-parameter robust fit to the laser depth (same for all methods).\n"
              "The encoding (z-depth vs distance along the ray) is detected per method (BAGS uses ray distance).")


if __name__ == "__main__":
    main()
