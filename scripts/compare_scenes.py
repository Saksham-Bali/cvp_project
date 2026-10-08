"""Combine the published-methods geometry results of several scenes (outputs/<Scene>/published/summary.csv).

For each blur condition and method: F@5cm per scene, the mean over scenes, and the method's rank per
scene (by PSNR and by geometry), so you can see whether a ranking holds across scenes.

Usage:  python scripts/compare_scenes.py --scenes Ujikintoki Laboratory
Writes: outputs/compare_scenes/table.csv, fig_scenes_<condition>.png
"""
import argparse
import csv
import os

import numpy as np

PRETTY = {"3dgs": "3DGS", "Deblurring3DGS": "Deblurring-3DGS", "BAGS": "BAGS", "Bad_gs": "BAD-Gaussians",
          "CoCoGS": "CoCoGaussian", "Deblur_GS": "Deblur-GS"}
COLS = [("F5_overall", "overall"), ("F5_curv_low", "curv low"), ("F5_curv_high", "curv high"),
        ("F5_prox_tight", "prox tight"), ("F5_prox_open", "prox open")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}   # (cond, method) -> {scene: row}
    for s in a.scenes:
        p = os.path.join(a.out, s, "published", "summary.csv")
        if not os.path.exists(p):
            print(f"missing {p} (run eval_published.py --scene {s} first)")
            continue
        for r in csv.DictReader(open(p)):
            if r.get("F5_overall"):
                data.setdefault((r["condition"], r["method"]), {})[s] = r
    scenes = [s for s in a.scenes if any(s in v for v in data.values())]
    out = os.path.join(a.out, "compare_scenes")
    os.makedirs(out, exist_ok=True)
    rows = []
    for cond in sorted({c for c, _ in data}):
        methods = [m for (c, m) in data if c == cond]
        # ranks per scene
        ranks = {}
        for s in scenes:
            present = [m for m in methods if s in data[(cond, m)]]
            for key in ("PSNR_train", "F5_overall"):
                order = sorted(present, key=lambda m: -float(data[(cond, m)][s][key]))
                for i, m in enumerate(order):
                    ranks[(m, s, key)] = i + 1
        for m in methods:
            row = {"condition": cond, "method": PRETTY.get(m, m)}
            for col, _ in COLS + [("PSNR_train", "")]:
                vals = [float(data[(cond, m)][s][col]) for s in scenes if s in data[(cond, m)]]
                for s in scenes:
                    if s in data[(cond, m)]:
                        row[f"{col}|{s}"] = round(float(data[(cond, m)][s][col]), 3)
                row[f"{col}|mean"] = round(float(np.mean(vals)), 3) if vals else None
            for s in scenes:
                row[f"rank_PSNR|{s}"] = ranks.get((m, s, "PSNR_train"))
                row[f"rank_F5|{s}"] = ranks.get((m, s, "F5_overall"))
            rows.append(row)
        # figure: per-scene F5 overall + strata, grouped by method
        fig, axes = plt.subplots(1, len(COLS), figsize=(4 * len(COLS), 3.8), sharey=True)
        x = np.arange(len(methods))
        w = 0.8 / max(len(scenes), 1)
        for ax, (col, lab) in zip(axes, COLS):
            for i, s in enumerate(scenes):
                v = [float(data[(cond, m)][s][col]) if s in data[(cond, m)] else np.nan for m in methods]
                ax.bar(x + (i - (len(scenes) - 1) / 2) * w, v, w, label=s)
            ax.set_xticks(x)
            ax.set_xticklabels([PRETTY.get(m, m) for m in methods], rotation=40, ha="right", fontsize=8)
            ax.set_title(lab)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("F@5cm")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{cond}: published methods' geometry, per scene")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_scenes_{cond}.png"), dpi=140)
        plt.close(fig)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["condition"], f"{r['method']:16s}", "F5:", " ".join(f"{s}={r.get('F5_overall|' + s)}" for s in scenes),
              "| ranks PSNR/F5:", " ".join(f"{r.get('rank_PSNR|' + s)}/{r.get('rank_F5|' + s)}" for s in scenes))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
