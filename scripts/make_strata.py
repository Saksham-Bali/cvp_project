"""Compute the structural strata on the laser scan for one scene and draw them (Stage 1, Fig. 2).

Strata (see cvp/strata.py): curvature tertiles (low/mid/high surface variation, ~2 cm scale) and
proximity bins (tight < 5 cm, near 5-20 cm, open >= 20 cm to the nearest other surface or through
the object). Only laser points seen by >= 2 training views are used.

Usage:  python scripts/make_strata.py --data data --scene Ujikintoki [--views 0001.JPG 0016.JPG]
Outputs (outputs/<Scene>/strata/): fig_strata_<view>.png (photo | curvature | proximity),
        strata_curvature.ply, strata_proximity.ply (open in CloudCompare/MeshLab), summary.json
        The labels themselves are cached in outputs/gt_cache/<Scene>_strata.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.data import CONDITIONS, Scene  # noqa: E402
from cvp.stage1 import gt_cache_path, prepare_gt  # noqa: E402
from cvp.strata import CURV_NAMES, PROX_NAMES  # noqa: E402
from cvp.viz import STRATUM_COLORS, overlay, splat, write_ply_colored  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--views", nargs="*", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    a = ap.parse_args()
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cond = next(c for c in CONDITIONS if os.path.isdir(os.path.join(a.data, "data_4", c, a.scene)))
    sc = Scene(a.data, a.scene, cond)
    print("Computing strata on the laser scan (about 1-3 minutes the first time) ...")
    gt = prepare_gt(sc, a.out, force=a.force)
    out = os.path.join(a.out, a.scene, "strata")
    os.makedirs(out, exist_ok=True)
    P = gt["points"].astype(np.float64)
    cc = STRATUM_COLORS["curvature"][gt["curvature"]]
    pc = STRATUM_COLORS["proximity"][gt["proximity"]]
    write_ply_colored(os.path.join(out, "strata_curvature.ply"), P, cc,
                      {"sigma": gt["sigma2"].astype(np.float32)})
    write_ply_colored(os.path.join(out, "strata_proximity.ply"), P, pc,
                      {"proximity_m": np.where(np.isfinite(gt["prox"]), gt["prox"], 9.0).astype(np.float32)})
    tr = sc.splits["train"]
    views = a.views or [tr[len(tr) // 2], tr[len(tr) // 6]]
    for v in views:
        cam = sc.camera(v)
        img = sc.load_image(v, "sharp")
        D = sc.gt_depth(v)
        L1, m1 = splat(cam, P, cc, cam.H, cam.W, radius=2, depth_ref=D)
        L2, m2 = splat(cam, P, pc, cam.H, cam.W, radius=2, depth_ref=D)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
        axes[0].imshow(img); axes[0].set_title(f"{a.scene} {v} (sharp reference)")
        axes[1].imshow(overlay(img * 0.6, L1, m1, 0.85)); axes[1].set_title("curvature strata (laser scan)")
        axes[1].legend(handles=[Patch(color=STRATUM_COLORS["curvature"][i], label=n) for i, n in enumerate(CURV_NAMES)],
                       loc="lower right", fontsize=8)
        axes[2].imshow(overlay(img * 0.6, L2, m2, 0.85)); axes[2].set_title("proximity strata: tight <5 cm, near 5-20 cm, open")
        axes[2].legend(handles=[Patch(color=STRATUM_COLORS["proximity"][i], label=n) for i, n in enumerate(PROX_NAMES)],
                       loc="lower right", fontsize=8)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_strata_{os.path.splitext(v)[0]}.png"), dpi=130)
        plt.close(fig)
    with open(gt_cache_path(a.out, a.scene).replace(".npz", ".json")) as f:
        summ = json.load(f)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    print(f"Saved to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
