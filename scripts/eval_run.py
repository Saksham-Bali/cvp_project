"""Score the geometry of one of OUR trained runs (scripts/train_gsplat.py) against the laser scan,
overall and per curvature / proximity stratum. Runs on CPU (laptop, Colab or Kaggle).

Uses the run's exported per-view depth (metric, already in the laser frame - no alignment needed):
pixels with alpha > 0.5 and depth-std/depth < --max-rel-std (filters pixels that blend two surfaces,
the main source of "floaters"), restricted to pixels where the laser depth is valid.

Usage:  python scripts/eval_run.py --run runs/Ujikintoki/motion_strong/vanilla/s0 --data data
Writes: <run>/geometry.json, <run>/geometry.csv; and, if outputs/<Scene>/published/results.json
        exists, a comparison figure with RealX3D's published methods:
        outputs/<Scene>/compare_<condition>.png
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.data import Scene  # noqa: E402
from cvp.geom import flat_table  # noqa: E402
from cvp.published import align_views, normalise_like_published  # noqa: E402
from cvp.stage1 import evaluate_depth_maps, prepare_gt, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-rel-std", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    with open(os.path.join(a.run, "metrics.json")) as f:
        meta = json.load(f)
    cfg = meta["config"]
    sc = Scene(a.data, cfg["scene"], cfg["condition"])
    gt = prepare_gt(sc, a.out)
    depth, mask, raw = {}, {}, {}
    for n in sc.splits["train"]:
        p = os.path.join(a.run, "depth", "train", os.path.splitext(n)[0] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        d = z["depth"].astype(np.float32)
        s = z["std"].astype(np.float32)
        al = z["alpha"].astype(np.float32)
        depth[n] = np.where(al > 0.5, d, 0)
        raw[n] = d
        mask[n] = (al > 0.5) & (s <= a.max_rel_std * np.maximum(d, 1e-6))
    if not depth:
        sys.exit("No depth files found in the run folder (was it trained with --no-depth?)")
    res, _ = evaluate_depth_maps(sc, gt, depth, mask, stride=a.stride)
    res["photometric"] = {k: meta.get(k) for k in ("train_vs_sharp_ref", "test")}
    # Same protocol as for RealX3D's published depth (min-max normalise each view, 2-parameter fit to
    # the laser depth, no alpha/std masks) so that the comparison with published methods is fair.
    print("scoring again with the published-depth protocol (for the comparison figure) ...")
    aligned, _ = align_views(sc, {n: normalise_like_published(v) for n, v in raw.items()}, encoding="z")
    res_pub, _ = evaluate_depth_maps(sc, gt, aligned, stride=a.stride)
    res["published_protocol"] = res_pub
    res["filter"] = {"alpha": 0.5, "max_rel_std": a.max_rel_std}
    save_json(res, os.path.join(a.run, "geometry.json"))
    rows = flat_table(res, cfg["tag"])
    with open(os.path.join(a.run, "geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    o, st = res["overall"], res["strata"]
    print(f"{cfg['scene']} / {cfg['condition']} / {cfg['tag']}:  F@5cm overall {o['F@5cm']:.3f} "
          f"(P {o['P@5cm']:.3f}, R {o['R@5cm']:.3f}); acc median {o['acc_median_cm']:.1f} cm")
    for ax, d in st.items():
        print(f"   {ax:10s} " + "  ".join(f"{k}: {v['F@5cm']:.3f}" for k, v in d.items()))
    print(f"  published-protocol F@5cm overall {res_pub['overall']['F@5cm']:.3f} "
          f"(use this one when comparing with RealX3D's published methods)")

    pub = os.path.join(a.out, cfg["scene"], "published", "results.json")
    if os.path.exists(pub):
        with open(pub) as f:
            allres = json.load(f)
        if cfg["condition"] in allres:
            sys.path.insert(0, os.path.dirname(__file__))
            from eval_published import bar_figure
            merged = dict(allres[cfg["condition"]])
            merged[f"ours: gsplat {cfg['tag']}"] = res_pub
            path = os.path.join(a.out, cfg["scene"], f"compare_{cfg['condition']}_{cfg['tag']}.png")
            bar_figure(merged, cfg["condition"], path)
            print(f"comparison figure: {path}")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
