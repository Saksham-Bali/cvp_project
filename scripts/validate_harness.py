"""Validate the geometry harness before trusting any number (PLAN.md section B6.7).

V3  frame check: laser points projected with our cameras must match the GT depth maps (< 1 cm)
V2  GT depth through the full pipeline (fusion + metrics) must score almost perfectly
V1  synthetic: GT + 5 mm noise -> F@5cm ~ 1; removing a 30 cm patch lowers completeness by
    exactly the removed share, in the right strata

Usage:
  python scripts/validate_harness.py --data data --scene Ujikintoki
Outputs: outputs/<scene>/validation.json (+ printed PASS/FAIL)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.data import Scene  # noqa: E402
from cvp.geom import evaluate, sample_map  # noqa: E402
from cvp.stage1 import LABEL_NAMES, evaluate_depth_maps, laser_tree, prepare_gt, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--condition", default="motion_strong", help="any; poses are shared")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    t0 = time.time()
    sc = Scene(a.data, a.scene, a.condition)
    gt = prepare_gt(sc, a.out)
    rep = {}

    # V3 -------------------------------------------------------------
    P = sc.gt_points()
    sub = P[np.random.default_rng(0).choice(len(P), min(len(P), 500_000), replace=False)]
    v3 = []
    for n in sc.splits["train"][::7] + sc.splits["test"][:1]:
        cam, D = sc.camera(n), sc.gt_depth(n)
        d, z, inside = sample_map(cam, sub, D)
        ok = inside & (d > 0)
        diff = np.abs(z[ok] - d[ok])
        vis = diff < 0.03
        v3.append({"view": n, "visible_frac": float(vis.mean()),
                   "median_abs_diff_mm": float(np.median(diff[vis]) * 1000)})
    med = float(np.median([r["median_abs_diff_mm"] for r in v3]))
    rep["V3_frame_check"] = {"views": v3, "median_mm": med, "pass": med < 10}
    print(f"V3 frame check: median |z_proj - z_gt| = {med:.1f} mm  -> {'PASS' if med < 10 else 'FAIL'}")

    # V2 -------------------------------------------------------------
    depth = {n: sc.gt_depth(n) for n in sc.splits["train"]}
    res, pred = evaluate_depth_maps(sc, gt, depth)
    f5 = res["overall"]["F@5cm"]
    f2 = res["overall"]["F@2cm"]
    rep["V2_gt_depth_through_pipeline"] = {"overall": res["overall"],
                                            "strata": res["strata"], "pass": f5 > 0.97}
    print(f"V2 GT depth through pipeline: F@5cm={f5:.3f}  F@2cm={f2:.3f}  "
          f"(P@5={res['overall']['P@5cm']:.3f}, R@5={res['overall']['R@5cm']:.3f}) "
          f"-> {'PASS' if f5 > 0.97 else 'FAIL'}")
    for ax, d in res["strata"].items():
        print("   ", ax, {k: round(v["F@5cm"], 3) for k, v in d.items()})

    # V1 -------------------------------------------------------------
    E = gt["points"].astype(np.float64)
    labels = {"curvature": gt["curvature"].astype(int), "proximity": gt["proximity"].astype(int)}
    rng = np.random.default_rng(1)
    noisy = E + rng.normal(0, 0.005, E.shape)
    r1 = evaluate(noisy, laser_tree(sc), E, labels, LABEL_NAMES)
    centre = E[rng.integers(len(E))]
    keep = np.linalg.norm(E - centre, axis=1) > 0.30
    removed = ~keep
    r2 = evaluate(noisy[keep], laser_tree(sc), E, labels, LABEL_NAMES)
    exp_R = keep.mean()
    got_R = r2["overall"]["R@5cm"]
    ok1 = r1["overall"]["F@5cm"] > 0.99 and abs(got_R - exp_R) < 0.01
    per = {}
    for ax, lab in labels.items():
        per[ax] = {}
        for k, name in enumerate(LABEL_NAMES[ax]):
            m = lab == k
            per[ax][name] = {"expected_R": float(keep[m].mean()) if m.any() else None,
                             "got_R": r2["strata"][ax][name]["R@5cm"]}
    rep["V1_synthetic"] = {"noise_F5": r1["overall"]["F@5cm"], "hole_points": int(removed.sum()),
                           "expected_R": float(exp_R), "got_R": got_R, "per_stratum": per, "pass": ok1}
    print(f"V1 synthetic: noise-only F@5cm={r1['overall']['F@5cm']:.4f}; hole removed "
          f"{removed.sum()} pts -> R@5 expected {exp_R:.4f}, got {got_R:.4f} -> {'PASS' if ok1 else 'FAIL'}")

    os.makedirs(os.path.join(a.out, a.scene), exist_ok=True)
    save_json(rep, os.path.join(a.out, a.scene, "validation.json"))
    print(f"saved {os.path.join(a.out, a.scene, 'validation.json')}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
