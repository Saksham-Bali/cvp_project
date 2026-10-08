"""Geometry harness: GT visibility, depth fusion, per-stratum accuracy / completeness / F-score.

Protocol (PLAN.md section B6), simplified where noted for the mid-term:
* Everything lives in the metric laser frame (cvp.data cameras); no ICP.
* Prediction = per-view metric depth maps of the TRAINING views, back-projected only where the
  GT depth is valid (so prediction and GT cover the same laser-observed region), optionally
  filtered by alpha / depth-std, then a multi-view consistency check, then voxel-downsampled.
* GT evaluation points = laser cloud voxel-downsampled, kept only if observed (depth-tested)
  in >= 2 training views.
* Accuracy: predicted point -> nearest laser point (full-res cloud, ~5 mm spacing).
  Completeness: observed GT eval point -> nearest predicted point.
* Per stratum: GT points use their own label; predicted points take the label of the nearest
  GT eval point (if within 20 cm; otherwise they only count in the overall precision).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from .data import Camera, voxel_downsample

TAUS = (0.02, 0.05, 0.10)


def sample_map(cam: Camera, X: np.ndarray, M: np.ndarray):
    """Project world points into `cam` and read map M (H,W) at the pixel they fall in.
    Returns (value, z, inside) with value=0 where outside."""
    u, v, z = cam.project(X)
    H, W = M.shape[:2]
    inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    val = np.zeros(len(X), dtype=np.float32)
    ui = np.floor(u[inside]).astype(int)
    vi = np.floor(v[inside]).astype(int)
    val[inside] = M[vi, ui]
    return val, z, inside


def gt_visibility(cams: List[Camera], gt_depths: List[np.ndarray], P: np.ndarray,
                  tol_abs: float = 0.01, tol_rel: float = 0.005) -> np.ndarray:
    """Number of views in which each point is visible (projects inside and passes a depth test
    against that view's GT depth map)."""
    count = np.zeros(len(P), dtype=np.int32)
    for cam, D in zip(cams, gt_depths):
        d, z, inside = sample_map(cam, P, D)
        ok = inside & (d > 0) & (np.abs(z - d) < np.maximum(tol_abs, tol_rel * z))
        count += ok
    return count


def fuse_depths(cams: List[Camera], depths: List[np.ndarray],
                masks: Optional[List[np.ndarray]] = None, stride: int = 2,
                min_consistent: int = 2, tol_abs: float = 0.02, tol_rel: float = 0.02,
                voxel: float = 0.01, verbose: bool = False) -> np.ndarray:
    """Back-project per-view depth maps to world points with a multi-view consistency check.

    A point from view i is kept if, projected into the other views, its depth agrees with their
    depth maps (within max(tol_abs, tol_rel*z)) in at least `min_consistent` of them.
    `min_consistent=0` disables the check. Returns voxel-downsampled points (N,3)."""
    out = []
    for i, (cam, D) in enumerate(zip(cams, depths)):
        m = masks[i] if masks is not None else None
        X, _ = cam.backproject(D, m, stride=stride)
        if len(X) == 0:
            continue
        if min_consistent > 0:
            agree = np.zeros(len(X), dtype=np.int32)
            for j, (cj, Dj) in enumerate(zip(cams, depths)):
                if j == i:
                    continue
                d, z, inside = sample_map(cj, X, Dj)
                agree += inside & (d > 0) & (np.abs(z - d) < np.maximum(tol_abs, tol_rel * z))
            X = X[agree >= min_consistent]
        if voxel:
            X = voxel_downsample(X, voxel)
        out.append(X)
        if verbose:
            print(f"    view {i + 1}/{len(cams)}: {len(X)} points kept", flush=True)
    P = np.concatenate(out, 0) if out else np.zeros((0, 3))
    return voxel_downsample(P, voxel) if (voxel and len(P)) else P


def _prf(dist_pred, dist_gt, tau):
    p = float((dist_pred < tau).mean()) if len(dist_pred) else float("nan")
    r = float((dist_gt < tau).mean()) if len(dist_gt) else float("nan")
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate(pred: np.ndarray, gt_tree: cKDTree, gt_eval: np.ndarray,
             labels: Dict[str, np.ndarray], label_names: Dict[str, List[str]],
             taus: Iterable[float] = TAUS, assign_max: float = 0.20, clip: float = 0.20) -> dict:
    """Overall and per-stratum precision/recall/F at each tau, plus mean/median distances.

    gt_tree:   KD-tree over the full-resolution laser cloud (for accuracy)
    gt_eval:   observed, downsampled GT points (for completeness and stratum assignment)
    labels:    {axis: int array (len(gt_eval),)}; -1 = unlabelled
    """
    res = {"n_pred": int(len(pred)), "n_gt_eval": int(len(gt_eval))}
    if len(pred) == 0:
        res["error"] = "empty prediction"
        return res
    d_acc, _ = gt_tree.query(pred, k=1, workers=-1)                 # pred -> laser
    pred_tree = cKDTree(pred)
    d_comp, _ = pred_tree.query(gt_eval, k=1, workers=-1)           # GT -> pred
    eval_tree = cKDTree(gt_eval)
    d_assign, nn = eval_tree.query(pred, k=1, workers=-1)           # pred -> nearest GT eval pt
    assigned = d_assign < assign_max

    def block(da, dc):
        b = {"acc_mean_cm": float(np.minimum(da, clip).mean() * 100) if len(da) else None,
             "acc_median_cm": float(np.median(da) * 100) if len(da) else None,
             "comp_mean_cm": float(np.minimum(dc, clip).mean() * 100) if len(dc) else None,
             "comp_median_cm": float(np.median(dc) * 100) if len(dc) else None,
             "n_pred": int(len(da)), "n_gt": int(len(dc))}
        for tau in taus:
            p, r, f = _prf(da, dc, tau)
            k = f"{int(round(tau * 100))}cm"
            b[f"P@{k}"], b[f"R@{k}"], b[f"F@{k}"] = p, r, f
        return b

    res["overall"] = block(d_acc, d_comp)
    res["outlier_frac_unassigned"] = float(1 - assigned.mean())
    res["strata"] = {}
    for axis, lab in labels.items():
        res["strata"][axis] = {}
        plab = np.where(assigned, lab[nn], -1)
        for k, name in enumerate(label_names[axis]):
            res["strata"][axis][name] = block(d_acc[plab == k], d_comp[lab == k])
    return res


def flat_table(res: dict, method: str, tau_key: str = "5cm") -> List[dict]:
    """Rows for a CSV: one per (axis, stratum) plus overall."""
    rows = [{"method": method, "axis": "all", "stratum": "all",
             **{k: v for k, v in res["overall"].items()}}]
    for axis, d in res.get("strata", {}).items():
        for s, b in d.items():
            rows.append({"method": method, "axis": axis, "stratum": s, **b})
    return rows
