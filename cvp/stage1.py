"""Stage 1 pipeline pieces shared by the scripts: GT preparation (cached) and depth-map evaluation."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from . import strata as S
from .data import Scene, voxel_downsample
from .geom import evaluate, fuse_depths, gt_visibility

LABEL_NAMES = {"curvature": S.CURV_NAMES, "proximity": S.PROX_NAMES}


def gt_cache_path(out_root: str, scene: str) -> str:
    return os.path.join(out_root, "gt_cache", f"{scene}_strata.npz")


def prepare_gt(sc: Scene, out_root: str, voxel: float = 0.01, k: int = 48,
               force: bool = False, verbose: bool = True) -> dict:
    """Build (or load) the per-scene GT evaluation set with strata labels. Cached as .npz."""
    path = gt_cache_path(out_root, sc.scene)
    if os.path.exists(path) and not force:
        z = dict(np.load(path))
        return z
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    P = sc.gt_points()
    log(f"  laser cloud: {len(P):,} points ({time.time() - t0:.0f}s)")
    tree = cKDTree(P)
    E = voxel_downsample(P, voxel)
    log(f"  eval points after {voxel * 100:.0f} cm voxel: {len(E):,}")
    names = sc.splits["train"]
    cams = [sc.camera(n) for n in names]
    deps = [sc.gt_depth(n) for n in names]
    vis = gt_visibility(cams, deps, E)
    obs = vis >= 2
    E, vis = E[obs], vis[obs]
    log(f"  observed in >=2 training views: {len(E):,} ({obs.mean() * 100:.1f}%)  ({time.time() - t0:.0f}s)")
    N, sig2 = S.pca_normals_curvature(E, tree, P, k=k)
    tree1 = cKDTree(voxel_downsample(P, 0.01))
    P1 = tree1.data
    _, sig4 = S.pca_normals_curvature(E, tree1, P1, k=k)
    centers = np.array([sc.cams[n].center for n in sc.cams])
    N = S.orient_normals(E, N.astype(np.float64), centers)
    log(f"  normals + curvature done ({time.time() - t0:.0f}s)")
    occ = S.Occupancy(P, voxel=0.01)
    front, back = S.proximity(occ, E, N)
    prox = np.minimum(front, back)
    log(f"  proximity done ({time.time() - t0:.0f}s)")
    # laser noise makes single-point curvature speckled on flat walls: median-filter it over the
    # 16 nearest evaluation points (~2 cm neighbourhood on the 1 cm grid) before binning
    _, nbr = cKDTree(E).query(E, k=16, workers=-1)
    sig2 = np.median(sig2[nbr], axis=1).astype(np.float32)
    sig4 = np.median(sig4[nbr], axis=1).astype(np.float32)
    curv_lab, edges = S.tertile_labels(sig2)
    curv_lab4, _ = S.tertile_labels(sig4)
    prox_lab = S.proximity_labels(prox)
    out = dict(points=E.astype(np.float32), vis=vis.astype(np.int16), normals=N.astype(np.float32),
               sigma2=sig2, sigma4=sig4, front=front, back=back, prox=prox,
               curvature=curv_lab, curvature_r4=curv_lab4, proximity=prox_lab,
               curv_edges=np.array(edges))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    summary = {
        "scene": sc.scene, "n_laser": int(len(P)), "n_eval_observed": int(len(E)),
        "observed_fraction": float(obs.mean()), "curvature_tertile_edges": list(map(float, edges)),
        "curvature_shares": np.bincount(curv_lab, minlength=3).tolist(),
        "proximity_shares": np.bincount(prox_lab, minlength=3).tolist(),
        "stability_spearman_r2_r4": S.spearman(sig2, sig4),
        "stability_kappa_r2_r4": S.cohen_kappa(curv_lab, curv_lab4),
        "seconds": time.time() - t0,
    }
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  saved {path}")
    return out


_TREE_CACHE: Dict[str, cKDTree] = {}


def laser_tree(sc: Scene) -> cKDTree:
    if sc.scene not in _TREE_CACHE:
        _TREE_CACHE[sc.scene] = cKDTree(sc.gt_points())
    return _TREE_CACHE[sc.scene]


def evaluate_depth_maps(sc: Scene, gt: dict, depth_by_name: Dict[str, np.ndarray],
                        mask_by_name: Optional[Dict[str, np.ndarray]] = None,
                        stride: int = 4, min_consistent: int = 2, voxel: float = 0.01,
                        verbose: bool = False) -> dict:
    """Fuse metric depth maps of training views (restricted to GT-valid pixels) and score them."""
    names = [n for n in sc.splits["train"] if n in depth_by_name]
    cams, deps, masks = [], [], []
    for n in names:
        D = depth_by_name[n].astype(np.float32)
        cam = sc.camera(n)
        if D.shape != (cam.H, cam.W):
            cam = cam.scaled(cam.W / D.shape[1])
        G = sc.gt_depth(n, downscale=sc.camera(n).W / cam.W)
        m = G > 0
        if mask_by_name is not None and n in mask_by_name:
            m &= mask_by_name[n]
        cams.append(cam)
        deps.append(D)
        masks.append(m)
    pred = fuse_depths(cams, deps, masks, stride=stride, min_consistent=min_consistent,
                       voxel=voxel, verbose=verbose)
    labels = {"curvature": gt["curvature"].astype(int), "proximity": gt["proximity"].astype(int)}
    res = evaluate(pred, laser_tree(sc), gt["points"].astype(np.float64), labels, LABEL_NAMES)
    res["n_views"] = len(names)
    return res, pred


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
