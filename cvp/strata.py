"""Structural strata on the laser ground truth (PLAN.md section B6.4).

Axis 1, curvature: surface variation sigma = l0/(l0+l1+l2) from PCA of the k nearest laser points.
    Primary scale: k=48 neighbours on the full laser cloud (~3 mm spacing, so ~1.5-2 cm radius). Stability scale: k=48 on a
    1 cm-voxelised cloud (~4 cm radius). Bins: tertiles (low / mid / high).
Axis 2, proximity: rays are marched through a 1 cm occupancy grid of the laser cloud, starting
    2 cm off the surface, along +n (towards the camera side: "clearance" to the facing surface)
    and -n (through the object: "thickness"). 5 rays in a 15-degree cone per direction, median.
    proximity = min(front, back). Bins: tight < 5 cm, near 5-20 cm, open >= 20 cm (or no hit).
No watertight mesh is needed. Pure numpy/scipy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

CURV_NAMES = ["low", "mid", "high"]
PROX_NAMES = ["tight", "near", "open"]
PROX_EDGES = (0.05, 0.20)


def pca_normals_curvature(Q: np.ndarray, tree: cKDTree, P: np.ndarray, k: int = 48,
                          chunk: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """For query points Q, PCA over their k nearest neighbours in P. Returns (normals, sigma)."""
    normals = np.zeros((len(Q), 3), dtype=np.float32)
    sigma = np.zeros(len(Q), dtype=np.float32)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        _, idx = tree.query(q, k=k, workers=-1)
        nb = P[idx].astype(np.float64)                     # (n,k,3)
        X = nb - nb.mean(1, keepdims=True)
        C = np.einsum("nki,nkj->nij", X, X) / k
        w, V = np.linalg.eigh(C)                           # ascending
        w = np.clip(w, 0, None)
        sigma[s:s + chunk] = w[:, 0] / np.maximum(w.sum(1), 1e-12)
        normals[s:s + chunk] = V[:, :, 0]
    return normals, sigma


def orient_normals(Q: np.ndarray, N: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Flip normals to point towards the nearest camera centre (the observed side)."""
    ct = cKDTree(centers)
    _, j = ct.query(Q, k=1)
    to_cam = centers[j] - Q
    flip = (N * to_cam).sum(1) < 0
    N = N.copy()
    N[flip] *= -1
    return N


class Occupancy:
    def __init__(self, P: np.ndarray, voxel: float = 0.01, pad: float = 0.6):
        self.voxel = voxel
        self.lo = P.min(0) - pad
        dims = np.ceil((P.max(0) + pad - self.lo) / voxel).astype(int) + 1
        if np.prod(dims) > 600_000_000:
            raise MemoryError(f"occupancy grid too large {dims}; use a bigger voxel")
        self.dims = dims
        self.grid = np.zeros(dims, dtype=bool)
        ijk = np.floor((P - self.lo) / voxel).astype(int)
        self.grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    def occupied(self, X: np.ndarray) -> np.ndarray:
        ijk = np.floor((X - self.lo) / self.voxel).astype(int)
        ok = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        out = np.zeros(len(X), dtype=bool)
        i = ijk[ok]
        out[ok] = self.grid[i[:, 0], i[:, 1], i[:, 2]]
        return out


def _cone_dirs(n: np.ndarray, angle_deg: float = 15.0) -> List[np.ndarray]:
    """Unit directions: n and 4 directions tilted by angle around it."""
    a = np.radians(angle_deg)
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    dirs = [n]
    for t in (t1, -t1, t2, -t2):
        d = np.cos(a) * n + np.sin(a) * t
        dirs.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    return dirs


def march(occ: Occupancy, Q: np.ndarray, D: np.ndarray, start: float = 0.025,
          step: float = 0.005, max_dist: float = 0.5, chunk: int = 200_000) -> np.ndarray:
    """Distance from Q along D to the first occupied voxel (inf if none within max_dist)."""
    hit = np.full(len(Q), np.inf, dtype=np.float32)
    ts = np.arange(start, max_dist + 1e-9, step)
    for s in range(0, len(Q), chunk):
        q, d, h = Q[s:s + chunk], D[s:s + chunk], hit[s:s + chunk]
        alive = np.ones(len(q), dtype=bool)
        for t in ts:
            if not alive.any():
                break
            ia = np.nonzero(alive)[0]
            occ_now = occ.occupied(q[ia] + t * d[ia])
            h[ia[occ_now]] = t
            alive[ia[occ_now]] = False
        hit[s:s + chunk] = h
    return hit


def proximity(occ: Occupancy, Q: np.ndarray, N: np.ndarray, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """(front, back) median hit distances over a 5-ray cone along +n and -n."""
    out = []
    for sign in (1.0, -1.0):
        dists = np.stack([march(occ, Q, d, **kw) for d in _cone_dirs(sign * N)], 1)
        out.append(np.median(dists, axis=1))
    return out[0], out[1]


def tertile_labels(x: np.ndarray, edges: Optional[Tuple[float, float]] = None):
    if edges is None:
        edges = tuple(np.quantile(x, [1 / 3, 2 / 3]))
    lab = np.digitize(x, edges).astype(np.int8)
    return lab, edges


def proximity_labels(prox: np.ndarray, edges=PROX_EDGES):
    lab = np.digitize(np.where(np.isfinite(prox), prox, 1e9), edges).astype(np.int8)
    return lab


def cohen_kappa(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe + 1e-12))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    n = min(len(a), 200_000)
    idx = np.random.default_rng(0).choice(len(a), n, replace=False)
    return float(spearmanr(a[idx], b[idx]).correlation)
