"""RealX3D's published baseline results (baseline_results/<cond>/<Scene>/<method>/...).

Each method folder has eval_train.json / eval_test.json (PSNR, SSIM, LPIPS, depth_L1) and
train/{rgb,depth}/NNNN.png, test/{rgb,depth}/NNNN.png (NNNN = 0001.. in split order).

The depth PNGs are uint16 and MIN-MAX NORMALISED PER IMAGE (verified 8 Oct 2026: 0 and 65535 in
every image; linear in z, not in 1/z). They carry no metric scale, so we recover metric depth per
view with a robust affine fit z = a + b*x against the laser GT depth (2 numbers per view).
Caveat: this per-view alignment is generous (it removes per-view scale/offset errors) but it is the
same for every method, and errors *within* a view (wrong shapes, floaters) remain.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Verified 8 Oct 2026 on Ujikintoki: RealX3D's released BAD-Gaussians depth maps are byte-identical
# in all four blur conditions (only 30 unique files for 4 x 30 views), while its RGB renders differ.
# The depth was evidently not re-exported per condition, so its geometry cannot be scored from the
# release. Its photometric numbers are still reported.
DEPTH_UNRELIABLE = {"Bad_gs": "published depth maps identical across all 4 blur conditions (release bug)"}

METHODS = ["3dgs", "Deblurring3DGS", "BAGS", "Bad_gs", "CoCoGS", "Deblur_GS"]
PRETTY = {"3dgs": "3DGS", "Deblurring3DGS": "Deblurring-3DGS", "BAGS": "BAGS",
          "Bad_gs": "BAD-Gaussians", "CoCoGS": "CoCoGaussian", "Deblur_GS": "Deblur-GS"}


def method_dir(data_root: str, cond: str, scene: str, method: str) -> str:
    return os.path.join(data_root, "baseline_results", cond, scene, method)


def available_methods(data_root: str, cond: str, scene: str) -> List[str]:
    d = os.path.join(data_root, "baseline_results", cond, scene)
    if not os.path.isdir(d):
        return []
    have = set(os.listdir(d))
    return [m for m in METHODS if m in have] + sorted(have - set(METHODS))


def read_eval(data_root: str, cond: str, scene: str, method: str) -> dict:
    out = {}
    for split in ("train", "test"):
        p = os.path.join(method_dir(data_root, cond, scene, method), f"eval_{split}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[split] = json.load(f)
    return out


def robust_affine(x: np.ndarray, z: np.ndarray, iters: int = 15, delta: float = 0.05,
                  max_n: int = 200_000, seed: int = 0) -> Tuple[float, float]:
    """Fit z = a + b*x with Huber IRLS (delta in metres)."""
    if len(x) > max_n:
        idx = np.random.default_rng(seed).choice(len(x), max_n, replace=False)
        x, z = x[idx], z[idx]
    A = np.stack([np.ones_like(x), x], 1)
    w = np.ones_like(x)
    coef = np.zeros(2)
    for _ in range(iters):
        Aw = A * w[:, None]
        coef = np.linalg.lstsq(Aw.T @ A, Aw.T @ z, rcond=None)[0]
        r = np.abs(z - A @ coef)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return float(coef[0]), float(coef[1])


def ray_factor(cam, H: int, W: int) -> np.ndarray:
    """Ray length per unit z-depth for each pixel: sqrt(1 + x^2 + y^2) in normalised coordinates."""
    u = (np.arange(W) + 0.5 - cam.K[0, 2]) / cam.K[0, 0]
    v = (np.arange(H) + 0.5 - cam.K[1, 2]) / cam.K[1, 1]
    return np.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2)


def _fit_view(x: np.ndarray, G: np.ndarray, ray: np.ndarray, encoding: str):
    """Fit one view; returns metric z-depth map and the median |z - G| on valid pixels (cm)."""
    m = G > 0
    if encoding == "z":
        a, b = robust_affine(x[m], G[m].astype(np.float64))
        Z = a + b * x
    else:                                   # "ray": the PNG encodes distance along the camera ray
        a, b = robust_affine(x[m], (G * ray)[m].astype(np.float64))
        Z = (a + b * x) / ray
    Z = Z.astype(np.float32)
    Z[Z <= 0] = 0
    return Z, a, b, float(np.median(np.abs(Z[m] - G[m])) * 100)


def align_views(sc, x_by_name: Dict[str, np.ndarray], encoding: str = "auto") -> Tuple[Dict[str, np.ndarray], dict]:
    """Normalised depth maps (x in [0,1] per view) -> metric z-depth, per-view robust affine fit to the
    laser depth. encoding: 'z', 'ray' (distance along the ray) or 'auto' (pick the one with the lower
    median residual over every 3rd view, then use it for ALL views of this method)."""
    names = list(x_by_name)
    if encoding == "auto":
        res = {"z": [], "ray": []}
        for n in names[::3]:
            x = x_by_name[n]
            G = sc.gt_depth(n, downscale=sc.camera(n).W / x.shape[1])
            ray = ray_factor(sc.camera(n).scaled(sc.camera(n).W / x.shape[1]), *x.shape)
            for e in res:
                res[e].append(_fit_view(x, G, ray, e)[3])
        encoding = min(res, key=lambda e: np.median(res[e]))
        choice = {"encoding": encoding, "probe_median_err_cm": {e: float(np.median(v)) for e, v in res.items()}}
    else:
        choice = {"encoding": encoding}
    out, fits = {}, {}
    for n in names:
        x = x_by_name[n]
        G = sc.gt_depth(n, downscale=sc.camera(n).W / x.shape[1])
        ray = ray_factor(sc.camera(n).scaled(sc.camera(n).W / x.shape[1]), *x.shape)
        Z, a, b, err = _fit_view(x, G, ray, encoding)
        out[n] = Z
        fits[n] = {"a": a, "b": b, "median_abs_err_cm": err}
    return out, {"choice": choice, "views": fits}


def load_metric_depths(sc, data_root: str, cond: str, method: str, encoding: str = "auto"):
    """Metric depth maps for the training views of `method`, aligned per view to GT.

    Verified 8 Oct 2026 (independent review): BAGS's PNGs encode distance along the ray, the other
    methods encode z-depth; 'auto' detects this per method from the fit residuals."""
    d = os.path.join(method_dir(data_root, cond, sc.scene, method), "train", "depth")
    files = sorted(os.listdir(d))
    names = sc.splits["train"]
    if len(files) != len(names):
        raise ValueError(f"{method}: {len(files)} depth files vs {len(names)} train views")
    xs = {n: np.asarray(Image.open(os.path.join(d, f))).astype(np.float64) / 65535.0
          for f, n in zip(files, names)}
    return align_views(sc, xs, encoding)


def normalise_like_published(depth: np.ndarray) -> np.ndarray:
    """Min-max normalise a metric depth map per image, as RealX3D's released PNGs are."""
    v = depth[depth > 0]
    if v.size == 0:
        return depth
    lo, hi = float(v.min()), float(v.max())
    return np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
