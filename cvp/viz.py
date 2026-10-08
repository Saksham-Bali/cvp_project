"""Small plotting / export helpers (matplotlib + numpy only)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

STRATUM_COLORS = {
    "curvature": np.array([[0.20, 0.55, 0.85], [0.95, 0.75, 0.20], [0.85, 0.20, 0.20]]),  # low mid high
    "proximity": np.array([[0.85, 0.20, 0.20], [0.95, 0.75, 0.20], [0.20, 0.55, 0.85]]),  # tight near open
}


def splat(cam, P: np.ndarray, colors: np.ndarray, H: int, W: int, radius: int = 2,
          depth_ref: Optional[np.ndarray] = None, tol: float = 0.02):
    """Render coloured points into an image with a z-buffer (nearest wins).
    If depth_ref is given, only points whose depth matches it (visible surface) are drawn."""
    u, v, z = cam.project(P)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, c = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int), z[ok], colors[ok]
    if depth_ref is not None:
        d = depth_ref[v, u]
        vis = (d > 0) & (np.abs(d - z) < np.maximum(tol, 0.01 * z))
        u, v, z, c = u[vis], v[vis], z[vis], c[vis]
    img = np.zeros((H, W, colors.shape[1]), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    order = np.argsort(-z)                       # far first, near last (overwrites)
    u, v, z, c = u[order], v[order], z[order], c[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = np.clip(u + dx, 0, W - 1), np.clip(v + dy, 0, H - 1)
            closer = z <= zbuf[vv, uu] + 1e-6
            img[vv[closer], uu[closer]] = c[closer]
            zbuf[vv[closer], uu[closer]] = z[closer]
            mask[vv[closer], uu[closer]] = True
    return img, mask


def overlay(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float = 0.65):
    out = base.copy()
    out[mask] = (1 - alpha) * base[mask] + alpha * layer[mask]
    return out


def write_ply_colored(path: str, P: np.ndarray, rgb01: np.ndarray, extra: Optional[dict] = None):
    """Binary PLY with float xyz, uchar rgb and optional float scalar fields (opens in
    CloudCompare / MeshLab)."""
    extra = extra or {}
    n = len(P)
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    dt += [(k, "<f4") for k in extra]
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
    c = np.clip(rgb01 * 255, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    for k, v in extra.items():
        arr[k] = v
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for name, t in dt:
        header.append(f"property {'float' if t == '<f4' else 'uchar'} {name}")
    header.append("end_header")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())


def colormap(x: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    t = np.clip((x - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    return cm(t)[..., :3].astype(np.float32)
