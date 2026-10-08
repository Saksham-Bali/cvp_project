"""CPU stand-in for gsplat.rasterization, built from gsplat's own pure-PyTorch reference code.

Only for smoke-testing the training/export pipeline on machines without an NVIDIA GPU (tiny
images, few steps). Real training must use CUDA (Colab/Kaggle/HPC).
"""
from __future__ import annotations

import torch


def cpu_rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                      sh_degree=None, render_mode="RGB", tile_size=16, **_):
    from gsplat.cuda._torch_impl import (_fully_fused_projection, _isect_offset_encode,
                                         _isect_tiles, _quat_scale_to_covar_preci,
                                         _rasterize_to_pixels, _spherical_harmonics)
    C = viewmats.shape[0]
    covars, _ = _quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False)
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        means, covars, viewmats, Ks, width, height)
    if sh_degree is not None:
        campos = torch.linalg.inv(viewmats)[:, :3, 3]                        # [C,3]
        dirs = means[None] - campos[:, None]                                  # [C,N,3]
        K = (sh_degree + 1) ** 2
        rgb = _spherical_harmonics(sh_degree, dirs, colors[None].expand(C, -1, -1, -1))
        rgb = torch.clamp_min(rgb + 0.5, 0.0)
    else:
        rgb = colors[None].expand(C, -1, -1) if colors.dim() == 2 else colors
    feats = rgb if rgb.dim() == 3 else rgb[None]
    if render_mode in ("RGB+ED", "ED", "RGB+D", "D"):
        feats = torch.cat([rgb, depths[..., None]], -1) if "RGB" in render_mode else depths[..., None]
    # dense alpha compositing (tiny images only): pixels x Gaussians, sorted front to back
    C_out_img, C_out_alpha = [], []
    ys, xs = torch.meshgrid(torch.arange(height, dtype=means.dtype) + 0.5,
                            torch.arange(width, dtype=means.dtype) + 0.5, indexing="ij")
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)                  # [P,2]
    for c in range(C):
        vis = (radii[c] > 0).all(-1)
        idx = torch.nonzero(vis).squeeze(-1)
        idx = idx[torch.argsort(depths[c, idx])]
        d = pix[:, None, :] - means2d[c, idx][None]                           # [P,G,2]
        co = conics[c, idx]
        sig = 0.5 * (co[:, 0] * d[..., 0] ** 2 + co[:, 2] * d[..., 1] ** 2) + co[:, 1] * d[..., 0] * d[..., 1]
        a = (opacities[idx][None] * torch.exp(-sig)).clamp(max=0.99)
        a = torch.where(a < 1.0 / 255.0, torch.zeros_like(a), a)
        T = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
        w = a * T                                                             # [P,G]
        C_out_img.append((w @ feats[c, idx]).reshape(height, width, -1))
        C_out_alpha.append(w.sum(1).reshape(height, width, 1))
    img, alpha = torch.stack(C_out_img), torch.stack(C_out_alpha)
    if render_mode in ("RGB+ED", "ED"):
        img = torch.cat([img[..., :-1], img[..., -1:] / alpha.clamp_min(1e-10)], -1)
    means2d.retain_grad() if means2d.requires_grad else None
    info = {"means2d": means2d, "radii": radii, "depths": depths, "width": width, "height": height,
            "n_cameras": C, "tile_size": tile_size, "gaussian_ids": None}
    return img, alpha, info


def patch_strategy_ops_for_cpu():
    """gsplat's split() calls the CUDA covariance kernel; swap in the torch reference on CPU."""
    import gsplat.strategy.ops as ops
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci

    def q2c(quats, scales, compute_covar=True, compute_preci=True, triu=False, **_):
        return _quat_scale_to_covar_preci(quats, scales, compute_covar, compute_preci, triu)
    ops.quat_scale_to_covar_preci = q2c
