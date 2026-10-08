"""Vanilla 3D Gaussian Splatting on RealX3D with gsplat 1.5.x (the project's Stage 0 baseline and
the host for the Stage 2 densification rules later).

Follows gsplat's examples/simple_trainer.py `default` recipe (learning rates, SH schedule,
DefaultStrategy = original 3DGS adaptive density control), with these RealX3D specifics:
* cameras come from cvp.data (metric laser frame, full intrinsics incl. principal point), so the
  trained Gaussians and their depth renders are directly in metres in the ground-truth frame;
* training images: blurred inputs (train/), the sharp references (val/) for the sharp control,
  or a mix ("mixed:0.25" = 25% of training views replaced by their sharp reference);
* evaluation exactly like RealX3D section 5.1: training views rendered and compared with the SHARP
  references ("train" PSNR), plus the held-out test views ("NVS"/test PSNR);
* exports: point_cloud.ply (standard 3DGS format, opens in SuperSplat), per-view depth .npz
  (expected depth, depth std, alpha) for the geometry evaluation, preview images, metrics.json.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import Scene

SH_C0 = 0.28209479177387814


@dataclass
class TrainConfig:
    data: str = "data"
    scene: str = "Ujikintoki"
    condition: str = "motion_strong"
    images: str = "blurred"            # blurred | sharp | mixed:<fraction>
    out: str = "runs"
    tag: str = "vanilla"
    steps: int = 30_000
    downscale: float = 1.0             # 1 = data_4 native (~1758x1168), as in RealX3D
    sh_degree: int = 3
    seed: int = 0
    absgrad: bool = False              # AbsGS-style densification statistic (Stage 2 competitor)
    grow_grad2d: Optional[float] = None
    ssim_lambda: float = 0.2
    eval_lpips: bool = True
    device: str = "cuda"
    log_every: int = 500
    save_depth: bool = True
    max_init: Optional[int] = None     # subsample initial points (CPU smoke tests only)


# ----------------------------------------------------------------------------- losses / metrics
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.exp(-(torch.arange(size, device=device) - size // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """img, ref: [H,W,3] in [0,1]."""
    x, y = img.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]
    w = _gauss_window(device=img.device).to(img.dtype)
    mu_x = F.conv2d(x, w, padding=5, groups=3)
    mu_y = F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean()


def psnr(img: torch.Tensor, ref: torch.Tensor) -> float:
    mse = F.mse_loss(img.clamp(0, 1), ref).item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


# ----------------------------------------------------------------------------- trainer
class GSTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.dev = torch.device(cfg.device)
        if cfg.device == "cpu":
            from .gs_cpu import cpu_rasterization, patch_strategy_ops_for_cpu
            patch_strategy_ops_for_cpu()
            self.raster = cpu_rasterization
        else:
            from gsplat import rasterization
            self.raster = rasterization
        self.sc = Scene(cfg.data, cfg.scene, cfg.condition)
        self.run_dir = os.path.join(cfg.out, cfg.scene, cfg.condition, cfg.tag, f"s{cfg.seed}")
        os.makedirs(self.run_dir, exist_ok=True)
        self._load_views()
        self._init_gaussians()

    # ------------------------------------------------------------------ data
    def _load_views(self):
        sc, ds = self.sc, self.cfg.downscale
        self.train_names = sc.splits["train"]
        self.test_names = sc.splits["test"]
        kinds = ["blurred"] * len(self.train_names)
        mode = self.cfg.images
        if mode == "sharp":
            kinds = ["sharp"] * len(self.train_names)
        elif mode.startswith("mixed:"):
            p = float(mode.split(":")[1])
            rng = np.random.default_rng(1234 + self.cfg.seed)
            k = int(round(p * len(self.train_names)))
            sel = set(rng.choice(len(self.train_names), k, replace=False).tolist())
            kinds = ["sharp" if i in sel else "blurred" for i in range(len(self.train_names))]
        self.train_kinds = kinds

        def stack(names, kind_list):
            return torch.from_numpy(np.stack([
                (sc.load_image(n, k, ds) * 255).astype(np.uint8) for n, k in zip(names, kind_list)]))

        self.train_imgs = stack(self.train_names, kinds).to(self.dev)
        self.ref_imgs = stack(self.train_names, ["sharp"] * len(self.train_names))       # CPU
        self.blur_imgs = stack(self.train_names, ["blurred"] * len(self.train_names))    # CPU
        self.test_imgs = stack(self.test_names, ["test"] * len(self.test_names))          # CPU
        self.H, self.W = self.train_imgs.shape[1:3]

        def cams(names):
            cs = [sc.camera(n, ds) for n in names]
            V = torch.tensor(np.stack([c.viewmat for c in cs]), dtype=torch.float32)
            K = torch.tensor(np.stack([c.K for c in cs]), dtype=torch.float32)
            return V.to(self.dev), K.to(self.dev), cs

        self.train_V, self.train_K, self.train_cams = cams(self.train_names)
        self.test_V, self.test_K, self.test_cams = cams(self.test_names)
        centers = np.stack([c.center for c in self.train_cams])
        self.scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)

    def _init_gaussians(self):
        from scipy.spatial import cKDTree
        P = self.sc.sfm_points.astype(np.float32)
        Cc = self.sc.sfm_colors.astype(np.float32) / 255.0
        if self.cfg.max_init and len(P) > self.cfg.max_init:
            sel = np.random.default_rng(0).choice(len(P), self.cfg.max_init, replace=False)
            P, Cc = P[sel], Cc[sel]
        d, _ = cKDTree(P).query(P, k=4)
        dist = np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-4, None)
        N = len(P)
        K = (self.cfg.sh_degree + 1) ** 2
        sh0 = (torch.from_numpy(Cc) - 0.5) / SH_C0
        params = {
            "means": torch.from_numpy(P),
            "scales": torch.log(torch.from_numpy(dist))[:, None].repeat(1, 3),
            "quats": F.normalize(torch.rand(N, 4), dim=-1),
            "opacities": torch.logit(torch.full((N,), 0.1)),
            "sh0": sh0[:, None, :],
            "shN": torch.zeros(N, K - 1, 3),
        }
        self.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().to(self.dev)) for k, v in params.items()})
        s = self.scene_scale
        lrs = {"means": 1.6e-4 * s, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
               "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        self.optimizers = {k: torch.optim.Adam([{"params": self.splats[k], "lr": lr, "name": k}], eps=1e-15)
                           for k, lr in lrs.items()}
        self.sched = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / self.cfg.steps))
        from gsplat.strategy import DefaultStrategy
        g = self.cfg.grow_grad2d or (0.0008 if self.cfg.absgrad else 0.0002)
        # gsplat simple_trainer defaults: densify from step 500 to 15000, reset opacity every 3000
        self.strategy = DefaultStrategy(absgrad=self.cfg.absgrad, grow_grad2d=g, verbose=False)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)

    # ------------------------------------------------------------------ render
    def render(self, V, K, sh_degree, render_mode="RGB"):
        sp = self.splats
        colors = torch.cat([sp["sh0"], sp["shN"]], 1)
        return self.raster(
            means=sp["means"], quats=sp["quats"], scales=torch.exp(sp["scales"]),
            opacities=torch.sigmoid(sp["opacities"]), colors=colors, viewmats=V, Ks=K,
            width=self.W, height=self.H, sh_degree=sh_degree, packed=False,
            absgrad=self.cfg.absgrad, render_mode=render_mode)

    def render_depth_stats(self, V, K):
        """Expected depth, depth std and alpha for one camera (features = [z, z^2])."""
        sp = self.splats
        means = sp["means"]
        z = (means @ V[0, :3, :3].T + V[0, :3, 3])[:, 2]
        feats = torch.stack([z, z * z], -1)
        out, alpha, _ = self.raster(
            means=means, quats=sp["quats"], scales=torch.exp(sp["scales"]),
            opacities=torch.sigmoid(sp["opacities"]), colors=feats, viewmats=V, Ks=K,
            width=self.W, height=self.H, sh_degree=None, packed=False)
        a = alpha[0, ..., 0].clamp_min(1e-6)
        ez = out[0, ..., 0] / a
        ez2 = out[0, ..., 1] / a
        std = torch.sqrt(torch.clamp(ez2 - ez * ez, min=0))
        return ez, std, alpha[0, ..., 0]

    # ------------------------------------------------------------------ train
    def train(self):
        cfg = self.cfg
        n = len(self.train_names)
        t0 = time.time()
        log = []
        order = []
        for step in range(cfg.steps):
            if not order:
                order = list(np.random.permutation(n))
            i = order.pop()
            gt = self.train_imgs[i].float() / 255.0
            shd = min(step // 1000, cfg.sh_degree)
            img, alpha, info = self.render(self.train_V[i:i + 1], self.train_K[i:i + 1], shd)
            img = img[0]
            self.strategy.step_pre_backward(self.splats, self.optimizers, self.strategy_state, step, info)
            l1 = (img - gt).abs().mean()
            loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim(img, gt))
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.strategy.step_post_backward(self.splats, self.optimizers, self.strategy_state,
                                             step, info, packed=False)
            # gsplat 1.5.3 never resets opacities: its condition `step % reset_every == 0 & step > 0`
            # parses as a chained comparison that is always False. Do the original-3DGS reset here.
            st = self.strategy
            if 0 < step < st.refine_stop_iter and step % st.reset_every == 0:
                from gsplat.strategy.ops import reset_opa
                reset_opa(params=self.splats, optimizers=self.optimizers,
                          state=self.strategy_state, value=st.prune_opa * 2.0)
            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                ng = len(self.splats["means"])
                el = time.time() - t0
                eta = el / (step + 1) * (cfg.steps - step - 1)
                print(f"step {step:6d}/{cfg.steps}  loss {loss.item():.4f}  #gaussians {ng:,}  "
                      f"elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)
                log.append({"step": step, "loss": loss.item(), "n_gaussians": ng, "sec": el})
        self.train_seconds = time.time() - t0
        self.train_log = log

    # ------------------------------------------------------------------ eval + export
    @torch.no_grad()
    def evaluate(self) -> dict:
        cfg = self.cfg
        lp = None
        if cfg.eval_lpips:
            try:
                import lpips
                lp = lpips.LPIPS(net="alex", verbose=False).to(self.dev)
            except Exception as e:  # noqa: BLE001
                print(f"(LPIPS skipped: {e})")

        def score(V, K, refs, names, tag):
            ps, ss, ls = [], [], []
            for i in range(len(names)):
                img = self.render(V[i:i + 1], K[i:i + 1], self.cfg.sh_degree)[0][0].clamp(0, 1)
                ref = refs[i].to(self.dev).float() / 255.0
                ps.append(psnr(img, ref))
                ss.append(ssim(img, ref).item())
                if lp is not None:
                    ls.append(lp(img.permute(2, 0, 1)[None] * 2 - 1, ref.permute(2, 0, 1)[None] * 2 - 1).item())
            r = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(names),
                 "per_view_psnr": dict(zip(names, map(float, ps)))}
            if ls:
                r["lpips"] = float(np.mean(ls))
            print(f"  {tag:28s} PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.3f}" +
                  (f"  LPIPS {r['lpips']:.3f}" if ls else ""), flush=True)
            return r

        print("Evaluation (as in RealX3D: training views vs sharp references, test views):")
        res = {
            "train_vs_sharp_ref": score(self.train_V, self.train_K, self.ref_imgs, self.train_names, "train views vs sharp ref"),
            "train_vs_blurred_input": score(self.train_V, self.train_K, self.blur_imgs, self.train_names, "train views vs blurred input"),
            "test": score(self.test_V, self.test_K, self.test_imgs, self.test_names, "test views (held out)"),
        }
        pub = os.path.join(cfg.data, "baseline_results", cfg.condition, cfg.scene, "3dgs")
        if cfg.images == "blurred" and os.path.isdir(pub):
            ref = {}
            for split in ("train", "test"):
                p = os.path.join(pub, f"eval_{split}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        ref[split] = json.load(f)
            res["realx3d_published_3dgs"] = ref
            if "train" in ref:
                print(f"  RealX3D published 3DGS, same scene/condition: train PSNR {ref['train']['psnr']:.2f}"
                      + (f", test PSNR {ref['test']['psnr']:.2f}" if "test" in ref else ""))
        return res

    @torch.no_grad()
    def export(self, metrics: dict):
        cfg = self.cfg
        d = self.run_dir
        self.save_ply(os.path.join(d, "point_cloud.ply"))
        if cfg.save_depth:
            for split, V, K, names in (("train", self.train_V, self.train_K, self.train_names),
                                       ("test", self.test_V, self.test_K, self.test_names)):
                os.makedirs(os.path.join(d, "depth", split), exist_ok=True)
                for i, n in enumerate(names):
                    ez, std, a = self.render_depth_stats(V[i:i + 1], K[i:i + 1])
                    np.savez_compressed(os.path.join(d, "depth", split, os.path.splitext(n)[0] + ".npz"),
                                        depth=ez.cpu().numpy().astype(np.float32),
                                        std=std.cpu().numpy().astype(np.float16),
                                        alpha=a.cpu().numpy().astype(np.float16))
        self.save_previews()
        out = {"config": asdict(cfg), "scene_scale": self.scene_scale,
               "n_gaussians": int(len(self.splats["means"])),
               "train_minutes": getattr(self, "train_seconds", 0) / 60,
               "train_kinds": dict(zip(self.train_names, self.train_kinds)),
               "image_size": [int(self.W), int(self.H)],
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and cfg.device != "cpu" else "cpu",
               "log": getattr(self, "train_log", []), **metrics}
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved run to {d}")

    @torch.no_grad()
    def save_previews(self, k: int = 2):
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        d = os.path.join(self.run_dir, "previews")
        os.makedirs(d, exist_ok=True)
        idx = np.linspace(0, len(self.train_names) - 1, k).astype(int)
        for i in idx:
            img = self.render(self.train_V[i:i + 1], self.train_K[i:i + 1], self.cfg.sh_degree)[0][0].clamp(0, 1)
            ez, _, a = self.render_depth_stats(self.train_V[i:i + 1], self.train_K[i:i + 1])
            z = ez.cpu().numpy()
            lo, hi = np.percentile(z[a.cpu().numpy() > 0.5], [2, 98]) if (a > 0.5).any() else (0, 1)
            dep = cm.turbo(np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1))[..., :3]
            row = np.concatenate([self.blur_imgs[i].numpy() / 255.0, img.cpu().numpy(),
                                  self.ref_imgs[i].numpy() / 255.0, dep], 1)
            Image.fromarray((row * 255).astype(np.uint8)).save(
                os.path.join(d, f"{os.path.splitext(self.train_names[i])[0]}_input-render-sharp-depth.jpg"), quality=90)

    @torch.no_grad()
    def save_ply(self, path: str):
        sp = {k: v.detach().cpu().numpy() for k, v in self.splats.items()}
        N = len(sp["means"])
        f_dc = sp["sh0"].reshape(N, 3)
        f_rest = sp["shN"].transpose(0, 2, 1).reshape(N, -1)       # 3DGS order: channel-major
        cols = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
                + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        data = np.concatenate([sp["means"], np.zeros((N, 3), np.float32), f_dc, f_rest,
                               sp["opacities"][:, None], sp["scales"], sp["quats"]], 1).astype(np.float32)
        header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N
        header += "".join(f"property float {c}\n" for c in cols) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())
