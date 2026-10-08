"""Train vanilla 3D Gaussian Splatting (gsplat) on one RealX3D scene + condition.

GPU (Colab / Kaggle / HPC):
  python scripts/train_gsplat.py --data data --scene Ujikintoki --condition motion_strong
  python scripts/train_gsplat.py --condition defocus_strong --steps 7000          # quick run
  python scripts/train_gsplat.py --condition motion_strong --images sharp --tag sharp_control
CPU smoke test only (tiny, minutes; checks the pipeline, not a real result):
  python scripts/train_gsplat.py --device cpu --steps 30 --downscale 8 --no-lpips

Output: runs/<Scene>/<condition>/<tag>/s<seed>/{metrics.json, point_cloud.ply, depth/, previews/}
Then score the geometry:  python scripts/eval_run.py --run <that folder>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cvp.gs_train import GSTrainer, TrainConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="Ujikintoki")
    ap.add_argument("--condition", default="motion_strong",
                    choices=["motion_mild", "motion_strong", "defocus_mild", "defocus_strong"])
    ap.add_argument("--images", default="blurred", help="blurred | sharp | mixed:0.25")
    ap.add_argument("--tag", default=None, help="run name (default: vanilla, or sharp_control / mixed0.25)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--downscale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absgrad", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tag = a.tag or {"blurred": "vanilla", "sharp": "sharp_control"}.get(a.images, a.images.replace(":", ""))
    if a.absgrad and a.tag is None:
        tag += "_absgrad"
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            sys.exit("No CUDA GPU found. On Colab: Runtime > Change runtime type > T4 GPU. "
                     "For a CPU smoke test use --device cpu --steps 30 --downscale 8 --no-lpips")
    cfg = TrainConfig(data=a.data, scene=a.scene, condition=a.condition, images=a.images, out=a.out,
                      tag=tag, steps=a.steps, downscale=a.downscale, seed=a.seed, absgrad=a.absgrad,
                      eval_lpips=not a.no_lpips, device=a.device, save_depth=not a.no_depth,
                      log_every=max(1, min(500, a.steps // 10)),
                      max_init=2000 if a.device == "cpu" else None)
    tr = GSTrainer(cfg)
    print(f"Scene {a.scene} / {a.condition} / images={a.images}: {len(tr.train_names)} train views, "
          f"{len(tr.test_names)} test views, {tr.W}x{tr.H}, {len(tr.splats['means']):,} initial Gaussians, "
          f"scene scale {tr.scene_scale:.2f} m")
    tr.train()
    m = tr.evaluate()
    tr.export(m)


if __name__ == "__main__":
    main()
