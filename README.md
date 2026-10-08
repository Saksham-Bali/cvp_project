# Blur-robust 3D Gaussian Splatting on RealX3D: geometry by structure type

CVP course project (Plaksha University, Monsoon 2026), Saksham Bali and Aditya Masutey.

**Question.** Blur-aware 3D Gaussian Splatting methods are ranked in the RealX3D paper by image
quality (PSNR/SSIM/LPIPS); its release also stores a per-view `depth_L1`, but no laser-scan
geometry score for these methods.
Is their *geometry*, measured against a laser scan, damaged differently on different kinds of
structure (high-curvature, thin or tightly spaced parts)? Does the image ranking hide this?
Can a change to how 3DGS adds Gaussians (densification) recover what blur destroys?
Full plan: [`docs/PLAN.md`](docs/PLAN.md). Mid-term run guide: [`RUN_MIDTERM.md`](RUN_MIDTERM.md).

## Layout

```
cvp/                     python package
  data.py                RealX3D loader: cameras in the metric laser frame, images, GT depth
  geom.py                depth fusion, GT visibility, per-stratum accuracy/completeness/F-score
  strata.py              curvature and proximity strata on the laser scan
  stage1.py              GT preparation (cached) + "evaluate these depth maps"
  published.py           RealX3D's published baseline depth maps -> metric (per-view affine fit)
  gs_train.py            vanilla 3DGS training with gsplat (GPU); gs_cpu.py = CPU smoke-test stand-in
  viz.py                 overlays, colour maps, coloured PLY export
scripts/
  download_realx3d.py    get the data from Hugging Face (ToferFish/RealX3D)
  run_midterm_cpu.py     runs the five CPU steps below in order
  inspect_scene.py       dataset facts + figures
  blur_maps.py           "blur is geometric": predicted and measured blur vs depth
  make_strata.py         strata on the laser scan + figures
  validate_harness.py    sanity checks V1-V3 for the geometry scoring
  eval_published.py      per-stratum geometry of RealX3D's six published methods (no training)
  train_gsplat.py        train 3DGS on one scene/condition (GPU)
  eval_run.py            score one of our trained runs
  gsplat_smoke.py        check gsplat compiles/runs on this GPU
notebooks/midterm_train_colab_kaggle.ipynb   GPU training on Colab/Kaggle, start to finish
tests/test_strata.py     unit tests for the strata (python -m pytest tests -q)
docs/                    plan, proposal, slides, reference paper
data/  outputs/  runs/   (not in git) dataset, figures/metrics, training runs
```

## Verified dataset facts (8 Oct 2026, on Ujikintoki)

- `train/` = blurred inputs; `val/` = **sharp references** from the same camera poses (pixel-aligned
  for defocus; for motion the blurred content is shifted ~half the blur, as the pose is the exposure end)
  (identical across all four blur conditions, as are the poses); `test/` = sharp held-out views.
- `transforms_*.json` poses are in a rescaled "NeRF" frame (distances = 0.413 × COLMAP). We use the
  COLMAP poses in `sparse/0` with `colmap2world` (a similarity, scale 0.532) to get a metric frame.
  Our cameras reproduce RealX3D's own laser depth maps to about 2 mm (a consistency check: both use
  the same transform). The real photo-to-laser registration error reported in the paper is 1.2 cm.
- The GT depth PNGs are uint16 millimetres at full resolution (~4× data_4).
- RealX3D's published baseline depth PNGs are min–max normalised per image and linear in depth:
  z-depth for most methods, but **distance along the ray for BAGS** (detected automatically).
  BAD-Gaussians' published depth is byte-identical in all four conditions (release bug), so its
  geometry is not scored.
- The per-frame `sharpness` field is computed on the sharp images (same value in every condition).
