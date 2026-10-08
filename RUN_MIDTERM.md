# Mid-term: what is ready, and what to run (Thu 8 Oct → Fri 9 Oct, 10:00)

Everything below was written and tested overnight on the real RealX3D files for the scene
**Ujikintoki**. The results for that scene are **already computed** and sit in `outputs\Ujikintoki\`,
so you can look at them before running anything.

> Plan for the day:
> - **Aditya** runs the GPU training on Colab or Kaggle (step 3). It takes ~1–2 h of mostly waiting.
> - **Saksham** downloads the data, re-runs the CPU steps if he wants (step 2), and builds the slides from `outputs\`.

---

## 0. One-time setup (5 min, either laptop)

1. **Push the code to GitHub.** Colab clones it from there. Open a terminal in `D:\cvp_project`:
   ```
   git add .
   git commit -m "Mid-term code: data loader, strata, geometry harness, gsplat trainer"
   git push
   ```
   `.gitignore` keeps `data\`, `outputs\` and `runs\` out of git. Check that `git status` doesn't list them.
2. **Delete `D:\cvp_project\data\_to_delete`** (3.3 GB). It holds leftovers from an overnight download attempt; I wasn't allowed to delete files on your PC.

## 1. Look at what is already computed (no running needed)

| File in `outputs\Ujikintoki\` | What it shows | Use it for |
|---|---|---|
| `inspect\fig_pairs.png` | One view: sharp vs motion/defocus × mild/strong, with zoom | Goal 1, "the data" slide |
| `inspect\fig_cameras.png` | Top view of the laser scan + 30 train / 5 test cameras | Goal 1 |
| `inspect\laser_preview.png` | Photo next to the laser scan seen from the same camera | Goal 1 ("ground truth lines up") |
| `inspect\fig_sharpness.png`, `inspect\facts.json` | Sharpness per view, dataset facts | Goal 1 |
| `blur\fig_blur_maps.png` | Laser depth, predicted motion-blur length, defocus map | "Blur is geometric" (Lee & Lee) slide |
| `blur\fig_blur_vs_depth.png` | **Measured** blur vs 1/depth for motion and defocus | Same slide; a real finding (below) |
| `strata\fig_strata_0016.png` | Curvature and proximity strata painted onto the photo | Goal 3, Stage 1 slide |
| `strata\strata_*.ply` | Strata as coloured point clouds (open in CloudCompare or MeshLab) | Optional demo |
| `validation.json` | Checks of the geometry scoring (all pass; see 4.4 for what they do and don't cover) | One line on a slide |
| `published\fig_strata_motion_strong.png`, `..._defocus_strong.png` | **Per-stratum geometry of RealX3D's published methods** | Goal 3, main result slide |
| `published\fig_ranking.png`, `published\summary.csv` | Image ranking (PSNR) vs geometry ranking, with RealX3D's own depth_L1 and outlier shares | Goal 3 |

## 2. Laptop (CPU) – re-run or extend (optional for the mid-term)

```
cd D:\cvp_project
pip install -r requirements-cpu.txt
python scripts\download_realx3d.py --what quarter gt --scenes Ujikintoki
python scripts\download_realx3d.py --what baselines --scenes Ujikintoki --conditions motion_strong defocus_strong
python scripts\run_midterm_cpu.py --data data --scene Ujikintoki
```

- **Downloads:** about 2 GB, 10–20 min. The two commands get one scene in all 4 blur settings, its laser scan, and RealX3D's published results for the strong settings.
- **The last command** runs the 5 CPU steps (about 20–40 min on a laptop; step 5 is the slowest) and rewrites `outputs\Ujikintoki\`.
- **To skip step 5:** add `--skip 5`.
- **Everything at once:** for all 8 scenes and 4 settings, run `python scripts\download_realx3d.py --what quarter gt` (~6 GB).

## 3. GPU training on Colab or Kaggle (Aditya)

1. Open the notebook:
   - **Colab:** <https://colab.research.google.com> → File → Open notebook → GitHub → paste `Saksham-Bali/cvp_project` → pick `notebooks/midterm_train_colab_kaggle.ipynb`. Then Runtime → Change runtime type → **T4 GPU**.
   - **Kaggle:** New Notebook → File → Import Notebook → upload the `.ipynb`. In Settings: Accelerator **GPU T4 x1**, Internet **On** (needs a phone-verified account).
2. **Run the cells in order.** Expected times on a T4 (estimates):

   | Step | Time |
   |---|---|
   | install + gsplat compile | 5–10 min |
   | data download | 2–5 min |
   | training, 30 000 steps | ~45–90 min per condition |
   | geometry scoring | ~3–5 min |

   - **Short on time?** Set `STEPS = 7000` in cell 5 (~10–20 min) and label the result as a 7k-step run.
3. **Download `cvp_results.zip`** at the end (last cell). Unzip it into `D:\cvp_project` so `runs\` and `outputs\` merge.

**What you get:**
- train/test PSNR next to **RealX3D's published 3DGS number for the same scene** (printed by cell 5; motion-strong train PSNR 22.05 / test 21.86);
- a preview image (blurred input | our render | sharp reference | depth);
- per-stratum F-scores for our own reconstruction, plus a figure comparing it with the published methods (`outputs/Ujikintoki/compare_motion_strong_vanilla.png`).

**If something fails:**
- *gsplat fails to compile* (a new Colab PyTorch can break 1.5.3): run `!pip install git+https://github.com/nerfstudio-project/gsplat.git`, which builds the latest gsplat (~10 min), then re-run from cell 3. Our code only uses `rasterization` and `DefaultStrategy`, which haven't changed.
- *Colab disconnects:* use Kaggle (30 GPU-hours/week), or run 7k steps.
- *Out of memory:* add `--downscale 2` to the training command (half resolution). The PSNR is then not comparable to RealX3D, so say so.
- *Nothing works by Thursday evening:* present the published-methods results from step 1. They are a complete Goal 3 result without our own training.

## 4. Results from tonight (scene Ujikintoki, single scene, preliminary)

All numbers were re-checked after an **independent review** (section 6), which found and fixed two bugs.

### 4.1 Dataset facts we verified
- **Sharp twins:** every training view has a sharp twin taken from the same camera pose (`train\` blurred, `val\` sharp). For defocus they are pixel-aligned. For motion blur the blurred content is shifted by about half the blur length (~14 px strong, ~5 px mild), because the pose is the *end* of the exposure. Sharp references, poses and the sparse model are byte-identical in all 4 blur settings, so one "sharp control" run per scene covers every condition.
- **Two pose frames:** `transforms_*.json` poses are in a rescaled frame (distances = 0.413 × COLMAP). We use the COLMAP poses with `colmap2world` to get metres in the laser frame.
  - Our cameras reproduce RealX3D's laser depth maps to ~2 mm. That is a consistency check, since both use the same transform.
  - The true photo-to-laser registration error stated in the paper is **1.2 cm**.
- **The JSON `sharpness` field** is computed on the sharp images (same value in every setting), so it is not a blur measure.
- **Per-view blur varies within a strong-motion setting.** The blurred/sharp sharpness ratio spans ~3×, i.e. roughly 1.7× in blur length. Treat it as a ranking proxy; it also depends on image texture.
  - Correction to the plan: per-view reliability weighting is not a no-op for motion blur.
  - Defocus views are more uniform (~1.6× spread in the same ratio).

### 4.2 Blur and depth (`blur\fig_blur_vs_depth.png`)
- **Motion blur:** measured ~30 px (strong) and ~10 px (mild) at quarter resolution, with only a **weak** trend with depth.
  - A 6 cm sideways move alone would give 15–35 px, growing strongly for near points.
  - With depth spanning only 2.6–5 m here, the fit (24 px + 31/Z) is weakly constrained. Present it as "≈30 px, mostly depth-independent".
  - A small camera rotation during the exposure would explain this, but that is a **hypothesis**.
  - Consequence: per-pixel depth-based weighting (rule C in the plan) has less to work with than assumed.
- **Blur is one-sided:** a one-sided blur kernel fits better than a centred one on most patches. That matches the paper's "path *preceding* the target pose": the sharp image sits at the **end** of the blur path.
  - So blurred content is displaced by roughly half the blur (~15 px ≈ 4 cm at 4 m) relative to the given pose. That is close to our 5 cm threshold, a handicap for methods that don't model the camera path. Plan item H3.
- **Defocus:** a disc of ~6 px (mild) and ~12 px (strong) radius, independent of depth, as predicted (focus at 0.4–0.6 m, scene at 2.6–5.1 m).

### 4.3 Strata on the laser scan (`strata\`)
- **Coverage:** 638k laser points (1 cm grid) seen by ≥ 2 training views, 85 % of the scan.
- **Curvature:** tertiles, stable between 2 cm and 4 cm scales (Spearman 0.87, κ 0.63). High = window lattices, plants, furniture edges; low = mostly floor (and tabletops); walls come out mostly "mid".
- **Proximity:** distance to the nearest other surface in front of or behind each point, by ray casting; tight < 5 cm (11 %), near 5–20 cm (9 %), open (80 %). It picks out window frames, umbrella shafts, chair legs and plants.
  - **Limitations:**
    - 91 % of "tight" points are also high-curvature, so the two axes overlap heavily on this scene.
    - About 40 % of "tight" points hit at the very first ray step (2.5 cm), which may sometimes be the point's own curved or noisy surface.
    - The "behind" ray only measures thickness where the far side was scanned.
    - Making the axes independent is a post-mid-term item.

### 4.4 The scoring is trustworthy, within limits (`validation.json`, `tests\`)
- **Frame check:** 2 mm.
- **Laser depth through the full pipeline:** F@5cm 0.978. This is the ceiling, and it tests only fusion and projection.
- **Synthetic noise/hole tests and unit tests:** pass (two plates, slab, plane).
- **Not yet independently validated:** the decoding of RealX3D's published depth. That is exactly where the review found a bug (BAGS, below).

### 4.5 Published methods, per structure type (`published\`)

F@5cm against the laser scan, **one scene, no error bars yet**.
- "Overall" counts every predicted point.
- Per-stratum precision ignores points more than 20 cm from any observed laser point; the "outliers" column gives that share.

| Strong motion | PSNR train | F@5 overall | curv. low / mid / high | prox. tight / near / open | outliers |
|---|---|---|---|---|---|
| CoCoGaussian | 22.52 | 0.64 | 0.59 / 0.67 / 0.73 | 0.77 / 0.78 / 0.64 | 7 % |
| BAGS | 20.78 | 0.62 | 0.56 / 0.66 / 0.72 | 0.79 / 0.77 / 0.61 | 8 % |
| 3DGS | 22.05 | 0.56 | 0.51 / 0.61 / 0.63 | 0.67 / 0.68 / 0.57 | 8 % |
| Deblur-GS | 20.54 | 0.51 | 0.49 / 0.58 / 0.61 | 0.67 / 0.70 / 0.54 | 15 % |
| Deblurring-3DGS | 20.48 | 0.33 | 0.46 / 0.50 / 0.46 | 0.51 / 0.53 / 0.46 | 42 % |

| Strong defocus | PSNR train | F@5 overall | curv. low / mid / high | prox. tight / near / open | outliers |
|---|---|---|---|---|---|
| Deblurring-3DGS | 23.23 | 0.63 | 0.56 / 0.71 / 0.72 | 0.75 / 0.79 / 0.64 | 8 % |
| BAGS | 23.16 | 0.60 | 0.54 / 0.65 / 0.72 | 0.79 / 0.79 / 0.60 | 12 % |
| CoCoGaussian | 23.04 | 0.59 | 0.55 / 0.63 / 0.69 | 0.71 / 0.72 / 0.60 | 10 % |
| 3DGS | 23.03 | 0.56 | 0.50 / 0.62 / 0.63 | 0.68 / 0.70 / 0.56 | 9 % |
| Deblur-GS | 20.71 | 0.50 | 0.49 / 0.56 / 0.63 | 0.67 / 0.66 / 0.54 | 19 % |

BAD-Gaussians is excluded from geometry: its released depth maps are byte-identical in all 4 blur settings while its images differ, a bug in the release. Its PSNR is 22.85 (motion) and 22.14 (defocus).

What this says, honestly, for **one scene**:
1. **Defocus:** the geometry ranking is the **same** as the PSNR ranking. But the top four PSNRs are within 0.2 dB (CoCoGaussian vs 3DGS: 0.01 dB), so this agreement means little.
2. **Motion:** the rankings mostly agree, with two exceptions.
   - BAGS is 3rd of 5 by PSNR but 2nd by geometry; plain 3DGS is 2nd by PSNR but 3rd by geometry.
   - The striking case: **Deblur-GS and Deblurring-3DGS** have almost the same PSNR (20.54 vs 20.48 dB), but F@5 0.51 vs 0.33. Deblurring-3DGS's geometry collapses: 42 % of its points are far from any surface and its per-view depth error is ~37 cm. PSNR does not show this.
   - RealX3D's own `depth_L1` (0.75, the worst) agrees here: the one case where both metrics point the same way.
3. **Structure types:** on this scene the hardest stratum is **flat, low-curvature surfaces**, not thin or high-curvature ones. Tight/near regions score *higher* than open ones.
   - The review traced the low-curvature gap to **completeness**, not floaters, and found a confound: the low tertile is mostly floor (and tabletops) seen at grazing angles.
   - We cannot yet tell "damaged by blur" from "hard anyway". The **sharp-control** run separates them (notebook cell 8; first job after the mid-term).
4. **Ranking by stratum:** within each blur type, the order of methods is nearly the same in every stratum. On one scene, Q1 leans "no", but the top three are within 0.05 (defocus) and 0.08 (motion) and there are no error bars. Not a conclusion yet.
5. **RealX3D's own `depth_L1`** (in the released eval files) disagrees with ours for BAGS in both conditions and for Deblurring-3DGS under defocus. Their metric's definition isn't documented, and BAGS's ray-distance encoding may affect it. Unresolved; say so if asked.

**Method caveat to say out loud:** RealX3D's published depth maps are stretched to 0–65535 in each image. We restore metres with a 2-number fit per view against the laser depth. The fit is generous, but it is the same for all methods. The encoding (z-depth vs distance along the ray, which BAGS uses) is detected per method. For our own runs, `eval_run.py` reports both the native metric score and a score under this same protocol, and only the latter goes in the comparison figure.

## 5. Talking points (8–10 min)

1. **Principle:** Lee & Lee: blur is a geometric effect of camera motion and depth. Show `fig_blur_maps.png` and `fig_blur_vs_depth.png`. On RealX3D the motion blur is one-sided (end-of-exposure pose) and mostly depth-independent; defocus is uniform.
2. **Dataset as it really is:**
   - real (optical) defocus; motion blur synthesized from a reconstruction;
   - a sharp twin for every view;
   - a laser scan in mm (`laser_preview.png`), 1.2 cm registration.
3. **Gap:** the RealX3D paper ranks the six methods by image quality (Table 2); its laser-scan geometry metrics are reported for feed-forward models. Our first numbers: PSNR mostly tracks geometry on this scene, **except** under motion blur, where Deblur-GS and Deblurring-3DGS have the same PSNR but very different geometry (F 0.51 vs 0.33).
4. **What runs:**
   - our loader;
   - strata on the laser scan;
   - the validated geometry scoring;
   - per-stratum geometry of the published methods;
   - our own gsplat 3DGS training (PSNR vs RealX3D's number for the same scene).
5. **What we learned / changed since the proposal:**
   - "real blur" becomes "real defocus + rendered motion blur";
   - no ICP (dataset transform);
   - a sharp control is essential;
   - per-view blur does vary for motion;
   - per-pixel depth weighting is weaker than assumed;
   - two release issues found: BAD-Gaussians depth duplicated across conditions, and an undocumented ray-distance encoding for BAGS;
   - first one-scene evidence that the weakest stratum is flat surfaces seen at grazing angles, not thin structure, so Stage 2's target will be decided by the data (decision tree at gate G2 in the plan);
   - gsplat 1.5.3's opacity reset never fires (operator-precedence bug); our trainer does it explicitly.

## 6. Independent review (done overnight)

After the code was finished, a separate reviewer that hadn't seen the reasoning re-checked the claims against the data and the code.

**Confirmed:**
- frames and pixel conventions (2e-7 m agreement with an independent projection);
- GT depth format;
- identical splits and poses across conditions;
- linear (not inverse) depth encoding;
- the BAD-Gaussians duplication;
- the trainer matches the gsplat 1.5.3 API and `simple_trainer` defaults.

**Found and fixed:**
1. BAGS's published depth is distance along the ray, not z. Decoding it as z made BAGS look like the worst geometry (F 0.34); correctly decoded it is 2nd (F 0.60–0.62). An earlier headline built on this ("0.07 dB apart but 0.63 vs 0.34") was **wrong and has been removed**.
2. The proximity "thickness" ray was cast in the wrong direction, so thin structure was never measured. Fixed, with unit tests added.
3. Our own runs are now also scored under the published-depth protocol, so the comparison is fair.
4. `download_realx3d.py` is now part of the repo; small outputs are no longer gitignored, so Colab can make the comparison figure.
5. The missing opacity reset.
6. Overclaims softened: the rotation explanation, the "2 mm" alignment, "PSNR only".

**Still open:**
- error bars (bootstrap over views);
- the sharp-control run;
- mild conditions and more scenes;
- making the curvature and proximity axes independent;
- reconciling with RealX3D's `depth_L1`.
