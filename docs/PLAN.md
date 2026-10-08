# CVP Project Plan: from today (8 Oct 2026) to the end-term

**Project:** Blur-robust 3D Gaussian Splatting on RealX3D: stratified geometric diagnosis and fragility-aware densification
**Team:** Saksham Bali, Aditya Masutey · **Course:** Computer Vision & Perception, Plaksha University
**Document owner:** both · **Lives at:** `docs/PLAN.md` · **Last full revision:** 8 Oct 2026

---

## 0. How to use this document

**Task IDs.** Every piece of work has an ID:

| Prefix | Meaning |
|---|---|
| `M*` | mid-term sprint |
| `S0.*` | setup |
| `S1.*` | Stage 1 (diagnosis) |
| `S2.*` | Stage 2 (intervention) |
| `X*` | stretch |
| `D*` | deliverables |

The full, self-contained specification of every task is in **Appendix T (task cards)**.

**How to start a task.** Open a Claude Code cloud session on the repo and say:

> Do task S1.3 from docs/PLAN.md. Follow Appendix T and the "Rules for every task" in T.0.

**Execution tags on each task:**

| Tag | Meaning |
|---|---|
| `[CC]` | Code task a Claude Code cloud session can finish on its own (no GPU, 4 CPU, 16 GB). It writes the code plus tests on synthetic data or the small committed fixture. |
| `[RUN-GPU]` | A human runs it on the HPC, or on Colab/Kaggle as fallback. |
| `[RUN-CPU]` | A human runs it on a laptop or HPC CPU nodes. |
| `[HUMAN]` | Reading, deciding or writing that needs you. |

**Layer tags:** **Core** (must exist for a credible end-term), **Ext** (Extended: what we're aiming for), **Str** (Stretch).

**Evidence tags used throughout:**

| Tag | Meaning |
|---|---|
| ✅ | Verified by web search / dataset pages on 8 Oct 2026 |
| 🔶 | From our earlier chats or general knowledge, not re-verified today |
| ❓ | Assumption or estimate; must be checked (most have a day-1 check) |

---

## 0.1 What changed today: facts verified while writing this plan

These change the plan, so read them first.

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| F1 | `data_4/` (quarter-res) is **one tarball per condition**: `data4_motion_mild` 0.87 GB, `motion_strong` 0.86 GB, `defocus_mild` 0.85 GB, `defocus_strong` 0.84 GB (sizes in bytes on the HF tree). **You cannot download a single scene's images.** | ✅ HF API tree | Mid-term needs ≈1.7 GB of images plus one scene's ground-truth tar (≈0.2–0.4 GB). Fine on campus Wi-Fi. |
| F2 | Ground truth is **per scene**: `pointclouds/<Scene>.tar.gz` (0.18–0.41 GB each, 18 scenes listed incl. Futaba, Midori, Tsubaki, which are not in the README's 15-name list). Each holds `cull_pointcloud.ply`, `cull_mesh.ply`, `colmap2world.npy` (4×4), `depth/` (16-bit PNG rendered from the mesh). The paper says depth is in **millimetres**. | ✅ HF tree + README + paper | We can fetch GT for one scene at a time. |
| F3 | **`colmap2world` exists**: a fixed 4×4 COLMAP→laser-world transform per scene. The paper says it was made with 5–8 manual correspondences + ICP, registration RMSE **1.2 cm**. | ✅ | Use one shared transform per scene for all methods. **No per-method ICP** in the main protocol (Problem 3 solved). F@2 cm sits near the 1.2 cm registration noise floor, so treat it as a sensitivity number only. |
| F4 | The per-frame `sharpness` field almost certainly comes from instant-ngp's `colmap2nerf.py`: `cv2.Laplacian(gray, CV_64F).var()` on the full-res image (**variance of the Laplacian**). The README's `transforms.json` fields (`camera_angle_x`, `aabb_scale`, `fl_x`…) match that script's output exactly. | ✅ script source; ❓ that RealX3D used it | `sharpness` is a crude, texture-dependent blur score. **Big caveat:** by default `colmap2nerf.py` **re-orients, re-centres and rescales** poses (cameras averaged to distance 4). So `transforms_*.json` poses may be in a normalised "NeRF" frame, *not* the COLMAP frame `colmap2world` expects. Day-1 check D3 resolves this. |
| F5 | **Motion blur is rendered from a reconstruction of the sharp scene**: "reconstruct a clean 3D scene from the sharp GT images", constant speed, 64 sub-poses along the 2 cm / 6 cm path **preceding** the target pose (Fig. 3 caption says 5 cm). The renderer is not named. Defocus is optical (focus at 0.6 m / 0.4 m, scene at 3–5 m), and "both levels for every scene". | ✅ paper text | (a) Our "real blur" claim must be split: real defocus, rendered motion blur. (b) **The sharp reference pose is the *end* of the exposure path**, while trajectory methods usually treat the *middle* as the sharp pose. That is a concrete, testable explanation for BAD-Gaussians' weak score (new H3, §B7.6). |
| F6 | "Train" PSNR = render each training view at its pose and compare with the **paired pixel-aligned reference image**. Poses fixed to GT poses. Results averaged per scene. Per-scene numbers are in the appendix (Tables 5–13). | ✅ paper §5.1 | Our mid-term PSNR must be compared with the **per-scene** appendix number, not the 8-scene mean in Table 2. |
| F7 | RealX3D Table 2 attributes "DeBlurring-3DGS" to "Zhang et al., 2024a", the same key it uses for GS-W. Probably a bib error. | ✅ (citation key only) | Assume the method is Lee et al. (benhenryL repo). Cite Lee et al., and note the discrepancy if asked. |
| F8 | `github.com/ShuhongLL/RealX3D` is **empty**, and the HF repo tree has **no `scripts/` folder** despite the README. | ✅ | **No public RealX3D evaluation code.** We write our own harness and must validate it ourselves (§B6.7). |
| F9 | Deblurring-3DGS pins Python 3.7, **CUDA 11.6**, PyTorch 1.12.1. BAGS pins Python 3.8, **CUDA 11.3**, PyTorch 1.12.1 (built on Mip-Splatting, Apache-2.0). BAD-Gaussians needs nerfstudio 1.0.3, PyTorch 2.1.2+cu118 (Apache-2.0). CoCoGaussian code is public (built on Deblurring-3DGS), no LICENSE file seen. Deblur-GS: `Chaphlagical/Deblur-GS` (I3D 2024). Deblurring-3DGS: no LICENSE seen. | ✅ repos | CUDA 11.3/11.6 **cannot compile for sm_89 (Ada) or sm_90 (Hopper)**; CUDA 11.8 is the first that can 🔶. Every 2024 repo needs a re-pinned environment (§B5.3). |
| F10 | gsplat (Apache-2.0): latest tag v1.5.3; `main` (v1.6) requires PyTorch ≥ 2.7. It has MCMC, `absgrad` (AbsGS-style), expected-depth render modes (`ED`, `RGB+ED`), full intrinsics K, antialiased mode. **Median depth only exists for 2DGS** (`rasterization_2dgs` → `render_median`). | ✅ docs | Standardise *our* code on gsplat 1.5.3. Median depth for 3DGS needs another rasterizer (RaDe-GS/GOF), so it is Stretch. We approximate with expected depth plus a depth-variance filter (§B6.2). |
| F11 | **Related work you hadn't listed:** **EntON** (arXiv 2603.06216, Mar 2026) densifies using kNN-covariance *eigenentropy* of neighbouring Gaussian centres, densifying **flat** neighbourhoods and pruning spherical ones. **EA-3DGS** (arXiv 2505.10787) is an outdoor-UAV method whose "structure-aware densification" densifies **low-curvature** regions. **BSGS** (arXiv 2510.12493) uses a depth-conditioned threshold τ0(1+α·e^(−βd)) plus a time-decayed threshold, on ExBluRF/Deblur-NeRF. **"Exploration Matters for Escaping the Blur Trap in 3DGS"** (arXiv 2607.17965, Jul 2026): "blur trap" there means an *optimisation* local optimum, not input blur. | ✅ abstracts | (a) Our factor (ii), "kNN-PCA over Gaussian centres", is close to EntON's machinery but points the **opposite way** (densify fragile, not flat). That gives us a ready-made competing rule, used as an ablation arm. (b) "Structure-aware densification" is literally EA-3DGS's phrase: **rename ours** (§B12). (c) A "depth-only" version of our reliability weighting collapses into BSGS, so it must be framed and tested against that. |
| F12 | Original-3DGS loaders ignore the principal-point offset (they use symmetric FoV) 🔶. The README's full-res example has cx = 3649 vs w/2 = 3614.5, so **≈8.6 px off-centre at quarter res** if typical. Blender-format loaders also use only `camera_angle_x`, but fl_x ≠ fl_y here (4778 vs 4928). | ✅ numbers in README; 🔶 loader behaviour | A possibly systematic geometric bias in Inria-based baselines. Fix: centre-crop all images to the principal point for native repos and use full K in gsplat. Day-1 check D5 measures the offset per scene. |

---

# A. Mid-term sprint (now → Fri 9 Oct, 10:00)

About 32 hours remain. Mid-term evaluates understanding, plus ideally one running implementation.

## A.1 Split: confirmed, with one change

Your parallel split is right: the stratification work is CPU-only and independent of the GPU baseline.

**Change:** add a third, AI-only lane. Two Claude Code cloud sessions write the code *this morning* while you set up downloads and Colab. Code should not be the bottleneck in a 32-hour sprint; data access and GPU setup are.

| Lane | Who | Runs on | Goal |
|---|---|---|---|
| GPU | **Aditya** | Colab T4 now (HPC only if already working) | Goal 2: vanilla 3DGS (gsplat) on one scene, strong motion + strong defocus |
| Data + geometry | **Saksham** | Laptop (CPU, Open3D) | Goal 1 (real-data understanding) + Goal 3 (strata, first harness numbers) |
| Code | Claude Code sessions | Cloud CPU | Write M1, M2, M5–M7 so both humans only *run* things |

## A.2 Timeline (IST)

| When | Saksham | Aditya | Cloud sessions |
|---|---|---|---|
| **Thu 09:00–09:30** | Push `docs/` to GitHub (the GitHub repo is currently **empty**; the files exist only on `D:\`). Start download M3. | Open Colab with a T4 GPU. Run the install cell (M4 step 1) to learn the gsplat compile time. | Session 1: **M1 then M2**. Session 2: **M5, M6, M7**. |
| 09:30–12:00 | M3: list the scenes in the tar, pick one scene, extract it, fetch its GT tar. Run `inspect_scene.py` (M1 output) → dataset facts. | M4: smoke-test gsplat on the scene as soon as M2 lands (pull the repo). | Finish, then push. |
| 12:00–16:00 | M6: blurred/sharp pair figure + motion-blur-length map. M5: strata on laser cloud + screenshots. | M4: 30k-iteration runs (motion-s, then defocus-s); export ply + depth. Record PSNR. | (idle; fix bugs if pinged) |
| 16:00–21:00 | M7: run harness V2 (GT depth through pipeline). Then, if Aditya's depth export exists, compute the first per-stratum Acc/Comp/F. | Optional M9 (Deblurring-3DGS build) **only if** both vanilla runs finished by 16:00. Otherwise copy results to the laptop. | |
| 21:00–00:30 | **Both: slides (M8)** using the talk outline in A.5. | | A session can draft slide text and figure captions from `results/`. |
| **Fri 07:30–09:30** | Rehearse twice (aim for 8–10 min plus questions). | | |

## A.3 Exact steps and commands

### M3: data (Saksham, laptop; also needed on Colab)

```bash
pip install -U "huggingface_hub[cli]"          # gives the `hf` CLI (older versions: huggingface-cli)
# Strong conditions only for the mid-term (≈1.7 GB):
hf download ToferFish/RealX3D data_4/data4_motion_strong.tar.gz  --repo-type dataset --local-dir realx3d
hf download ToferFish/RealX3D data_4/data4_defocus_strong.tar.gz --repo-type dataset --local-dir realx3d
# Which scenes are inside? (paths inside the tar are unknown, so inspect the first levels)
tar -tzf realx3d/data_4/data4_motion_strong.tar.gz | cut -d/ -f1-3 | sort -u | head -50
# Extract one scene only (adapt the pattern to what the listing shows):
tar -xzf realx3d/data_4/data4_motion_strong.tar.gz  --wildcards "*/<Scene>/*" -C realx3d/
tar -xzf realx3d/data_4/data4_defocus_strong.tar.gz --wildcards "*/<Scene>/*" -C realx3d/
hf download ToferFish/RealX3D pointclouds/<Scene>.tar.gz --repo-type dataset --local-dir realx3d
```

Python equivalent: `hf_hub_download("ToferFish/RealX3D", "data_4/data4_motion_strong.tar.gz", repo_type="dataset", local_dir="realx3d")`.

- **Windows:** `tar` exists in PowerShell (bsdtar); `--wildcards` may not be needed. Use `tar -xzf file.tar.gz "<path>/<Scene>"`.
- **Expected time:** ≈3–6 min per GB at 30–50 Mbit/s ❓.
- **Picking the scene:** choose one that appears in *both* motion and defocus tars and has visible thin structure (chair legs, plants, shelves). Write its name in `docs/dataset_notes.md`.

### M1 + M2: code (cloud session writes; you pull)

- **M1 produces:**
  - `scripts/inspect_scene.py`: prints splits, image counts, intrinsics, principal-point offset, `sharpness` stats per split, whether `val/` images are pixel-aligned sharp references of `train/`, and a pose-frame report (transforms vs COLMAP sparse).
  - `cvp/data/realx3d.py`: a loader.
- **M2 produces:**
  - `scripts/train_gsplat.py`: a fork of gsplat's `simple_trainer.py` `default` strategy using our loader, with full K and **no silent world normalisation** (any normalisation is saved as a 4×4 in `frame.json`).
  - `notebooks/colab_midterm.ipynb`.

### M4: Colab run (Aditya)

```python
# Cell 1: install (first gsplat call JIT-compiles CUDA: expect 5–10 min on Colab ❓)
!nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
!git clone https://github.com/Saksham-Bali/cvp_project && pip -q install -r cvp_project/env/requirements-train.txt
!pip -q install ninja gsplat==1.5.3
import torch, gsplat; print(torch.__version__, torch.version.cuda, gsplat.__version__)
!python cvp_project/scripts/gsplat_smoke.py      # renders 1 random Gaussian scene → forces the compile now

# Cell 2: data (to /content, ≈2 GB; Drive is slower)
!hf download ToferFish/RealX3D data_4/data4_motion_strong.tar.gz --repo-type dataset --local-dir /content/realx3d
# … extract the chosen scene as in M3 …

# Cell 3: train. Smoke 7k first, then 30k
!python cvp_project/scripts/train_gsplat.py --scene_dir /content/realx3d/<…>/<Scene> \
    --cond motion_strong --steps 7000 --out /content/runs/midterm/<Scene>/motion_strong/vanilla/s0
!python cvp_project/scripts/train_gsplat.py ... --steps 30000 --out .../s0_30k --save_depth train,test
```

- **Expected times on a T4 at ≈1800×1200 with ~30 views ❓:**
  - 7k steps ≈ 15–25 min;
  - 30k steps ≈ 60–100 min;
  - peak VRAM ≈ 6–12 GB (use `packed=True`).
- **Output:** `metrics.json` with train-view PSNR/SSIM/LPIPS vs the sharp references and test PSNR. Compare with the RealX3D **per-scene** appendix numbers (Saksham looks these up in the paper PDF, Tables 5–13).
- **Expect** our 7k numbers to be 0.3–1 dB below 30k, and gsplat vs the original code to differ by a few tenths of a dB.

### M5–M7: geometry (Saksham, laptop)

```bash
conda create -n cvp-cpu python=3.11 -y && conda activate cvp-cpu
pip install -r env/requirements-cpu.txt      # numpy scipy open3d trimesh opencv-python matplotlib pandas plyfile pytest tqdm
python scripts/strata_preview.py  --gt realx3d/pointclouds/<Scene> --out results/midterm/<Scene>/strata   # M5
python scripts/blur_maps.py       --scene_dir … --gt … --out results/midterm/<Scene>/blur               # M6
python scripts/harness_v2_gtdepth.py --scene_dir … --gt … --out results/midterm/<Scene>/v2             # M7 validation
python scripts/eval_geometry.py   --run runs/midterm/<Scene>/motion_strong/vanilla/s0_30k --gt …        # M7 if depth exists
```

Expected CPU time per scene ❓:
- curvature at two radii on a 1 cm-voxelised cloud: 1–3 min;
- ray-cast proximity (5 rays/point): 1–3 min;
- fusion + metrics: 2–5 min.

## A.4 What is realistic by 10:00 Friday, with fallbacks

| Goal | Realistic target | If it fails → fallback |
|---|---|---|
| **G1 Understanding with real data** | One scene: 4 pairs (motion/defocus × mild/strong, if you also pull the mild tars, +1.7 GB), poses plotted, laser mesh screenshot, `sharpness` table, **motion-blur-length map b(u)=f·L/Z(u)** from GT depth, near-uniform defocus explanation. | Strong-only pairs. If HF download is slow, use 1 tar. |
| **G2 Run baseline** | Vanilla gsplat 30k on motion-strong (+ defocus-strong if time). Train/test PSNR vs the RealX3D per-scene number. A render plus a depth map. | (a) 7k-step result, clearly labelled. (b) Kaggle T4 if Colab disconnects. (c) If gsplat won't compile on Colab: the original `graphdeco-inria/gaussian-splatting` repo (widely run on Colab), using a COLMAP layout. (d) Worst case: show a training curve / partial run plus our verified dataset facts. |
| **G3 Start the novel part** | Curvature + proximity strata on the laser cloud, visualised. Harness validated on GT depth (V2). With G2 output: a **first per-stratum Acc/Comp/F table for one scene** (blurred vanilla), clearly labelled preliminary. | Strata figure + V2 validation + an *example* table produced by feeding a deliberately degraded GT (noise + hole) through the harness. That shows the harness works and how strata separate. |
| M9 Deblurring-3DGS | Only if G2 finished by 16:00. **Don't plan on it.** | Say "builds next week on HPC; legacy CUDA pins documented". |

## A.5 What to say (4–5 points, ≈8 min) plus "what we learned"

1. **Principle (Lee & Lee 2013):**
   - Blur is geometric: the kernel is the projected trajectory of a 3D point during exposure, fixed by depth.
   - Show **our blur-length map**: on RealX3D strong motion (6 cm path), a point at 1 m smears ≈70 px at quarter resolution, but only ≈15 px at 5 m (focal ≈1195 px ❓ per scene).
   - Defocus is the opposite case: the lens focuses at 0.4–0.6 m and the scene sits at 3–5 m, so the blur circle is almost the same size everywhere (±3–5 %).
2. **The dataset, as it really is:**
   - Defocus is optical; motion blur is rendered from a sharp-scene reconstruction.
   - Ground truth is a laser scan in mm, with a fixed COLMAP→world transform (1.2 cm RMSE).
   - `sharpness` is variance of the Laplacian.
   - There is no public evaluation code, so we validated our own harness.
3. **The gap:** RealX3D scores the six blur-aware methods with PSNR/SSIM/LPIPS only. Geometry is reported only for feed-forward models. Rankings flip between motion and defocus, and vanilla 3DGS is competitive.
4. **What runs today:**
   - gsplat baseline on scene X: PSNR vs the paper's per-scene value.
   - Strata on the laser cloud.
   - The harness passing the GT-depth validation.
   - A first per-stratum table, if available.
5. **Refined plan:** two stages, layered Core/Extended/Stretch; Stage 1 first; held-out scenes for Stage 2; timeline to 27 Nov.

**"What we learned since the proposal":** one slide, framed as rigour.

- "Real blur" → *real defocus, rendered motion blur*. We now interpret them separately.
- The "mild-blur plain 3DGS is best" claim is withdrawn (the main table is strong-only), and "first real-capture benchmark" is softened.
- Per-view sharpness weighting is a near no-op when every view has the same blur. We will test it where it *can* matter: mixed sharp/blurred captures, and per-pixel depth-dependent motion blur.
- Alignment: use the dataset's fixed COLMAP→world transform, not per-method ICP, which would absorb systematic error.
- Depth extraction: alpha-blended depth creates in-between "floaters" exactly in thin regions. We filter with depth variance and multi-view consistency and report extractor sensitivity.
- New controls:
  - a **sharp-capture control** to separate "hard structure" from "blur-damaged structure";
  - **budget-matched** baselines, so "more Gaussians" isn't mistaken for "smarter Gaussians".
- New related work found (EntON, BSGS, Exploration/"blur trap"). Our densification is reframed as a controlled intervention study.
- **The exposure-anchor hypothesis:** the sharp reference pose sits at the *end* of the rendered blur path, while trajectory methods assume the *middle*. This is a candidate explanation for BAD-Gaussians' score, and we will test it.

## A.6 What Claude Code cloud sessions can and cannot do here

- **Can:** write and unit-test all Python (loaders, harness, strata, statistics, figure scripts, SLURM files, env files), the notebook, README, slide text and report LaTeX.
- **Cannot:**
  - run CUDA (gsplat has no CPU path; GPU code paths get mocked tests);
  - download RealX3D: assume Hugging Face is **not** reachable from cloud sessions. An attempt to reach it from this planning session's shell was refused by policy.
- **Workaround:** S0.6 builds a small **committed fixture** (`data_fixture/`, < 15 MB) from one real scene, so later cloud sessions test on real formats.
- **Workflow:** run scripts yourself, then paste error output back into a session.

---

# B. The full plan

## B0. Layered scope

All three layers are kept. Each later layer only *adds* runs or analyses on top of the earlier layer's code and results, so dropping a later layer never invalidates an earlier one.

| Layer | Contents | Presentable result if we stop here | Effort (person-h) |
|---|---|---|---|
| **Core** | **Infrastructure:** validated geometry harness (fixed transform, visibility masks, strata, 3D + 2.5D metrics); vanilla gsplat on **8 scenes × {motion-strong, defocus-strong}**; **sharp-capture controls**; seed-noise runs. **Baselines:** Deblurring-3DGS + BAGS (native repos) on the same 16 scene-conditions. **Stage 1 (Q1):** per-stratum analysis + H1 density diagnostic. **Stage 2 v1** on the vanilla host (rules A, B, B−, AbsGS, MCMC, budget-matched ADC), tuned on 2 dev scenes, reported on 6 held-out scenes; the **mixed sharp/blurred protocol at one ratio** (p = 0.25). **Deliverables:** report, slides, video, repo. | "First laser-scan geometric evaluation of blur-aware 3DGS on RealX3D, by structure type, with a sharp control; plus a controlled densification intervention with an honest result." Answers Q1 fully and Q2 on one host. | **85–100** (of which deliverables ≈25) |
| **Extended** | **Mild** conditions for the Core methods. Our own **blur-aware host in gsplat** (motion: learned linear SE(3) exposure trajectory; defocus: learned global PSF), with Stage 2 rules on it (does densification help only when blur is modelled?). **H3** exposure-anchor test. Mixed-protocol **sweep** p ∈ {0, 0.1, 0.25, 0.5, 1}. Rule C (per-pixel blur-extent weighting, motion). Principal-point bias check. | Adds severity trends and the "densification × blur model" interaction (a cleaner H1). Adds a mechanistic explanation of the BAD-Gaussians anomaly. | +30–40 |
| **Stretch** | Full six-method grid (BAD-Gaussians, CoCoGaussian, Deblur-GS). **H2** blur-parameter recovery (learned PSF size / trajectory length vs known 2/6 cm and misfocus). Median-depth extractor (RaDe-GS/GOF rasterizer). Web viewer demo. | A benchmark-extension-grade result. | +30–50 |

**Honest fit.**
- **Core** fits ~7 weeks at the lower end of your hours (~5 h/week each, plus a few 10-hour weeks).
- **Extended** fits only if the GPU queues behave and the baseline builds go smoothly by week 2. Expect to finish about half of it.
- **Stretch** is unlikely, except one cheap item (H2 analysis needs no new training if the Extended hosts exist).

## B1. Critical review of the current plan (blunt)

### What is strong

1. **The question is real and under-served.** RealX3D has laser ground truth but never used it for the optimisation methods. Stratified geometric evaluation of blur-aware 3DGS has not been done (we found none) ✅. This is the project's most defensible novelty, and it satisfies "extend existing work": we extend a benchmark's protocol and findings.
2. **Methodological hygiene in the proposal is good:**
   - a common extractor;
   - no Gaussian centres;
   - marginal (not joint) strata;
   - no ground-truth leakage into training;
   - a null result is acceptable.
3. **Lee & Lee is used correctly**, as the principle rather than a method to re-implement. RealX3D's motion blur makes the link quantitative: blur length ∝ 1/depth.

### What is weak or risky, and the fix

| # | Problem | Why it matters | Fix (adopted in this plan) |
|---|---|---|---|
| W1 | **No sharp control in the analysis design.** | Thin, high-curvature structure is hard to reconstruct *even from sharp images*. A per-stratum error under blur confounds "intrinsically hard" with "damaged by blur". | Core: train vanilla on the pixel-aligned **sharp references** for each scene-condition. The primary quantity is **ΔF_s = F_s(blurred) − F_s(sharp)** per stratum *s*. |
| W2 | **Stage 2 on vanilla 3DGS probably can't recover geometry under uniform blur.** | The photometric target itself lacks the detail. Extra Gaussians in fragile regions will fit the *blur* (fuzz, floaters), not sharp structure. Densification can help only if (a) some views carry the detail (mixed captures), or (b) the forward model explains the blur (blur-aware host). | Keep the vanilla-host experiment (Core: it is cheap and the honest baseline). Add the **mixed protocol** (Core, p = 0.25) and the **blur-aware gsplat host** (Ext). Predicted outcome, stated in advance: little or no gain on vanilla with uniform blur; gain on mixed captures and/or the blur-aware host. Whichever way it goes, it's a finding. |
| W3 | **Novelty of Stage 2 as a rule is thin.** | BSGS already uses depth-conditioned thresholds under motion blur. Pixel-GS/AbsGS already fix the gradient statistic. EntON already uses kNN-eigenvalue neighbourhood features in densification (✅, opposite direction). EA-3DGS uses the exact phrase "structure-aware densification" (✅). | Claim Stage 2 as a **controlled intervention study**: "does fragility-first allocation help *under blur*, measured by laser geometry, at matched primitive budget?" EntON-style *flat-first* (B−) and AbsGS/MCMC are **competing arms**, not citations to wave away. Rename it **fragility-aware** densification. |
| W4 | **"Lower the threshold" improvements can be just "more Gaussians".** | Any rule that lowers thresholds grows the model. | Core control: **budget-matched baselines**. Vanilla ADC with a global threshold set to reach the same final count, plus MCMC with `cap_max` = the same count. A rule only "works" if it beats both. |
| W5 | **Factor (i), per-view sharpness, is a near no-op on standard RealX3D** (your Problem 2). | Same blur for every view within a condition. Defocus is also near-uniform *within* a view (±3–5 % CoC across 3–5 m depths, since the focus plane is far in front). | (a) Per-view reliability is tested where views genuinely differ: the **mixed protocol**. (b) Per-pixel reliability is meaningful **for motion only** (blur ∝ 1/Z; 3–5× variation between 1 m and 5 m). That is rule C, framed honestly as a physically derived relative of BSGS's depth-conditioned threshold. (c) Report the no-op itself on uniform conditions as a sanity result (A ≈ ADC). |
| W6 | **Fragility estimator was vague** (Problem 6). | Not implementable. | Defined in §B7.2: opacity-weighted kNN-PCA (k=16) over Gaussian centres → surface variation + linearity, rank-normalised. Same definitions as the ground-truth curvature axis, so Stage 1 and Stage 2 speak one language. |
| W7 | **Pose frame is unknown** (F4). | If we evaluate in the wrong frame, every geometric number is garbage. | Day-1 check D3 plus validation V3 (GT mesh ray-cast through our cameras must match the provided depth PNGs to < 1 cm median). Nothing in Stage 1 is trusted before V3 passes. |
| W8 | **Completeness on parts of the laser scan no training view saw** is unfair. | It penalises all methods for unobservable regions, which are concentrated in occluded, tight-proximity strata (our key strata!). | Completeness only over GT points visible in ≥ 2 training views (ray-cast/depth-tested). Prediction back-projected only where GT depth is valid. |
| W9 | **0.3 dB reproduction tolerance is unrealistic across codebases.** | gsplat vs the original code, principal-point handling, iterations and antialiasing each move PSNR by tenths of a dB. | Tolerance: **≤ 0.5 dB on the 8-scene mean** and per-scene correlation ρ ≥ 0.8. Any larger gap triggers an investigation (principal point first). |
| W10 | **128-run grid as "guaranteed" was never realistic** for 70–140 person-hours. | | Core = strong conditions only (16 scene-conditions). Mild = Extended. Six methods = Stretch. |
| W11 | **Statistics with n = 8.** | An exact two-sided Wilcoxon on 8 pairs cannot go below p = 0.0078 (all 8 same sign). With 6 held-out scenes the floor is p = 0.031. | Pre-register a *small* set of primary tests. Report effect sizes with bootstrap CIs and sign counts. Use the seed-noise band as the "no difference" zone (§B8). |
| W12 | **H2 as written was vague.** | | Recast as two concrete measurements (§B7.6): **H2a** the learned defocus PSF is near-uniform and its radius matches between mild and strong in the expected direction; **H2b** the learned exposure path length ≈ 2 cm / 6 cm. **H3** becomes the exposure-anchor test (render along the learned trajectory; PSNR should peak at the *end* pose). |

### Does Stage 2 plausibly work on this dataset? Honest odds

| Setting | Likelihood of a clear fragile-stratum gain at matched budget | Why |
|---|---|---|
| Vanilla host, uniform blur | Low (~20 %) | The information isn't in the images. |
| Mixed protocol | Moderate (~40–50 %) | Sharp views carry the detail. The reliability weighting stops blurred views from dominating the statistic, whichever way they distort it. The direction is empirical: blurred views may *suppress* or *inflate* gradients at edges, and H1 instrumentation logs which. |
| Blur-aware host | Moderate (~35 %) | Mostly limited by implementation time. |

A null on vanilla plus a positive in one of the other settings is a good story ("densification is not the bottleneck unless blur is modelled or some sharp evidence exists").

### Does the novelty claim hold?

- **Stage 1: yes, as an evaluation contribution.** It is not a new metric; it is new evidence on a new axis (laser geometry × structure type × blur type × sharp control) for methods that were only scored photometrically.
- **Stage 2: only as framed in W3.** Do not claim "a new densification method". Claim "the first matched-budget, laser-validated test of whether blur-reliability and structural-fragility signals improve primitive allocation under real defocus and rendered motion blur". The **mixed-capture reliability weighting** is, to our search, unexplored (❓; re-search in S0.7 before claiming).

## B2. The end-term showcase, designed first

**One-sentence story (pick the version the data supports):**

- **If Stage 1 is positive:** "Blur doesn't damage 3D Gaussian Splatting uniformly. Measured against a laser scan, it erodes thin and tightly spaced structure first, the image-quality rankings hide this, and allocating primitives by blur reliability and local fragility recovers part of it (or doesn't, and here's why)."
- **If Stage 1 is null:** "On RealX3D, blur's geometric damage is uniform across structure types, so scene-averaged scores are adequate. The densification bottleneck appears only when some views are sharp or blur is modelled."

**Figures and tables that must exist** (each maps to a slide and a report figure):

| ID | Figure | Axes / layout | What a "good" result looks like | Layer |
|---|---|---|---|---|
| **Fig 1** | *Blur is geometric* | (a) Crops: sharp / motion-s / defocus-s. (b) Per-pixel expected motion-blur length b(u)=f·L/Z(u) from laser depth (colour, px). (c) Defocus CoC map (near-flat). | (b) shows 3–5× variation with depth; (c) is flat. Visually motivates "motion ≠ defocus". | Core (done at mid-term) |
| **Fig 2** | *Strata on the laser scan* | One scene's GT point cloud coloured by curvature tertile, and by proximity bin (tight / near / open). Inset histograms. | Strata look physically meaningful: edges and legs = high curvature; shelves and chair-under = tight. | Core |
| **Fig 3** | *Main Stage 1 result* | x = stratum (low/mid/high curvature; open/near/tight proximity), y = F@5 cm. One line per method (vanilla, Deblurring-3DGS, BAGS) + dashed **sharp control**. Panels: motion-s, defocus-s. Error bars = bootstrap 95 % CI over 8 scenes. | Lines **diverge or cross** in hard strata, i.e. the ranking changes. Or they stay parallel (null; still informative). | Core |
| **Fig 4** | *Where blur hurts* | Heatmap: rows = methods, columns = strata, cell = ΔF_s = F(blurred) − F(sharp), annotated with CI and sign count (e.g. "7/8"). | ΔF more negative in tight/high-curvature columns → blur damage is structure-dependent. | Core |
| **Tab 1** | *Photometric reproduction* | Our train/NVS PSNR/SSIM/LPIPS vs RealX3D Table 2 for the Core methods. | Within 0.5 dB on the mean. | Core |
| **Fig 5** | *H1 allocation diagnostic* | (a) Gaussians per m² of GT surface per stratum: sharp vs blurred (vanilla). (b) Per-view densification-gradient magnitude vs view sharpness (mixed protocol). | (a) Fewer Gaussians in fragile strata under blur supports "under-allocation". Equal or more refutes it and Stage 2 pivots (§B9). | Core |
| **Tab 2** | *Stage 2 ablation (held-out 6 scenes)* | Rows: ADC, ADC-budget-matched, AbsGS, MCMC(cap), A, B, B−(flat-first), A+B, [C]. Columns per condition: overall F@5, fragile-stratum F@5, test PSNR, #Gaussians, wins/6. | A or B beats **both** budget-matched baselines in fragile strata without overall regression. Otherwise a clean negative with a mechanism. | Core |
| **Fig 6** | *Qualitative* | (a) GT mesh coloured by completeness error (distance to nearest predicted point, 0–10 cm): vanilla vs best rule. (b) Before/after crops of a thin structure (render + depth). (c) Gaussian-centre density maps. | Visible recovery on legs/edges, or visible failure we can explain. | Core |
| Fig 7 | *Mixed captures* | x = fraction of sharp views p ∈ {0, .1, .25, .5, 1}; y = fragile-stratum F@5; lines ADC vs A. | A rises faster at small p. | Ext |
| Fig 8 | *Exposure anchor (H3)* | x = position t ∈ [0,1] along the learned exposure trajectory; y = train-view PSNR vs reference. | Peak at t = 1 (end), not t = 0.5. Explains the trajectory-method penalty. | Ext |
| Fig 9 | *Densification × blur model* | 2×2: {vanilla, blur-aware host} × {ADC, best rule}, fragile-stratum ΔF. | Rule helps mainly with the blur-aware host. | Ext |

**Demo:** a recorded 45–60 s video, not live. Live demos fail on borrowed laptops.
- Side-by-side fly-through of the same path: sharp control | blurred vanilla | blurred + our rule. Rendered with gsplat's camera-path rendering, or recorded from the SuperSplat web viewer.
- Then a rotating GT mesh coloured by completeness error.
- Embed as an MP4 on one slide (autoplay, muted, with captions).

## B3. Week-by-week plan (today → 27 Nov, worst case)

**Week grid and assumptions.**
- Weeks run Saturday to Friday. Gates fall on Fridays.
- Owners by default: Aditya = GPU/training/baselines/Stage 2 trainer code; Saksham = data, geometry harness, strata, statistics, fragility estimator. Writing is shared.
- ❓ Diwali falls in early November (≈8 Nov 2026; check). W5 is planned lighter.

| Week | Dates | Tasks (IDs → Appendix T) | Owner | Deliverable | Done when… |
|---|---|---|---|---|---|
| **W0** | Thu 8–Fri 9 Oct | M1–M8 (M9 optional) | both + CC | Mid-term talk | Talk given. Notes on prof's feedback in `docs/decisions.md` within 24 h. |
| **W1** | 10–16 Oct | S0.1 HPC recon; S0.2 envs + SLURM; S0.3 full data to HPC; S0.4 dataset checks D1–D8; S0.5 run registry; S0.6 fixture; S0.7 related-work re-check + prereg skeleton; S1.1–S1.4 harness modules | A: S0.1–S0.3, S0.5. S: S0.4, S0.6, S1.1–S1.4. Both: S0.7 | `docs/hpc.md`, `docs/dataset_notes.md`, envs, `data_fixture/` | **Gate G0 (Wed 14 Oct):** vanilla gsplat 30k completes on HPC (or fallback chosen) **and** V3 (frame check) passes on ≥ 2 scenes. |
| **W2** | 17–23 Oct | S1.5 final strata (8 scenes); S1.6 metrics; S1.7 validation V0–V4; S1.8 run E02/E03/E04; S1.9 build Deblurring-3DGS + BAGS (start E05/E06); S1.12 prereg finalised | S: S1.5–S1.7, S1.12. A: S1.8, S1.9 | Validated harness; vanilla + sharp results on 16 scene-conditions | **Gate G1 (Fri 23 Oct):** harness passes V0–V3; ≥ 1 native blur-aware baseline trains on RealX3D. If neither builds → §B9 R2. |
| **W3** | 24–30 Oct | Finish E05/E06; S1.10 H1 density + gradient logging; S1.11 Stage 1 stats + Figs 3–5 draft; S2.1–S2.3 Stage 2 code (A, B, B−, budget-match, reliability, mixed-protocol data) | A: E05/E06, S2.1. S: S1.10, S1.11, S2.2. Both: S2.3 | Stage 1 Core result (Figs 3–5, Tab 1 draft) | **Gate G2 (Fri 30 Oct):** Fig 3 and Fig 4 exist with CIs. **Decision recorded**: Stage 2 direction (fragile-first, flat-first, or reliability-only) per §B9 tree. |
| **W4** | 31 Oct–6 Nov | S2.4 dev sweep E07 (2 dev scenes); S2.5 mixed protocol E09; [Ext] S2.8 blur-aware host code; [Ext] S2.7 rule C | A: S2.4, S2.8. S: S2.5, S2.7 | Chosen hyperparameters frozen in `configs/stage2_final.yaml` | **Gate G3 (Fri 6 Nov):** one config per rule frozen from dev scenes only. No held-out scene has been looked at. |
| **W5** | 7–13 Nov (lighter) | S2.6 final held-out E08; [Ext] E10 mild runs; [Ext] E11 host runs; [Ext] S2.9 H3 | A: S2.6, E11. S: E10 queueing, S2.9 analysis | Tab 2 numbers | All Core runs finished or queued with ETA < 4 days. |
| **W6** | 14–20 Nov | S2.10 Stage 2 analysis + Fig 6; Ext analyses (Figs 7–9); D1 repo cleanup; D2 report full draft; D4 record demo | S: S2.10, D2 (§4–5). A: D1, D4, D2 (§3). | Report draft v1, all figures | **Gate G4 (Wed 18 Nov): experiment freeze.** After this only re-plots and reruns of crashed jobs. |
| **W7** | 21–27 Nov | D2 final; D3 slides; D4 video edit; D5 archive; rehearsals Tue 24 + Thu 26 | both | Final submission set | Two timed rehearsals ≤ 11:30; repo README reproduces Fig 3 from `results/` CSVs. |

**If the presentation is later than 27 Nov:** the extra days go to polishing and *pre-listed* Extended analyses that need no new GPU runs. Do not start new experiment families after G4.

## B4. Experiment matrix

**Unit cost estimates (❓, to be replaced by measured values in W1–W2).** Quarter-res ≈1800×1200, ~23–31 training views, on an RTX 6000-class or H200 GPU. Colab/Kaggle T4 is ≈3–4× slower.

| Method / host | GPU-h per run (budget) | Peak VRAM (est.) |
|---|---|---|
| gsplat vanilla 30k (ADC / AbsGS / MCMC / our rules) | 0.5 | 6–14 GB |
| gsplat + our rules incl. kNN fragility (CPU cKDTree every 500 its) | 0.6 | 6–14 GB |
| gsplat blur-aware host, motion (8 virtual views) | 1.2 | 15–30 GB |
| gsplat blur-aware host, defocus (global PSF) | 0.6 | 8–16 GB |
| Deblurring-3DGS (native) | 1.0 (defocus) / 1.5 (motion variant) | 10–24 GB |
| BAGS (native, 46k its, multi-scale kernels) | 1.5 | 12–24 GB |
| BAD-Gaussians / CoCoGaussian / Deblur-GS | 1.5 / 2.0 / 1.5 | up to 40 GB |
| Depth export (all views) | 0.05 | — |

**Matrix.** S = scene-conditions. "Strong" = {motion-s, defocus-s}.

| ID | Priority | Experiment | Runs | GPU-h | GPU | Answers |
|---|---|---|---|---|---|---|
| E01 | Must | Smoke + repro check: vanilla, 1 scene × strong | 2 | 1 | T4/any | pipeline |
| E02 | Must | Vanilla gsplat, 8 scenes × strong | 16 | 8 | any (pack 2–3/GPU on H200) | Q1, repro |
| E03 | Must | **Sharp control**: vanilla on sharp references, 8 scenes × {motion, defocus} sessions (8 runs if D4 shows the references are identical across conditions) | 8–16 | 4–8 | any | Q1 (ΔF) |
| E04 | Must | Seed noise: vanilla, 2 more seeds × 2 dev scenes × strong | 8 | 4 | any | noise band |
| E05 | Must | Deblurring-3DGS native, 8 × strong | 16 | 20 | RTX 6000 | Q1 |
| E06 | Must | BAGS native, 8 × strong | 16 | 24 | RTX 6000 | Q1 |
| E07 | Must | Stage 2 dev sweep (vanilla host): 2 dev scenes × strong × ~10 configs | ~40 | 22 | any | Q2 tuning |
| E08 | Must | Stage 2 final (vanilla host), 6 held-out × strong × {ADC, ADC-bm, AbsGS, MCMC-bm, B, B−, A+B} | 84 | 45 | any | Q2, H1 |
| E09 | Must | Mixed protocol p = 0.25, 8 × strong × {ADC, A} (dev scenes tune A's γ inside E07) | 32 | 16 | any | Q2 (factor i) |
| E10 | Should | Mild: vanilla + sharp-ctrl reuse + Deblurring-3DGS + BAGS, 8 × mild | 48 | 45 | mixed | severity trend |
| E11 | Should | Blur-aware host 8 × strong, + best rule on host | 32 | 30 | RTX 6000/H200 | H1 (interaction), Q2 |
| E12 | Should | H3 exposure anchor: motion host, anchor ∈ {start, mid, end} × 2 dev scenes × {mild, strong} | 12 | 15 | any | H3 |
| E13 | Should | Mixed sweep p ∈ {0.1, 0.5} (0, 0.25, 1 reused) × {ADC, A} × 4 scenes × motion-s | 16 | 8 | any | Fig 7 |
| E14 | Should | Rule C (motion only), 6 held-out × {mild, strong} | 12 | 8 | any | factor i per-pixel |
| E15 | Should | Principal-point bias: vanilla gsplat full-K vs centre-cropped, 2 scenes × strong | 4 | 2 | any | protocol bias (F12) |
| E16 | Could | BAD-Gaussians, CoCoGaussian, Deblur-GS × 8 × strong | 48 | 80 | RTX 6000/H200 | full grid |
| E17 | Could | Same × mild | 48 | 80 | | full grid |
| E18 | Could | H2 parameter recovery (analysis of E11/E12 outputs) | 0 | 0 | CPU | H2 |
| E19 | Could | Median-depth extractor (RaDe-GS/GOF rasterizer) re-render of Core models | 0 train | 5 | any | extractor sensitivity |

**Totals:**
- Must ≈ **140 GPU-h**
- Should ≈ **110 GPU-h**
- Could ≈ **165 GPU-h**

**Does it fit?** ❓ We don't know the queue share. Six weeks ≈ 1000 wall-clock hours.
- If we average **one GPU ~30 % of the time** (≈300 GPU-h), Must + Should fits.
- Packing 2–3 vanilla jobs per H200 (141 GB) roughly doubles throughput for the cheap runs.
- Overflow lanes:
  - **Kaggle** gives ~30 GPU-h/week free 🔶 (T4×2 or P100) and suits vanilla/rule runs;
  - Colab;
  - the on-request workstation for the native baselines.
- Run the Must list in this order: E02, E03, E04, E05/E06 (start builds early), E07, E09, E08.

## B5. Day-1 / week-1 technical checklist

### B5.1 Dataset download and verification (S0.3, S0.4)

**Download to HPC scratch** (login node; compute nodes often have no internet ❓):
- the 4 `data_4` blur tars, ≈3.4 GB;
- the 8 blur scenes' `pointclouds/*.tar.gz`, ≈2.5 GB.

Record SHA256 checksums in `docs/dataset_notes.md`. Laptop gets the same set for CPU work if disk allows; otherwise strong-only.

**Checks D1–D8** (script `scripts/verify_dataset.py`, task S0.4). All answers go in `docs/dataset_notes.md`.

| ID | Question | How | Decides |
|---|---|---|---|
| D1 | Which 8 scenes are in the blur tars? Do all 8 appear in **all four** blur tars? | List the tar contents (scene folder names) per condition. | Grid size (16 or fewer scene-conditions per severity). |
| D2 | What is `val/`? Pixel-aligned sharp references of `train/`? | Same file names? Mean abs diff vs blurred ≫ diff vs a shifted copy? Variance of the Laplacian higher? | Sharp control (E03) and train-view PSNR. |
| D3 | Which frame are the poses in? | Compare camera centres from `transforms_train.json` with COLMAP `sparse/` (images.bin/txt). Fit a similarity (Umeyama). Identity → COLMAP frame; scale ≈ 4/avg-dist → colmap2nerf-normalised. Also check whether `colmap2world` includes scale (cube root of the rotation block's determinant). | `T_train→world` per scene (saved to `frame.json`). |
| D4 | Are sharp references identical across the four blur conditions of a scene? | Hash or compare the `val/` images across tars. | E03 = 8 or 16 runs. |
| D5 | Principal-point offset and fx/fy per scene at quarter res. | Read the intrinsics; report `cx − w/2`, `cy − h/2`, `fy/fx`. | Need for centre-crop (F12). |
| D6 | `sharpness` variation **within** a condition, and vs our own quarter-res variance-of-Laplacian. | Stats per split/condition; scatter plot. | Confirms Problem 2 (no within-condition variation). Calibrates the reliability signal for the mixed protocol. |
| D7 | GT depth PNGs: resolution, which frames (train/val/test?), unit (expect mm), invalid value (0?). | Read a few; compare with a ray-cast of `cull_mesh.ply` through the corresponding camera. | V3 frame validation. |
| D8 | Test split: blurred or sharp inputs? Same views across conditions? | Inspect. | NVS metrics protocol. |

**Also look at:** the mild/strong motion path in the images (the 5 cm vs 6 cm discrepancy). Estimate the blur length from an edge spread at a known depth (Ext, H2b).

### B5.2 Which codebase to standardise on

**Decision:** two families, with one common evaluation path.

1. **Our code (vanilla host, Stage 2 rules, blur-aware hosts, depth export) uses gsplat 1.5.3.**
   - Why: pip-installable; handles full K (principal point, fx ≠ fy); has `absgrad` (AbsGS) and MCMC built in, so two competing arms come for free; modern CUDA; Apache-2.0.
   - Pin 1.5.3: `main`/v1.6 requires PyTorch ≥ 2.7.
2. **External baselines run in their native repos**, re-pinned, for fidelity with RealX3D's numbers.
   - Porting Deblurring-3DGS or BAGS into gsplat would produce "our version of their method", which examiners rightly distrust.
   - From each we **only import the final Gaussians** (`point_cloud.ply`; for BAGS, apply Mip-Splatting's 3D filter to scale/opacity). Then gsplat renders depth for every method: **one renderer, one extractor**.
3. **Fallback after G1:** if a native baseline won't build, implement "-lite" versions in gsplat and label them as ours ("Deblur-lite"). They are not compared with RealX3D numbers.

### B5.3 Environments per GPU type (S0.2)

**First command on every machine:**

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
```

**What "RTX 6000 (48 GB)" can mean:** 48 GB matches either the **RTX A6000** (Ampere, sm_86) or the **RTX 6000 Ada** (sm_89). The older Quadro RTX 6000 has 24 GB, and the RTX PRO 6000 Blackwell has 96 GB 🔶. So set `TORCH_CUDA_ARCH_LIST` from the output above, not from the name.

| GPU | Compute capability | Minimum CUDA to compile |
|---|---|---|
| T4 | sm_75 | any 11.x |
| A6000 | sm_86 | 11.1 |
| RTX 6000 Ada | sm_89 | 11.8 |
| H200 | sm_90 | 11.8 |
| RTX 50-series | sm_120 | 12.8, with PyTorch ≥ 2.7 |

(All 🔶.)

**The environments:**

| Env | Contents | Builds for |
|---|---|---|
| `cvp-gsplat` (main) | Python 3.10, PyTorch 2.4.1+cu121 (or the HPC module's CUDA 12.x), `gsplat==1.5.3` built from source on a GPU node with `TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"`, plus `env/requirements-train.txt` (numpy<2, tyro, imageio, torchmetrics, lpips, scipy, opencv, plyfile, tensorboard). | T4, A6000, Ada, H200 |
| `cvp-legacy-<method>` (one **per** Inria-derived repo) | Python 3.10, PyTorch 2.1.2+cu118, `cuda-toolkit 11.8` (conda `nvidia/label/cuda-11.8.0`), numpy<2, the repo's own requirements. Build its `submodules/*` with `TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"`. **Separate envs** because each fork installs a package named `diff_gaussian_rasterization` and they collide. Use `conda create --clone` from one base. Common patches (keep under `external/patches/`): add `#include <cstdint>` in `cuda_rasterizer/rasterizer_impl.h` and `#include <cfloat>` in `simple-knn/simple_knn.cu` for newer GCC 🔶; replace removed NumPy aliases. | A6000, Ada, H200 |
| `cvp-ns` (BAD-Gaussians, Str) | As the BAD-Gaussians README: nerfstudio 1.0.3, PyTorch 2.1.2+cu118, tiny-cuda-nn bindings (slow, fragile build). | Ada/H200 |
| `cvp-blackwell` (opportunistic) | PyTorch ≥ 2.7 cu128, gsplat built with `TORCH_CUDA_ARCH_LIST="12.0"`. Only for vanilla/rule runs. Never for legacy repos. | RTX 5060/5090 |
| `cvp-cpu` (laptop/cloud) | Python 3.11 (Open3D wheels lag the newest Python 🔶), numpy, scipy, open3d, trimesh, opencv-python, matplotlib, pandas, plyfile, pytest. | — |

**Containers.** Use conda first. If the HPC has Apptainer/Singularity and module conflicts appear, build a SIF from `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel` (legacy) or `…2.4.1-cuda12.1-cudnn9-devel` (gsplat); recipes go in `env/apptainer/`.

**Build on a GPU node** (an interactive `srun --gres=gpu:1 --pty bash`), so the CUDA arch and driver match.

### B5.4 Repository structure (create in M1/S0.5)

```
cvp_project/
├── README.md                  # what, how to reproduce Fig 3/Tab 2 from results/
├── docs/
│   ├── PLAN.md                # this file
│   ├── STATUS.md              # task checklist (ID, status, date, who, link to PR/commit)
│   ├── decisions.md           # dated decisions + gate outcomes + prof feedback
│   ├── dataset_notes.md       # D1–D8 answers, checksums, scene list, dev/held-out split
│   ├── hpc.md                 # partitions, GPU types, limits, module names, storage paths
│   ├── prereg.md              # primary metrics/tests fixed before looking at results
│   └── refs/                  # proposal, slides, Lee&Lee PDF (current docs/ files move here)
├── env/  requirements-cpu.txt  requirements-train.txt  cvp-gsplat.yml  legacy.yml  apptainer/
├── cvp/                       # python package
│   ├── data/      realx3d.py  frames.py  mixed.py          # loader, frame transforms, mixed-protocol builder
│   ├── train/     trainer.py  strategies.py  fragility.py  reliability.py  blur_models.py
│   ├── io/        gaussians.py  colmap_export.py           # load any method's ply; export COLMAP layout for native repos
│   ├── geometry/  render_depth.py  fuse.py  visibility.py  strata.py  metrics.py  align.py  labels2d.py
│   └── analysis/  stats.py  tables.py  figures.py
├── scripts/       inspect_scene.py  verify_dataset.py  make_fixture.py  train_gsplat.py  export_depth.py
│                  eval_geometry.py  make_manifest.py  aggregate.py  blur_maps.py  strata_preview.py  gsplat_smoke.py
├── configs/       base.yaml  experiments/E02.yaml …  stage2_final.yaml  scenes.yaml (dev/held-out)
├── slurm/         train_array.sbatch  legacy_array.sbatch  eval_cpu_array.sbatch  interactive.md
├── external/      README.md (pinned commits)  patches/*.patch  clone_baselines.sh   # repos cloned here, gitignored
├── notebooks/     colab_midterm.ipynb  kaggle_runner.ipynb
├── data_fixture/  # ≤15 MB real-format mini scene (committed)
├── tests/         test_*.py (pytest; GPU tests marked @pytest.mark.gpu)
├── results/       registry.csv  stage1/*.csv  stage2/*.csv  figures/*.pdf   # small, committed
├── report/        IEEE LaTeX (main.tex, figs/)          slides/
└── .gitignore     # data/, runs/, external/*/, *.ply (except data_fixture), *.npz, *.tar.gz
```

**Run-directory convention** (on HPC scratch, never in git):

```
runs/<exp_id>/<scene>/<condition>/<method>[_<rule>]/s<seed>/
    config.yaml   git_commit.txt   env.txt   frame.json   train.log   tb/
    point_cloud.ply   metrics_photo.json   depth/{train,test}/*.npz (ED, var, alpha)
    geometry/metrics_3d.json   geometry/metrics_25d.json   stats_gaussians.json
```

### B5.5 Logging and experiment tracking

- **Primary:** every run writes its `metrics_*.json`. `scripts/aggregate.py` appends one row per run to **`results/registry.csv`**:
  - columns: exp_id, scene, cond, method, rule, seed, git, gpu, hours, n_gauss, psnr_train, psnr_test, F5_all, F5_by_stratum…
  - The registry is committed. All figures are generated from it, so the report is reproducible from git.
- **TensorBoard** for curves (it works offline on compute nodes).
- **W&B is optional.** It needs internet on compute nodes, or offline sync. Not worth the setup unless it already works.

### B5.6 SLURM templates (S0.2) ❓ assuming SLURM; adapt names after S0.1

```bash
#!/bin/bash
# slurm/train_array.sbatch   usage: sbatch --array=1-$(($(wc -l < M.csv)-1))%4 slurm/train_array.sbatch M.csv
#SBATCH --job-name=cvp-train
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
MANIFEST=$1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cvp-gsplat
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")      # header on line 1
python -m cvp.train.trainer --manifest_row "$ROW" && python scripts/export_depth.py --manifest_row "$ROW"
```

**Variants:**
- `legacy_array.sbatch`: same, but activates `cvp-legacy-<method>` from a manifest column and calls `external/<method>/train.py` through `cvp/io/colmap_export.py`-prepared data.
- `eval_cpu_array.sbatch`: no GPU, `--cpus-per-task=16`, runs `scripts/eval_geometry.py`.

**Packing on an H200:** a `--pack N` option in the trainer wrapper launches N manifest rows as background processes on the same GPU and `wait`s for them. Use it only for vanilla/rule runs.

## B6. Exact Stage 1 protocol

### B6.1 Inputs and frames

- **Per scene:** `T_train→world` (4×4 similarity) = `colmap2world · T_train→colmap` (from D3). Saved to `frame.json`.
- Every method's Gaussians stay in the frame they were trained in. The extractor transforms the *fused points* to world.
- **No per-method ICP.**

### B6.2 Geometry extraction (common to all methods)

1. **Load Gaussians.** `cvp/io/gaussians.py` reads the standard 3DGS ply (xyz, f_dc, f_rest, opacity, scale, rot).
   - For Mip-Splatting/BAGS, fold the 3D filter in: s′² = s² + f3d², opacity′ = opacity·√(∏s²/∏s′²).
   - For nerfstudio exports, map the field names.
2. **Render with gsplat** at the data_4 resolution and full K, for **all training views** (primary) and test views (2.5D NVS metric).
   - Outputs per pixel: expected depth Ê[z] (`RGB+ED`), alpha, and **depth variance**. For the variance, render colours = [z, z²] as 2-channel features: Var = E[z²] − E[z]².
3. **Pixel filter.** Keep pixel u if all of these hold:
   - alpha ≥ 0.5;
   - √Var(u) / Ê[z](u) ≤ 0.05 (rejects pixels blending two surfaces, the floater source);
   - the GT depth at u is valid (> 0). This keeps both prediction and GT inside the laser-observed region.
4. **Back-project** to 3D, then transform with `T_train→world`.
5. **Multi-view consistency.** Keep a point if, reprojected into ≥ 2 other training views, it agrees with that view's filtered depth within max(1 cm, 1 % · z).
6. **Voxel-downsample to 5 mm** (the GT spacing). Cap at 5 M points.
7. **Sensitivity variants** (report in an appendix table, Core but cheap):
   - variance ratio {0.02, 0.1, ∞ (off)};
   - consistency off;
   - **TSDF** (Open3D ScalableTSDFVolume, voxel 1 cm, truncation 4 cm, sampled at 5 mm). TSDF erodes structures thinner than ~2 voxels, so it is a *conservative* check, not primary.
   - [Str] median depth via a RaDe-GS/GOF rasterizer.
8. **Extractor validity check:** feed the **provided GT depth PNGs** through steps 3–6 (V2). The result must score ≈ perfect.

### B6.3 Ground-truth preparation

1. **GT points:** `cull_pointcloud.ply` (5 mm). Normals from `cull_mesh.ply` vertex normals, transferred to the points by nearest vertex and oriented towards the nearest training camera.
2. **Visibility:** a GT point is *observed* if it projects inside ≥ 2 training images with |z_proj − D_gt(u)| < 1 cm (using the provided depth or a ray-cast of the mesh).
   - Completeness uses only observed points.
   - Report the observed fraction per stratum: it is itself informative (tight strata are less observed).
3. **Distances:**
   - Accuracy uses point→**mesh** distance (Open3D `RaycastingScene.compute_distance`, exact and fast).
   - Completeness uses GT point → nearest predicted point (KD-tree).

### B6.4 Stratification

All axes are computed on observed GT points, once per scene; labels are cached.

**Axis 1: curvature (surface variation).**
- Formula: σ = λ0/(λ0+λ1+λ2) from the PCA of neighbours within radius r. Primary **r = 2 cm** (≈50 points at 5 mm spacing); stability **r = 4 cm**.
- **Bins:** tertiles **pooled across the 8 scenes** (fixed thresholds, so "high curvature" means the same thing everywhere). Record the per-scene shares.
- **Stability check:**
  - Spearman ρ(σ_2cm, σ_4cm);
  - Cohen's κ between the two tertile labelings, accepted if κ ≥ 0.4.
  - If κ < 0.4, use the multi-scale mean σ̄ = ½(σ_2 + σ_4) and say so.
  - Laser noise (4 mm) inflates σ on flat surfaces at small r, which is why r < 2 cm is not used.

**Axis 2: proximity / thickness.** This replaces the SDF with a version that needs no watertight mesh.
- From p + ε·n (ε = 1 cm), cast 5 rays in a 15° cone around **+n**; the median hit distance is d_front (clearance to the facing surface).
- From p − ε·n, cast around **−n**; the median is d_back (local thickness through the object).
- A miss within 1 m → ∞.
- Proximity **d = min(d_front, d_back)**.
- **Bins, physical:** tight < 5 cm, near 5–20 cm, open ≥ 20 cm.
  - If any bin holds < 5 % of observed points pooled, fall back to pooled tertiles.
  - Report thickness (d_back finite and < 5 cm, i.e. "thin") as an optional third axis.

**Joint analysis:** none. Marginal analysis only, as proposed.

**Label transfer:**
- *Prediction points* take the stratum of their nearest GT point, if within 20 cm. Otherwise they are "unassigned outliers", which count in overall precision and are reported as outlier mass.
- *2D label maps*: ray-cast the mesh per view, take the hit triangle, then the majority label of its vertices (labels transferred to vertices by nearest GT point).

### B6.5 Metrics

**3D (primary), per stratum s and overall, at τ ∈ {2, 5, 10} cm:**
- Precision_s(τ) = fraction of predicted points assigned to s within τ of the mesh.
- Recall_s(τ) = fraction of observed GT points in s within τ of the prediction.
- F_s = 2PR/(P+R).
- Also mean and median accuracy and completeness in cm, clipped at 20 cm.
- **Primary metric = F@5 cm** (RealX3D's threshold). The 2 cm figure is near the 1.2 cm registration floor: report it, don't headline it.

**2.5D (secondary, extractor-light), per stratum on training and test views:**
- median |Ẑ − Z_gt|;
- % of pixels within 2/5/10 cm;
- computed over pixels with alpha ≥ 0.5 and valid GT, using 2D label maps.
- No median scaling: we are metric thanks to `colmap2world`. Report RealX3D's median-scaled depth L1 too, for comparability.

**Allocation (H1):**
- Gaussians per m² of observed GT surface per stratum: count Gaussian centres within 2 cm of GT points in s, divided by the area proxy (number of GT points in s × 25 mm²).
- Mean opacity and size per stratum.

**Photometric:** PSNR/SSIM/LPIPS train-view (vs references) and NVS, exactly as RealX3D §5.1.

**Sensitivity (appendix):**
- ICP-refined alignment: point-to-plane, run only on "low-curvature + open + high-visibility" points, max correspondence 5 cm. Report the ICP correction magnitude per method; a large correction = systematic drift (links to H3).
- Extractor variants (B6.2.7).

### B6.6 Primary quantities for Q1

- **Q1-rank:** per condition and stratum, the method ranking by F_s@5. Per bootstrap resample over scenes, record the probability that each method ranks first.
- **Q1-interaction** (the formal test): for methods A, B and per scene, I = (F_A − F_B)_hard − (F_A − F_B)_easy, where hard/easy = the top/bottom stratum of an axis. Wilcoxon on I across the 8 scenes.
- **Blur damage:** ΔF_s = F_s(blurred) − F_s(sharp control), per method; is it more negative in hard strata?

### B6.7 Validating the harness before trusting it (S1.7)

| ID | Test | Pass criterion |
|---|---|---|
| V0 | Unit tests on synthetic shapes: plane (σ ≈ 0), sphere radius R (σ in the analytic range), two parallel plates 3 cm apart (d_front ≈ 3 cm), 2 cm-thick slab (d_back ≈ 2 cm). | All pass in CI (`pytest`). |
| V1 | GT self-test: prediction = GT + N(0, 0.5 cm). Then delete a 30 cm patch. | F@5 ≥ 0.99. After deletion, completeness drops by exactly the deleted share in each stratum (±1 %). |
| V2 | GT depth through the extractor (B6.2.8). | F@5 ≥ 0.98, F@2 ≥ 0.90 ❓ (provisional; set from the first scene). |
| V3 | Frame check: ray-cast GT mesh through our cameras vs the provided depth PNG. | Median abs diff < 1 cm on ≥ 90 % of valid pixels, every scene. |
| V4 | Sanity ordering: vanilla on sharp > vanilla on strong blur in overall F@5, in ≥ 6/8 scenes. | If violated, investigate before analysis. |
| V5 [Str] | External anchor: run one feed-forward model from RealX3D's geometry table (e.g. VGGT) through our metric. | Within a few points of their reported F1. |

## B7. Stage 2 design in implementable detail

### B7.1 Where the code goes

`cvp/train/strategies.py` defines `FragilityStrategy(DefaultStrategy)` for gsplat 1.5.3.

The gsplat names below are 🔶 from the 1.5 API: `DefaultStrategy`, `grow_grad2d`, `state["grad2d"]`, `state["count"]`, `step_post_backward`, `_update_state`, `_grow_gs`. **Check them against the installed source** at the start of S2.1.

### B7.2 The training-time fragility estimator (resolves Problem 6)

```python
# cvp/train/fragility.py  (CPU; called every K_frag = 500 iterations during the refine window)
def fragility(means: np.ndarray, opacity: np.ndarray, k: int = 16):
    tree = cKDTree(means)                               # 1–3 M points: ~1–5 s with workers=-1
    _, idx = tree.query(means, k=k + 1, workers=-1)
    nb, w = means[idx[:, 1:]], opacity[idx[:, 1:]]      # (N,k,3), (N,k)   opacity-weighted neighbours
    mu = (w[..., None] * nb).sum(1) / w.sum(1, keepdims=True)
    X = nb - mu[:, None]
    C = np.einsum('nk,nki,nkj->nij', w, X, X) / w.sum(1)[:, None, None]
    lam = np.linalg.eigvalsh(C)                         # ascending λ0 ≤ λ1 ≤ λ2   (chunk over N to cap RAM)
    sv  = lam[:, 0] / lam.sum(1)                        # surface variation  → corners/edges/high curvature
    lin = (lam[:, 2] - lam[:, 1]) / lam[:, 2]           # linearity          → thin, rod-like structure
    r = lambda x: rankdata(x) / len(x)                  # percentile rank in [0,1], per scene/iteration
    return np.maximum(r(sv), r(lin)), r(sv)             # (state["frag"], state["sv_rank"]); B− uses flatness = 1 − sv_rank
```

- Gaussians born between updates inherit their parent's *f*.
- Optional extra signal (Ext): rendered-depth discontinuity. Mark Gaussians whose projected centre lies within 2 px of a large |∇Ẑ| edge.
- This uses **only the current reconstruction**: no ground-truth leakage.

### B7.3 The reliability signal (resolves Problem 2)

**Per-view r_v (rule A).**
- Compute variance of the Laplacian of view v at training resolution: s_v.
- Normalise: s̃_v = s_v / median_{v∈scene}(s_v).
- Weight: w_v = clip(s̃_v^γ, w_min, w_max), with w_min = 0.05, w_max = 4.
- Variance of the Laplacian is texture-dependent. That is acceptable here because the mixed protocol compares *the same view's* sharp and blurred versions across runs, and views of one scene share content. As an Ext check, compare against a no-reference blur metric (high-frequency energy ratio).

**Per-pixel w(u) (rule C, motion only).**
- Expected blur extent b̂(u) = f·L̂·|sin θ| / Ẑ(u). Ẑ is the current rendered depth; L̂ is the exposure path length; θ is the angle between the path and the viewing ray (≈90° on a sideways dolly ❓, check in D-checks).
- Weight: w(u) = 1 / (1 + (b̂(u)/b0)²), with b0 ∈ {8, 16} px.
- **Honesty note:** on the vanilla host L̂ is a hyperparameter (2 or 6 cm, i.e. benchmark knowledge). That makes rule C a physically derived cousin of BSGS's depth-conditioned threshold, and we say so. On the **blur-aware host** L̂ comes from the *learned* trajectory, which is legitimate.

**Mixed protocol (data side, `cvp/data/mixed.py`).** For fraction p, replace a deterministic, seed-fixed subset of ⌈p·N_train⌉ training images with their sharp references (D2). Poses are unchanged (motion: the reference pose = the exposure end pose, consistent with F5). Test views are untouched.

### B7.4 Candidate rules (pseudo-code for the modified ADC step)

```python
class FragilityStrategy(DefaultStrategy):
    rule: str = "adc"          # adc | A | B | Bminus | AB | C | CB
    beta: float = 0.5          # threshold reduction strength (B, B−)
    gamma: float = 1.0         # reliability exponent (A)
    q_gate: float | None = None  # optional: only modulate top-(1-q) fragile Gaussians

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # info["view_w"] (scalar per camera) set by the trainer for rules A/AB; default 1.0
        # info["means2d_grad_C"] set by the trainer for rule C (grad of pixel-weighted loss)
        super().step_post_backward(params, optimizers, state, step, info, packed)

    def _update_state(self, params, state, info, packed=False):
        g = info["means2d"].absgrad if self.absgrad else info["means2d"].grad   # (C,N,2) or packed
        if self.rule in ("C", "CB"):
            g = info["means2d_grad_C"]                                          # replaces the statistic only
        g = normalise_to_ndc(g, info)                                           # as gsplat does (×W/2, ×H/2)
        w = info.get("view_w", 1.0)                                             # rule A: per-view weight
        vis = info["radii"] > 0
        state["grad2d"][gs_ids(vis)] += w * g.norm(dim=-1)[vis]                 # weighted sum of |g|
        state["count"][gs_ids(vis)]  += w                                       # weighted visibility count

    def _grow_gs(self, params, optimizers, state, step):
        gbar = state["grad2d"] / state["count"].clamp_min(1e-6)
        if self.rule in ("B", "AB", "CB", "Bminus"):
            f = state["frag"] if self.rule != "Bminus" else 1.0 - state["sv_rank"]  # B−: flatness; refreshed every 500 its
            mod = f if self.q_gate is None else (f >= self.q_gate).float()
            tau = self.grow_grad2d * (1.0 - self.beta * mod)                       # per-Gaussian threshold
        else:
            tau = self.grow_grad2d
        is_grad_high = gbar > tau
        # unchanged below: clone if small (scale ≤ grow_scale3d·scene_scale), split if large; prune by opacity
        ...
```

| Rule | What changes | Factor |
|---|---|---|
| `adc` | nothing (gsplat default, τ0 = 2e-4) | baseline |
| `adc-bm` | global τ chosen so the final #Gaussians matches the rule being compared (±5 %); pick from τ0×{1, .8, .65, .5} | budget control |
| `absgs` | `absgrad=True`, τ0 = 8e-4 (gsplat's recommended pairing 🔶) | competitor |
| `mcmc-bm` | gsplat MCMC, `cap_max` = the rule's final count | competitor |
| `A` | reliability-weighted statistic | (i) per view |
| `B` | fragility-lowered threshold | (ii) |
| `Bminus` | **flat-first** threshold (EntON-like direction) | converse hypothesis |
| `AB`, `CB` | combinations | (i)+(ii) |
| `C` (Ext) | per-pixel blur-extent-weighted statistic: `torch.autograd.grad((w*ℓ).sum(), means2d, retain_graph=True)` (≈ +30 % step time) | (i) per pixel |
| `A-loss` (Ext) | also weights the photometric loss by w_v; separates "statistic" from "optimisation" effects | control |

### B7.5 Hyperparameters to sweep (dev scenes only, E07)

- **B:** β ∈ {0.3, 0.5, 0.7}; q_gate ∈ {None, 0.67, 0.9}; k ∈ {16} (32 only if noisy).
- **A:** γ ∈ {1, 2} (on mixed p = 0.25).
- **C:** b0 ∈ {8, 16} px.
- **Fixed:** refine window (500–15 000), K_frag = 500, 30 k iterations.

That is ≈ 10 configurations × 2 dev scenes × 2 conditions ≈ 40 runs. Selection criterion is pre-registered: the best fragile-stratum F@5 subject to the no-regression criterion on dev scenes.

### B7.6 Blur-aware host and the H2/H3 tests (Ext, S2.8–S2.9)

**Motion host (BAD-Gaussians idea, re-implemented in gsplat).**
- Each training image gets a learnable se(3) offset ξ_v for one exposure endpoint. The other endpoint is the provided pose (anchor).
- Render N = 8 virtual poses (linear interpolation in se(3)) as one batched gsplat call, average them, compare with the blurred image.
- ADC statistic: average over virtual views (scale the threshold accordingly, as BAD-Gaussians does).
- **Anchor options:** {start, mid, end}.
- **H3:** with anchor = mid and ξ free, render at t ∈ [0, 1] after training. If the sharp reference corresponds to t = 1 (as F5 implies), PSNR vs the reference peaks at t ≈ 1, and the learned midpoint sits ≈ L/2 from the provided pose.
- **H2b:** learned path length vs 2/6 cm.

**Defocus host.**
- The rendered image is convolved with a learnable, normalised PSF K = softmax(logits), 31×31, shared per scene (justified by the near-uniform CoC; a per-view variant is optional).
- Evaluate sharp renders without K.
- **H2a:** effective PSF radius larger for strong than mild, and near-constant across views.

### B7.7 Ablation ladder and no-regression criterion

**Ladder** (each step adds one thing): `adc` → `adc-bm` → `A` (mixed only) / `B` → `AB` → [`C`, `CB` motion] → [the same on the blur-aware host]. Competitors `absgs`, `mcmc-bm` and `Bminus` sit beside the ladder.

**No-regression criterion.** On held-out scenes, per condition, against `adc`:
- mean ΔF@5 overall ≥ −max(0.01, 2σ_seed);
- mean Δ test PSNR ≥ −0.2 dB;
- #Gaussians reported. If > 1.5× `adc`, flag it.

**Efficacy criterion** (pre-registered): fragile-stratum ΔF@5 > 2σ_seed vs **both** `adc-bm` and `mcmc-bm`, positive in ≥ 5/6 held-out scenes.

**Dev / held-out split:** 2 dev scenes (most thin structure, chosen in S0.4 *before* any Stage 2 run) and 6 held-out. Record it in `configs/scenes.yaml` and `docs/prereg.md`.

## B8. Analysis and statistics plan (8 scenes)

1. **Unit of analysis = scene.** Conditions are analysed separately (motion-s and defocus-s answer different physical questions; never pool across them). Mild results are reported in parallel, not pooled.
2. **Effect sizes first:** mean paired difference, 95 % bootstrap CI over scenes (10 000 resamples, percentile), plus the **sign count** (e.g. "7/8 scenes").
3. **Tests:** exact Wilcoxon signed-rank (two-sided).
   - **Primary family, pre-registered in `docs/prereg.md`:**
     - Q1-interaction for (Deblurring-3DGS vs vanilla) and (BAGS vs vanilla), on curvature-high vs low and proximity-tight vs open, per strong condition: 8 tests;
     - Stage 2 efficacy of the selected rule vs `adc-bm`, per condition: 2 tests.
   - Holm correction within the family.
   - With n = 8 the exact two-sided floor is p = 0.0078 (all 8 agree); with 6 held-out scenes it is 0.031. Say this explicitly. Everything else is labelled exploratory.
4. **Noise floor:** σ_seed per metric and stratum from E04. Differences smaller than 2σ_seed are reported as "no detectable difference", whatever the p-value.
5. **Rankings:** bootstrap rank-probability tables (P(method ranks 1st | stratum)) and Kendall τ between the overall ranking and each stratum's ranking.
6. **Show every scene:** dot plots (one dot per scene, lines connecting paired methods) in the report appendix.
7. **Robustness:** re-run the primary analyses under the alternatives below. Conclusions that flip under these are reported as fragile.
   - extractor variants;
   - r = 4 cm curvature;
   - tertile vs physical proximity bins;
   - ICP-refined alignment;
   - τ = 2/10 cm.

## B9. Risk register and decision tree

| ID | Risk | Likelihood | Trigger / gate | Response |
|---|---|---|---|---|
| R1 | HPC access or queues bad | Medium | G0 (14 Oct) not met, or median queue wait > 12 h in W2 | Vanilla/rule runs → Kaggle (2 notebooks in parallel) + Colab; native baselines → request the GPU workstation for a 3-day block. Cut the Must list to E02/E03/E05 + E07/E08 on 4 held-out scenes. |
| R2 | Native baseline won't build on sm_89/sm_90 | Medium | G1 (23 Oct) | Try in order: (a) cu118 env + patches (≤ 4 h per repo); (b) Apptainer image; (c) the T4 on Colab with the native env (sm_75 works with older CUDA); (d) gsplat "-lite" ports, labelled ours. If only one baseline works, Stage 1 is vanilla + that one + the sharp control: still a complete Q1. |
| R3 | Frame / pose confusion (D3/V3 fail) | Low–Med | V3 in W1 | Train from the COLMAP sparse model directly (gsplat COLMAP parser, normalisation off). Never estimate the frame by ICP to the GT. |
| R4 | **Stage 1 null** (no stratum dependence) | Medium | G2 | Report the null with CIs (it's a contribution: "scene-averaged metrics are adequate here"). Stage 2 pivots to the **mixed protocol** (reliability is the lever) and to allocation *efficiency* (same F with fewer Gaussians; B− flat-first becomes the main arm). |
| R5 | H1 refuted (blur does *not* under-allocate fragile strata) | Medium | Fig 5a at G2 | Drop "fragile-first" as the main rule. Test `Bminus` and pruning-oriented variants. Frame Stage 2 as "allocation is not the bottleneck; here is the evidence". |
| R6 | Stage 2 shows no gain anywhere | Med–High | G3 / final | A clean negative with budget-matched controls and mechanism plots (Figs 5, 9) is presentable. Spend the remaining time on Ext explanatory items (H3, blur-aware host) rather than more rule-tweaking. |
| R7 | PSNR reproduction off by > 0.5 dB | Medium | E02 | Check in order: principal point (E15), train-view reference (D2), iterations, antialiasing, background. Document the residual gap. Geometry conclusions are unaffected, since our comparisons are internal. |
| R8 | Laptop RAM/CPU too small for 8-scene geometry | Low | W2 | Run `eval_cpu_array` on HPC CPU nodes ("many CPU cores"). Laptops only for single-scene inspection. |
| R9 | Mid-term feedback changes scope | Medium | 9 Oct | Edit §B0 layers, not the architecture. Most likely asks: "more method / less evaluation" → promote the Ext blur-aware host to Core; "reduce scope" → drop mild + six-method grid (already non-Core). |
| R10 | Person-hours run out | Medium | Any week < 8 h combined | Freeze at the last completed layer. The deliverable plan (§B10) works from Core alone. |
| R11 | Disk/quota on HPC | Low–Med | S0.1 | Keep depth exports as float16 `.npz`; delete checkpoints except the final ply; tar finished runs. |
| R12 | Defocus refocus "breathing" breaks pixel alignment between sharp and defocused images | Low | D2 visual check | If misaligned by > 1 px, use per-condition sharp references only and note it. |

**Decision tree at G2 (30 Oct):**

```
Is fragile-stratum ΔF (blurred − sharp) more negative than easy-stratum ΔF for vanilla (Fig 4)?
├─ YES → Is Gaussian density in fragile strata lower under blur (Fig 5a)?
│        ├─ YES → Stage 2 main arm = B (fragile-first); A on mixed; Ext: host + C.
│        └─ NO  → damage is not under-allocation: main arm = A (reliability) + blur-aware host; B− as converse.
└─ NO  → Report Stage-1 null. Stage 2 main = mixed protocol with A; efficiency framing with B−; B kept as one arm.
```

## B10. Final deliverables plan

**D1: GitHub repo** (W6–W7, Aditya leads).
- README contents:
  - one-paragraph summary + teaser figure;
  - install (`env/`);
  - "reproduce our figures from `results/registry.csv` in 2 commands" (`python -m cvp.analysis.figures --all`);
  - "re-run an experiment" (manifest + SLURM);
  - dataset download script;
  - pinned baseline commits + patches;
  - licence (MIT for our code; baselines keep their own licences and are not vendored).
- Tag `v1.0-endterm`.

**D2: 6-page IEEE report** (draft v1 by Fri 20 Nov, final by 26 Nov).

| Section | Pages | Content / figures |
|---|---|---|
| Abstract + I. Introduction | 0.75 | Problem, Lee & Lee principle, contributions (3 bullets) |
| II. Related work | 0.6 | Blur-aware NeRF/3DGS; densification (Pixel-GS, AbsGS, Revising-densification, MCMC, BSGS, EntON, EA-3DGS); structure-conditioned evaluation; RealX3D |
| III. Data and protocol | 1.0 | RealX3D blur subset (real defocus, rendered motion), frames, extraction, visibility, strata (Fig 2), metrics, validation V0–V4 |
| IV. Stage 1 results | 1.25 | Tab 1 (repro), Fig 3, Fig 4, Fig 5 |
| V. Stage 2: method + results | 1.25 | Rules (short pseudo-code), Tab 2, Fig 6, [Figs 7–9] |
| VI. Discussion, limitations, conclusion | 0.5 | Synthetic-motion caveat, n = 8, extractor dependence, what fails |
| References | 0.65 | ~25 refs |

**D3: Slides** (10–12 min; ≈ 11 slides; build W7; Saksham leads the content, both present).

| # | Slide | Time |
|---|---|---|
| 1 | Title + one-sentence story | 0:20 |
| 2 | Blur is geometric (Lee & Lee) + Fig 1 blur-length map | 1:00 |
| 3 | RealX3D: what it is (real defocus vs rendered motion), and the gap (no geometry for these methods; ranking flips) | 1:00 |
| 4 | Our protocol: frames, extraction, visibility, strata (Fig 2) | 1:15 |
| 5 | Harness validated + PSNR reproduction (V-table + Tab 1) | 0:45 |
| 6 | **Main result:** Fig 3 (per-stratum F, sharp control) | 1:30 |
| 7 | Where blur hurts: Fig 4 + allocation Fig 5 | 1:00 |
| 8 | Stage 2: rules in one picture + controls (budget-matched, MCMC, flat-first) | 1:00 |
| 9 | Stage 2 result: Tab 2 + Fig 6 crops | 1:15 |
| 10 | Demo video (45–60 s) | 1:00 |
| 11 | Takeaways + limitations + what's next | 0:45 |
| | **Total** | **≈ 11:00** |

Backup slides: H3 anchor (Fig 8), mixed sweep (Fig 7), extractor sensitivity, statistics details.

**D4: Demo video** (record W6, edit W7, Aditya).
- 1080p, 60 s.
- Synchronised fly-through: sharp-control | blurred-vanilla | blurred + best rule. Then error-coloured GT mesh rotation.
- Captions on screen; no voice-over needed.
- Use gsplat's camera-path rendering (or SuperSplat) and ffmpeg `hstack`.
- Also upload it to the repo release.

**D5: Archive:** `results/` CSVs, figure PDFs, final configs, `docs/decisions.md`.

## B11. Showcase at each layer, including if things go badly

| Situation | What we show | Is it convincing? |
|---|---|---|
| **Worst realistic case**: no native baseline builds; Stage 2 null | Vanilla gsplat on 8 scenes × 2 strong conditions with **sharp controls**, a validated laser-geometry harness, per-stratum ΔF (blur damage by structure), allocation diagnostics; Stage 2 rules at matched budget with an explained null. | Yes. It extends RealX3D with the geometry it lacks, and shows a rigorous negative. |
| **Core** | The above + Deblurring-3DGS + BAGS (Q1 ranking question answered), mixed-protocol reliability result, held-out Stage 2 table. | Solid course project; reads like a workshop paper. |
| **Extended** | + mild severity, blur-aware gsplat host, densification × blur-model interaction (Fig 9), exposure-anchor explanation of the BAD-Gaussians anomaly (Fig 8), rule C. | Strong: both a diagnosis *and* mechanisms. |
| **Stretch** | + full six-method grid, H2 parameter recovery, median-depth extractor, web viewer. | Benchmark-extension quality. |

## B12. Wording fixes (title, abstract, claims)

**Title** (both options drop "Real Blur" and avoid EA-3DGS's phrase):
- Recommended: **"Where Does Blur Break the Geometry? Stratified Laser-Scan Evaluation and Fragility-Aware Densification for 3D Gaussian Splatting under Defocus and Motion Blur"**
- Shorter alternative: "Structure-Stratified Geometric Diagnosis of Blur-Aware 3D Gaussian Splatting on RealX3D"

**Abstract (replacement draft):**

> Blur-aware 3D Gaussian Splatting builds on a principle formalised by Lee and Lee (CVPR 2013): camera-motion blur is a geometric consequence of camera motion and scene depth. On RealX3D, a benchmark with physically captured defocus, trajectory-rendered motion blur and laser-scanned ground truth, blur-aware methods are scored only with image metrics, and their rankings invert between blur types. We ask whether *geometry* fails in a structure-dependent way that scene-averaged scores conceal. We evaluate reconstructions against the laser scan in a fixed, shared world frame, stratifying ground truth by local curvature and surface proximity and comparing each blurred reconstruction with a sharp-capture control. We then test, at matched primitive budgets, whether densification guided by blur reliability and local structural fragility recovers the structures blur damages, including in captures that mix sharp and blurred views.

**Claim edits:**

| In proposal / slides | Replace with |
|---|---|
| "under Real Blur"; "physically captured degradation" (for blur) | "real (optical) defocus and trajectory-rendered motion blur" |
| "Under both mild settings plain 3DGS attains the highest PSNR of all six" | Delete (the main table is strong-only). If needed: "under strong defocus, vanilla 3DGS outperforms four of five blur-aware methods" ✅ |
| "RealX3D is the first real-capture benchmark…" | "RealX3D is a recent real-capture benchmark…" |
| "[12] GeMS" for MCMC densification | Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo", NeurIPS 2024 🔶 venue (arXiv 2404.09591 ✅). Keep GeMS only as an MCMC-based extreme-blur pipeline. |
| "Clouds are ICP-aligned to the laser scan per RealX3D's protocol" | "Reconstructions are mapped to the scan frame with the dataset's fixed COLMAP-to-world transform (1.2 cm RMSE); per-method ICP is reported only as a sensitivity analysis" |
| "per-view sharpness estimate" (factor i) | "a reliability signal, per view for mixed sharp/blurred captures and per pixel (depth-dependent) for motion blur" |
| "structure-aware densification" | "fragility-aware densification" |
| "Deblurring-3DGS [5]" next to RealX3D's numbers | Add a footnote: RealX3D cites it as "Zhang et al., 2024a"; we assume Lee et al.'s official code |

**References to add** (venues ✅ where checked, otherwise 🔶):
- Pixel-GS (Zhang et al., arXiv 2403.15530; ECCV 2024 🔶)
- AbsGS (Ye et al., arXiv 2404.10484; ACM MM 2024 🔶)
- Revising Densification (Rota Bulò et al., ECCV 2024 🔶)
- 3DGS-MCMC (arXiv 2404.09591)
- BSGS (arXiv 2510.12493)
- EntON (Jäger & Jutzi, arXiv 2603.06216)
- EA-3DGS (Guo et al., arXiv 2505.10787)
- "Exploration Matters…Blur Trap" (Wang et al., arXiv 2607.17965): cite to disambiguate "blur"
- gsplat (Ye et al., arXiv 2409.06765)
- Deblur-GS (Chen & Liu, I3D 2024)
- RaDe-GS (arXiv 2406.01467) and 2DGS/GOF for depth extraction

## B13. Questions for you (answers would change the plan)

1. **Exact HPC facts:**
   - Is the "RTX 6000" an A6000 or an RTX 6000 Ada?
   - Is the scheduler SLURM? Per-user GPU limits and maximum wall time?
   - Do compute nodes have internet?
   - Is Apptainer available?

   This changes §B5.3/B5.6 and whether Must + Should fits.
2. **Report format and weighting:** is the 6-page report graded separately from the talk, and is the IEEE format likely? This changes how much W7 goes to writing vs slides.
3. **What did the professor mean by "develop a 3D reconstruction pipeline"?** If they expect a *pipeline you built* rather than an evaluation, promote the Ext blur-aware gsplat host to Core. Ask at the mid-term.
4. **End-semester exams or other deadlines in W5–W7** (and around Diwali)? This may move G3/G4 earlier.
5. **Is the friend's RTX 50-series realistically available for multi-hour runs?** If yes, it becomes a third lane for vanilla/rule runs (Blackwell env only).
6. **Can you both commit ~10 h in W2 and W3?** Those are the heaviest weeks (harness validation + baseline builds). If not, move E06 (BAGS) to Should.
7. **Should Stage 2 be judged on held-out scenes only** (as planned)? This is stricter but halves the statistical power for Stage 2.
8. **Do you want the mild conditions in Core?** If the prof stresses severity, move E10 up and E09 down.

---

# Appendix T: Task cards

## T.0 Rules for every task (read first in every Claude Code session)

1. **Environment.** Python ≥ 3.10. Package code goes in `cvp/`; entry points in `scripts/` (argparse/tyro, `--help` works). Type hints and a module docstring naming the task ID.
2. **Tests.**
   - Every `[CC]` task adds `tests/test_<module>.py`. These run with `pytest -m "not gpu"` on CPU in < 2 min.
   - Use synthetic data (Open3D primitives, random Gaussians) or `data_fixture/` once S0.6 is done.
   - GPU-only code gets `@pytest.mark.gpu` tests plus CPU tests of all the surrounding logic, with the rasterizer mocked.
3. **No data in git.** Large paths come from `configs/paths.yaml` (template committed, real values local). Never hard-code `D:\` or HPC paths.
4. **Outputs follow §B5.4.** Run directories, JSON metrics with explicit units (metres, cm, dB), and `results/registry.csv` columns as in §B5.5.
5. **When done:**
   - Update `docs/STATUS.md` (ID, status, date, one-line summary, commit).
   - Write any decision into `docs/decisions.md`.
   - Commit on a branch `task/<ID>` and open a PR, or commit directly if the team prefers. Message `"<ID>: <summary>"`.
6. **If blocked by missing data or facts:** implement with clearly marked assumptions (a `# ASSUMPTION(<ID>):` comment) plus a check function, and list them in STATUS.md. Don't stall.
7. **Gaps in this plan:** don't invent extra scope; implement what the card says and note suggestions in STATUS.md.
8. **Hugging Face is probably unreachable from cloud sessions.** Do not try to download RealX3D there.

Card format: **ID · title** — Layer · Owner · Where · Depends on. Then **Inputs**, **Do**, **Outputs**, **Done when**.

## Mid-term sprint

**M1 · Repo skeleton, loader, scene inspector** — Core · CC (+Saksham runs) · cloud → laptop · none
- **Inputs:** RealX3D layout facts (§0.1 F1–F4, F12; §B5.1).
- **Do:**
  - Create the §B5.4 tree (empty modules with docstrings), `.gitignore`, `env/requirements-cpu.txt`, `env/requirements-train.txt`, `configs/paths.example.yaml`, `docs/STATUS.md`, `docs/decisions.md`, `docs/dataset_notes.md` (template with D1–D8 headings).
  - Move the existing reference files into `docs/refs/`.
  - `cvp/data/realx3d.py`:
    - parse `transforms_{train,val,test}.json` → per-frame dicts (image path, 4×4 c2w in **both** OpenGL and OpenCV conventions, fx, fy, cx, cy, w, h, sharpness);
    - optional COLMAP sparse reader (text and binary) → camera centres;
    - `discover_scene(scene_dir)` tolerant to unknown nesting.
  - `cvp/data/frames.py`: Umeyama similarity fit; `colmap2world` loader (`.npy`/`.txt`); `compose()`.
  - `scripts/inspect_scene.py --scene_dir --gt_dir`: prints and writes JSON for
    - counts per split;
    - intrinsics + principal-point offset + fy/fx (D5);
    - sharpness stats per split (D6) plus our own variance of Laplacian at native res;
    - `val/` vs `train/` alignment test (D2: per-pair mean abs diff, file-name match);
    - transforms-vs-COLMAP similarity (D3: scale, rotation angle, residual), and whether `colmap2world` contains scale.
- **Outputs:** the files above + `tests/test_realx3d.py` (synthetic transforms + COLMAP text files).
- **Done when:** tests pass, and `inspect_scene.py --help` works. Saksham runs it on the chosen scene and pastes the JSON into `docs/dataset_notes.md`.

**M2 · gsplat trainer + Colab notebook** — Core · CC (+Aditya runs) · cloud → Colab · M1
- **Do:**
  - `scripts/gsplat_smoke.py`: render 1 000 random Gaussians at 256² (forces the JIT compile, prints the timing).
  - `scripts/train_gsplat.py` (and `cvp/train/trainer.py`): adapt gsplat 1.5.3's `examples/simple_trainer.py` `default` path.
    - Use our loader with full K per camera; images loaded at data_4 resolution.
    - No world normalisation, or record the applied similarity in `frame.json`.
    - Flags: `--cond`, `--steps`, `--strategy {default,mcmc}`, `--absgrad`, `--seed`, `--train_images {blurred,sharp,mixed:p}`.
    - Eval: train-view PSNR/SSIM/LPIPS vs **sharp references** (from D2; configurable path), and test-view metrics.
    - Save `point_cloud.ply` (standard 3DGS fields) and `metrics_photo.json`.
    - `--save_depth train,test` writes per-view `npz` (ED, E[z²]-based std, alpha) at float16.
  - `notebooks/colab_midterm.ipynb` with the §A.3 cells. Saves runs to Google Drive.
- **Outputs:** scripts, notebook, `tests/test_trainer_cpu.py` (config parsing, camera conversion, ply writer round-trip).
- **Done when:** CPU tests pass. On Colab: 7k steps complete on the chosen scene and `metrics_photo.json` exists.

**M3 · Download and choose scene** — Core · Saksham · laptop · none
- **Do:** §A.3 M3 commands; list the scenes; pick one; extract; download its GT. Optionally the mild tars too (+1.7 GB).
- **Done when:** the scene name, paths and listing output are in `docs/dataset_notes.md`.

**M4 · Vanilla baseline runs** — Core · Aditya · Colab (or HPC) · M2, M3
- **Do:** smoke 7k on motion-s; then 30k motion-s, then 30k defocus-s; `--save_depth train,test`. Copy the run folders to the laptop.
- **Done when:** PSNR numbers are recorded next to RealX3D's per-scene numbers in `results/midterm/psnr.csv`.

**M5 · Strata preview** — Core · CC (+Saksham runs) · cloud → laptop · M1
- **Do:**
  - `cvp/geometry/strata.py` implementing §B6.4: surface variation at 2 radii, ray-cone proximity/thickness via Open3D `RaycastingScene`, normals from the mesh, binning helpers (pooled or per-scene), stability stats (Spearman, κ).
  - `scripts/strata_preview.py`: voxel-downsample to 1 cm for speed, compute labels, save coloured PLYs and 4 PNG screenshots (offscreen Open3D or matplotlib scatter fallback) + histograms.
- **Tests:** V0 shapes.
- **Done when:** V0 tests pass; screenshots exist for the chosen scene.

**M6 · Blur figures** — Core · CC (+Saksham runs) · cloud → laptop · M1
- **Do:** `scripts/blur_maps.py`:
  - (a) a 2×3 crop grid sharp/motion/defocus (mild and strong if available);
  - (b) the motion blur-length map b(u)=f·L/Z(u) for L ∈ {2, 6} cm from GT depth (downsampled to data_4 res; mm → m), with a colourbar in px;
  - (c) a relative CoC map ∝ |1/Z_f − 1/Z| for Z_f ∈ {0.6, 0.4} m, normalised by its median, showing near-uniformity;
  - (d) a printed summary (5th/50th/95th percentile blur length).
- **Done when:** the PNGs are in `results/midterm/<Scene>/blur/`.

**M7 · Harness v0 + V2** — Core · CC (+Saksham runs) · cloud → laptop · M1, M5
- **Do:** first versions of `fuse.py` (B6.2 steps 3–6), `visibility.py`, `metrics.py` (B6.5 3D metrics per stratum), `scripts/harness_v2_gtdepth.py` (validation V2 using the provided depth PNGs) and `scripts/eval_geometry.py --run --gt` (uses a run's depth npz).
- **Done when:** V1 synthetic tests pass in CI, and V2 runs on the chosen scene with its numbers recorded. If M4 depth exists, a first per-stratum table is in `results/midterm/`.

**M8 · Mid-term slides** — Core · both (CC can draft text) · —
- **Do:** 7–8 slides following §A.5, using figures from M3–M7 and the "what we learned" slide.
- **Done when:** two timed rehearsals ≤ 10 min.

**M9 · (optional) Deblurring-3DGS on Colab** — Str for the mid-term · Aditya · Colab · M4 done by 16:00
- **Do:** clone the repo; build its submodules against Colab's PyTorch; prepare the COLMAP layout via a quick script; run ≤ 7k iterations.
- **Done when:** it trains or the build error is logged in `docs/decisions.md`.

## Setup (W1)

**S0.1 · HPC reconnaissance** — Core · Aditya · HPC · —
- **Do:** fill `docs/hpc.md`:
  - partitions, GPU names + `compute_cap`, CUDA/driver modules, conda availability, Apptainer, per-user limits, max wall time;
  - internet on compute nodes?, scratch path + quota;
  - test a 5-min interactive GPU job.
- **Done when:** `docs/hpc.md` is complete and the questions in B13.1 are answered.

**S0.2 · Environments + SLURM templates** — Core · CC (+Aditya) · cloud → HPC · S0.1
- **Do:**
  - `env/cvp-gsplat.yml`, `env/legacy.yml`, `env/README_env.md` (build commands from §B5.3, arch list from `docs/hpc.md`);
  - `slurm/*.sbatch` from §B5.6;
  - `scripts/make_manifest.py` (YAML experiment → CSV rows: exp_id, scene, cond, method, rule, seed, extra args);
  - the `--pack N` wrapper.
- **Done when:** on HPC, `gsplat_smoke.py` runs inside the SLURM job, and a 2-row manifest array completes.

**S0.3 · Full blur data on HPC** — Core · Aditya · HPC login node · S0.1, D1 known
- **Do:** `scripts/download_realx3d.py --conditions motion_mild,motion_strong,defocus_mild,defocus_strong --scenes <8>` (uses `hf_hub_download`, resumable, writes SHA256); extract to scratch.
- **Done when:** all 8 scenes × 4 conditions + 8 GT tars are extracted and checksums are recorded.

**S0.4 · Dataset verification D1–D8 + dev/held-out split** — Core · CC (+Saksham runs) · cloud → HPC CPU/laptop · M1, S0.3
- **Do:**
  - `scripts/verify_dataset.py` runs D1–D8 for all scenes and writes `docs/dataset_notes.md` tables;
  - writes `frame.json` (`T_train→world`) per scene;
  - runs V3;
  - proposes the 2 dev scenes (highest share of "thin"/tight-proximity points) and writes `configs/scenes.yaml`.
- **Done when:** V3 passes for all scenes (or failures documented), and the split is committed **before** any Stage 2 run.

**S0.5 · Run registry and aggregation** — Core · CC · cloud · M2
- **Do:** `cvp/analysis/tables.py` + `scripts/aggregate.py` (scan `runs/`, validate JSON schema, append/update `results/registry.csv`, never duplicate rows); JSON schemas in `cvp/schemas.py`.
- **Done when:** tests on synthetic run dirs pass.

**S0.6 · Fixture** — Core · CC writes, Saksham runs · laptop · M1
- **Do:** `scripts/make_fixture.py`:
  - from one scene, take 4 train + 4 matching references + 2 test views, downsampled to 450×300 with K scaled;
  - transforms subset;
  - GT cloud cropped to the views' frustum and voxelised to 2 cm (≤ 200 k points);
  - decimated mesh (≤ 50 k triangles);
  - matching GT depth (downsampled);
  - `colmap2world`;
  - output ≤ 15 MB under `data_fixture/` with a README.
- **Done when:** committed; `pytest` uses it for the loader, harness and V2 tests.

**S0.7 · Related-work re-check + prereg skeleton** — Core · both · —
- **Do:** 1 h search for "mixed sharp blurred views Gaussian splatting reliability weighting densification" and similar; record findings in `docs/decisions.md`. Create `docs/prereg.md` with the §B8 primary tests, metrics and selection criteria (finalised in S1.12).
- **Done when:** both files are committed.

## Stage 1

**S1.1 · Gaussian I/O for all methods** — Core · CC · cloud · M2
- **Do:** `cvp/io/gaussians.py`:
  - read standard 3DGS ply, Mip-Splatting/BAGS (3D filter fold-in, B6.2.1) and nerfstudio splat ply → one `GaussianSet` dataclass (torch tensors);
  - `frame` metadata;
  - `cvp/io/colmap_export.py` writes a COLMAP-layout folder (PINHOLE, given poses, optional centre-crop to the principal point; images blurred/sharp/mixed) for native repos.
- **Done when:** round-trip tests pass, plus a test that the crop shifts cx, cy correctly.

**S1.2 · Depth export** — Core · CC · cloud (GPU-mocked) · S1.1
- **Do:** `scripts/export_depth.py --run`: for train + test views, render with gsplat → ED, std (via [z, z²] features), alpha → `npz` float16. Works for any `GaussianSet`.
- **Done when:** CPU tests of the camera/feature plumbing pass, plus a GPU test (marked) to run on HPC.

**S1.3 · Fusion and consistency filter** — Core · CC · cloud · S1.2, S0.6
- **Do:** finalise `cvp/geometry/fuse.py` (B6.2 steps 3–7 incl. variants and TSDF); parameters from `configs/base.yaml`.
- **Done when:** V2 passes on the fixture (thresholds scaled for 2 cm voxels) and the variants run.

**S1.4 · Visibility and label maps** — Core · CC · cloud · S0.6
- **Do:** `visibility.py` (B6.3.2) and `labels2d.py` (ray-cast label maps, B6.4 label transfer).
- **Done when:** tests on a synthetic two-box scene with known occlusion pass.

**S1.5 · Final strata, 8 scenes** — Core · Saksham · HPC CPU/laptop · M5, S0.3
- **Do:** compute labels per scene at r = 2 and 4 cm; pooled thresholds; stability report (ρ, κ) → `results/stage1/strata_summary.csv`; Fig 2.
- **Done when:** the κ decision is recorded in `docs/decisions.md`.

**S1.6 · Metrics module complete** — Core · CC · cloud · S1.3, S1.4
- **Do:** `metrics.py`: 3D per-stratum P/R/F at {2, 5, 10} cm, mean/median distances, outlier mass; 2.5D per-stratum depth errors; allocation density; ICP sensitivity (`align.py`); `scripts/eval_geometry.py` writes `geometry/*.json`.
- **Done when:** V1 passes; outputs match the schema.

**S1.7 · Harness validation V0–V4** — Core · Saksham · HPC CPU · S1.5, S1.6, E02/E03
- **Do:** run the V-table (§B6.7) on all scenes; write `results/stage1/validation.csv` and a paragraph in `docs/decisions.md`.
- **Done when:** V0–V3 pass (V4 after E02/E03).

**S1.8 · Vanilla + sharp-control + seed runs (E02, E03, E04)** — Core · Aditya · HPC GPU · S0.2, S0.3, S0.4
- **Do:** manifests from `configs/experiments/E02.yaml` etc.; launch; aggregate.
- **Done when:** all rows are in the registry with depth exports and geometry metrics.

**S1.9 · Native baselines (E05, E06)** — Core · Aditya (+CC for adapters/patches) · HPC GPU · S1.1, S0.2
- **Do:**
  - `external/clone_baselines.sh` (pinned commits) and per-repo env;
  - patches in `external/patches/`;
  - data via `colmap_export.py` (centre-cropped);
  - confirm their eval reproduces RealX3D within tolerance on 1 scene;
  - then run 8 × strong;
  - import the final ply → S1.2/S1.6 pipeline.
- Note which test-time render path each method uses (Deblurring-3DGS: no MLP at test; BAGS: no kernel at test).
- **Done when:** 16 rows per method are in the registry, or the R2 fallback is triggered and documented.

**S1.10 · H1 instrumentation** — Core · CC · cloud · M2
- **Do:** trainer hooks: every 500 its, log the #Gaussians and per-view mean |∇μ2D| with view sharpness; at the end, allocation density per stratum (needs GT labels, **evaluation only**, computed after training).
- **Done when:** logs and `stats_gaussians.json` appear for E02/E03 runs; Fig 5a script exists.

**S1.11 · Stage 1 statistics and figures** — Core · CC (+Saksham) · cloud/laptop · S0.5, S1.8, S1.9
- **Do:** `cvp/analysis/stats.py` (paired bootstrap, exact Wilcoxon, Holm, rank probabilities, Kendall τ, seed band); `figures.py` for Tab 1 and Figs 3, 4, 5 from `registry.csv`; per-scene dot plots.
- **Done when:** `python -m cvp.analysis.figures --stage1` regenerates all Stage 1 figures from CSV.

**S1.12 · Pre-registration finalised** — Core · both · — · before looking at E05/E06 stratified results
- **Do:** freeze `docs/prereg.md`: primary metric, strata definitions (incl. the κ decision), tests, dev/held-out split, Stage 2 selection rule.
- **Done when:** committed and dated.

## Stage 2

**S2.1 · FragilityStrategy** — Core · CC · cloud · M2
- **Do:**
  - implement §B7.4 for gsplat 1.5.3 (verify the method names in the installed source first);
  - rules adc, A, B, Bminus, AB (C, CB as stubs for S2.7);
  - per-Gaussian `frag` state carried through clone/split/prune (index bookkeeping!);
  - budget-matched helper (`adc-bm` threshold search; `mcmc-bm` `cap_max` from a reference run's count);
  - trainer flags.
- **Done when:** CPU tests with fake `info` tensors check that (a) rule `adc` is bit-identical to `DefaultStrategy`, (b) w_v = 1 makes A ≡ adc, (c) β = 0 makes B ≡ adc, (d) state stays aligned after clone/split/prune.

**S2.2 · Fragility estimator** — Core · CC (+Saksham) · cloud · —
- **Do:** `cvp/train/fragility.py` per §B7.2 (chunked, float32, `workers=-1`, timing log).
- **Done when:** tests on synthetic plane/rod/corner point sets give fragility ranks rod > plane and corner > plane (flatness highest on the plane); 2 M points in < 10 s on 8 cores (logged).

**S2.3 · Reliability + mixed protocol** — Core · CC · cloud · M1
- **Do:** `cvp/train/reliability.py` (variance of the Laplacian per view, normalisation, w_v); `cvp/data/mixed.py` (seeded subset replacement, written to `run/config.yaml`); trainer option `--train_images mixed:0.25`.
- **Done when:** tests check determinism and pairing correctness.

**S2.4 · Dev sweep (E07)** — Core · Aditya · HPC · S2.1–S2.3, S1.12
- **Do:** run the §B7.5 grid on the 2 dev scenes; apply the pre-registered selection; write `configs/stage2_final.yaml`.
- **Done when:** Gate G3 is recorded.

**S2.5 · Mixed protocol (E09)** — Core · Saksham · HPC · S2.3, S2.4 (γ chosen)
- **Done when:** 32 rows are in the registry.

**S2.6 · Final held-out runs (E08)** — Core · Aditya · HPC · S2.4
- **Done when:** all rows are in the registry with geometry metrics.

**S2.7 · Rule C** — Ext · CC · cloud · S2.1
- **Do:** per-pixel blur-extent weights (§B7.3) and the extra autograd call; flag `--rule C --L 0.06 --b0 16`.
- **Done when:** a CPU test (mocked means2d graph) passes and the timing overhead is logged on GPU.

**S2.8 · Blur-aware host** — Ext · CC (+Aditya) · cloud → HPC · M2
- **Do:** `cvp/train/blur_models.py`: motion (learnable se(3) endpoint, N virtual views, anchor option, batched render) and defocus (shared learnable PSF). ADC statistic averaged over virtual views. Trainer `--blur_model {none,motion,defocus}`.
- **Done when:**
  - CPU tests of the se(3) interpolation and PSF normalisation pass;
  - on GPU, train PSNR on strong motion beats vanilla on 1 dev scene, or the failure is documented.

**S2.9 · H3/H2 analysis (E12, E18)** — Ext · Saksham · HPC · S2.8
- **Do:** render at t ∈ {0, .25, .5, .75, 1}; PSNR vs reference curve; learned path length; PSF radius stats; Fig 8.
- **Done when:** the figure and a paragraph are in `docs/decisions.md`.

**S2.10 · Stage 2 analysis and figures** — Core · CC (+both) · cloud · S2.5, S2.6
- **Do:** Tab 2, Fig 6 (error-coloured meshes via Open3D offscreen; crops), Figs 7 and 9 if data exists; no-regression and efficacy checks per §B7.7.
- **Done when:** `python -m cvp.analysis.figures --stage2` regenerates everything.

## Stretch

- **X1 · BAD-Gaussians / CoCoGaussian / Deblur-GS (E16–E17):** same procedure as S1.9.
- **X2 · Median-depth extractor (E19):** vendor a RaDe-GS or GOF rasterizer in a separate env; re-render Core models; compare extractor sensitivity.
- **X3 · Web viewer page:** export SuperSplat-compatible plys plus a static page comparing models.
- **X4 · External anchor V5:** run one feed-forward model through our metric.

## Deliverables

- **D1 · Repo release:** §B10 D1; done when a fresh clone reproduces the figures from CSV on a laptop.
- **D2 · Report:** §B10 D2; done when 6 pages compile with no overfull boxes, all figures come from `results/`, and the references are checked.
- **D3 · Slides:** §B10 D3; done after two rehearsals ≤ 11:30.
- **D4 · Demo video:** §B10 D4; ≤ 60 s MP4 embedded and in the release.
- **D5 · Archive:** tag `v1.0-endterm`; `results/` and `docs/` complete.

---

# Appendix S: Sources checked for this plan (8 Oct 2026)

- RealX3D dataset card and file trees: <https://huggingface.co/datasets/ToferFish/RealX3D> (API trees for `data_4/`, `pointclouds/`, `data/`, `baseline_results/`)
- RealX3D paper (v2, 21 Jan 2026): <https://arxiv.org/abs/2512.23437>, HTML <https://arxiv.org/html/2512.23437v2>
- RealX3D code repo (empty as of today): <https://github.com/ShuhongLL/RealX3D>
- instant-ngp `colmap2nerf.py` (sharpness, pose normalisation): <https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py>
- gsplat: <https://github.com/nerfstudio-project/gsplat>, rasterization API <https://docs.gsplat.studio/main/apis/rasterization.html>, COLMAP example <https://docs.gsplat.studio/main/examples/colmap.html>
- Deblurring-3DGS: <https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting> (environment.yml)
- BAGS: <https://github.com/snldmt/BAGS>
- BAD-Gaussians: <https://github.com/WU-CVGL/BAD-Gaussians>
- CoCoGaussian: <https://github.com/Jho-Yonsei/CoCoGaussian>
- Deblur-GS: <https://github.com/Chaphlagical/Deblur-GS>
- BSGS: <https://arxiv.org/abs/2510.12493>
- EntON: <https://arxiv.org/abs/2603.06216>
- EA-3DGS: <https://arxiv.org/abs/2505.10787>
- Exploration Matters / Blur Trap: <https://arxiv.org/abs/2607.17965>, <https://chengbo-wang.github.io/ExploreGS/>
- Pixel-GS: <https://arxiv.org/abs/2403.15530> · AbsGS: <https://arxiv.org/abs/2404.10484> · 3DGS-MCMC: <https://arxiv.org/abs/2404.09591>
- Lee & Lee, CVPR 2013 (local PDF in `docs/`), our proposal PDF and proposal slides (local, read in full).
