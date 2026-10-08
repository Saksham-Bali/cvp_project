# Full-semester plan for our CVP project

> **Before sending:** start a new chat in the cvp project, choose Opus 5.5, turn on web search, attach `Project_Proposal.pdf`, `CVP_Project_Presentation.pptx` and `Lee_Dense_3D_Reconstruction_2013_CVPR_paper.pdf` from `D:\cvp_project\docs`, then paste everything below the line.

---

**Where things are and what to produce.** You are in a Claude chat. Three files are attached to this message: `Project_Proposal.pdf`, `CVP_Project_Presentation.pptx` and `Lee_Dense_3D_Reconstruction_2013_CVPR_paper.pdf`. Read all three. Use web search to verify facts.

Our code repository is `github.com/Saksham-Bali/cvp_project`; locally it's `D:\cvp_project`. Right now it only holds a `docs/` folder with these reference files. Write the full plan as **one Markdown document** that we will save as `docs/PLAN.md` in that repo, and propose the repo's folder structure.

We will carry out the plan step by step in separate **Claude Code cloud sessions** on that repo. Those sessions have no GPU (4 CPUs, 16 GB RAM); GPU jobs run on our HPC or Colab. So give every task a short ID (e.g. `M1`, `S1.3`) and write it so it can be handed to one such session with a one-line instruction like "Do task S1.3 from docs/PLAN.md". Do your research and thinking first, then write the whole plan in one go.

---

You are helping a two-person undergraduate team plan the rest of a semester-long Computer Vision & Perception course project. It ends in an **end-term presentation** (and possibly a report). I want a carefully reasoned, realistic plan that takes us from where we are today to a strong final showcase.

I've given you everything we've studied, decided and verified so far (across earlier chats in this project). Don't just accept it. **Challenge it.** If part of the plan is weak, unclear or not feasible, say so and propose something better. Use web search to check any fact that matters, especially the dataset details and the code repositories. Where you're making an assumption, say so explicitly instead of stopping to ask. Put your questions for us at the end.

Three files are attached: our 2-page proposal (IEEE format), our 4-slide proposal presentation (it has speaker notes and a Q&A section), and the reference paper our professor assigned.

---

## 1. Course and team constraints

- **Course:** Computer Vision & Perception, BTech CS+AI, Plaksha University, India. Semester runs roughly Aug to Dec 2026.
- **The project must:**
  - be centred on computer vision and perception, not robotics or control;
  - **extend or improve existing work**, not just run existing code.

  The professor's project listing was "3D reconstruction from blurred images" and says "develop a 3D reconstruction pipeline." The anchor paper is Lee & Lee, CVPR 2013 (Section 2).
- **Team:** two people, Saksham Bali and Aditya Masutey.
  - As of August 2026, neither had hands-on experience with COLMAP, NeRF or 3DGS tooling.
  - Saksham has worked with point clouds before (RandLA-Net semantic segmentation on Toronto-3D), so Open3D-style point-cloud processing is familiar.
  - Plan for a learning curve on CUDA-heavy research code.
- **Already done:** 2-page proposal submitted (late Aug / early Sep) and proposal presentation given (~11 Sep 2026: 8 minutes, 4 slides). **Nothing has been implemented yet** (details in Section 9). The 12-week timeline we presented is therefore out of date, and about 7 weeks remain.

## 2. The reference paper (Lee & Lee, CVPR 2013)

"Dense 3D Reconstruction from Severely Blurred Images using a Single Moving Camera," H. S. Lee and K. M. Lee, CVPR 2013 (PDF attached).

**Core idea.** Motion blur caused by camera shake is not an arbitrary 2D corruption. Given:
- camera motion during the exposure, and
- the exposure time,

a pixel's blur kernel is the projected trajectory of its 3D scene point. That trajectory is fixed in closed form by the pixel's **depth**. So estimating the blur kernel is equivalent to estimating depth.

**Method:**
1. Use consecutive blurred video frames and the commutative property of blur, (L ⊗ K_{n−1}) ⊗ K_n = (L ⊗ K_n) ⊗ K_{n−1}, to build a blur-aware pixel correspondence. Exposure enters through shutter open/close timing coefficients.
2. Solve for inverse depth in a variational multi-view depth reconstruction, with camera motion from image registration.
3. Convert the depth into per-pixel, depth-dependent blur kernels for cheap non-uniform deblurring.

**How we use it.** We use the *principle* (blur carries geometric information and should be modelled geometrically), not the 2013 method itself. Modern blur-aware NeRF and 3DGS methods put this principle into practice. Our project asks whether they actually recover **geometry** under real-capture blur.

## 3. Domain background, as we understand it

- **3D Gaussian Splatting (3DGS)**, Kerbl et al., SIGGRAPH / ACM TOG 2023:
  - The scene is a set of anisotropic 3D Gaussians, each with a position, covariance, colour (spherical harmonics) and opacity.
  - Rendering projects the Gaussians to 2D and alpha-blends them.
  - Training minimises a photometric loss against posed training images.
- **Adaptive density control (ADC)** is the part of 3DGS we want to modify:
  - For each Gaussian, track the averaged view-space positional gradient.
  - If it exceeds a fixed threshold (0.0002 in the original), **clone** the Gaussian (if small) or **split** it (if large).
  - Prune Gaussians whose opacity is low.
- **Our Stage 2 hypothesis** (not yet established fact): blur removes high-frequency image content, so it weakens exactly this gradient signal. Thin and high-curvature regions may then never densify, while large textured surfaces keep gaining primitives.

## 4. Literature found so far

**Blur-aware methods.** RealX3D evaluates these six:

| Method | How it handles blur |
|---|---|
| Vanilla 3DGS | Baseline; no blur handling |
| Deblurring-3DGS (Lee et al., ECCV 2024, arXiv:2401.00834) | Small MLP predicts per-Gaussian rotation/scale offsets during training only. **Also modifies ADC:** adds points by kNN colour interpolation and uses depth-dependent pruning to keep far-plane Gaussians |
| BAD-Gaussians (Zhao et al., ECCV 2024, arXiv:2403.11831) | Physical motion-blur model: averages virtual sharp images along an SE(3) trajectory within the exposure, optimising the trajectory jointly. Built on nerfstudio/gsplat; rescales the densification threshold by virtual-view count |
| BAGS (Peng et al., ECCV 2024, arXiv:2403.04926) | Per-pixel multi-scale blur kernels; leaves ADC (Mip-Splatting's) untouched |
| CoCoGaussian (Lee et al., CVPR 2025, arXiv:2412.16028) | Models defocus through the circle of confusion; spawns multiple Gaussians per primitive |
| Deblur-GS (Chen & Liu, 2024) | Trajectory-based |

Related context: Deblur-NeRF (Ma et al., CVPR 2022), the NeRF predecessor, and the R3eVision survey (arXiv:2506.16262).

**Densification work.** This already exists, so our novelty must not be "a smarter densification rule" on its own:
- Pixel-GS and AbsGS: reweight or fix the gradient statistic.
- "Revising Densification in Gaussian Splatting" (Rota Bulò et al., ECCV 2024).
- EA-3DGS and EntON: curvature- and geometry-aware densification.
- BSGS: depth-conditioned thresholds under motion blur. This is the closest prior work to ours.
- 3DGS-MCMC (Kheradmand et al., NeurIPS 2024): replaces clone/split entirely.
- GeMS (arXiv:2508.14682): an extreme-motion-blur pipeline that uses MCMC densification.

**Structure-conditioned evaluation.** This is established practice; we adopt it and do not claim it:
- RNb-NeuS (CVPR 2024) reports Chamfer and normal error restricted to high-curvature and low-visibility areas.
- Multi-tiling NeRF (Photogrammetric Record 2024) does geometric assessment of aerial NeRF.

We found no work applying systematic curvature or proximity stratification to blur-degraded 3DGS.

**Geometry extraction from 3DGS.** 2DGS, GOF and RaDe-GS extract surfaces through rendered depth and fusion. Gaussian centres are a poor surface proxy.

## 5. The dataset: RealX3D

**Reference:** Liu et al., "RealX3D: A Physically-Degraded 3D Benchmark for Multi-view Visual Restoration and Reconstruction," arXiv:2512.23437, IJCV 2026. Hugging Face dataset: `ToferFish/RealX3D`.

**What we verified from the paper and dataset card:**
- 55 scenes from 15 indoor rooms. Degradations: defocus (mild/strong), motion blur (mild/strong), low-light, smoke, varying exposure, dynamic objects, reflection.
- A programmable rail re-runs identical camera trajectories, which gives pixel-aligned degraded/reference pairs.
- **Blur subset:** the paper says "defocus or camera motion blur with 8 scenes and 271 pairs, where each scene is captured at two blur severity levels."
- **Defocus is physically captured.** The lens is deliberately misfocused to 0.6 m (mild) and 0.4 m (strong); the scene sits at 3–5 m.
- **Motion blur is synthesized, not physically captured.** For each target frame they assume constant-speed motion along a path of 2 cm (mild) or 6 cm (strong; a figure caption says 5 cm), sample 64 intermediate poses, render those views, and integrate them with the target sharp image. This is why motion blur is missing from the RAW (`data_arw/`) subset.
- **Format:**
  - Full resolution ≈ 7008×4672; quarter resolution (`data_4/`) ≈ 1800×1200 with adjusted intrinsics.
  - Per scene: ~23–31 train, ~23–31 val and 4–6 test frames.
  - `transforms_{train,val,test}.json` carry poses, intrinsics and a per-frame **`sharpness`** field (e.g. `"sharpness": 25.72`, meaning undocumented).
  - Poses were estimated (COLMAP) on the sharp reference views.
  - Ground truth: laser-scanned culled point clouds, triangle meshes, and 16-bit metric depth maps rendered from the mesh.
- **RealX3D's evaluation:**
  - The six optimisation methods get **poses fixed to the provided poses** and are scored on **PSNR / SSIM / LPIPS only** (Table 2, strong blur only; no mild table in the main paper).
  - Geometry metrics (ICP alignment, 5 cm uniform subsampling, point-to-surface distance, Accuracy / Completeness / F1 at 5 cm; depth with median-ratio scaling) are reported **only for feed-forward models** (VGGT, π³, MapAnything, DepthAnything3).
- **Table 2, train-view PSNR, strong blur:**

  | | Motion | Defocus |
  |---|---|---|
  | Vanilla 3DGS | 20.33 | 20.83 |
  | Deblurring-3DGS | **20.64** | 19.82 |
  | Deblur-GS | 18.04 | 18.13 |
  | BAD-Gaussians | 19.28 | 19.35 |
  | BAGS | 18.75 | **21.01** |
  | CoCoGaussian | 20.14 | 20.40 |

  Rankings invert between blur types, no method dominates, and vanilla 3DGS stays competitive. The benchmark doesn't explain why.
- **Related challenge:** the NTIRE 2026 3D Restoration & Reconstruction challenge (arXiv:2604.04135; Codabench 13854) used RealX3D, but only for the low-light and smoke tracks.

**Still unverified. Please check, or plan a day-1 check:**
1. Do all 8 blur scenes have **both** motion and defocus? This decides whether our grid is 128 runs or 64. The README's folder listing shows 15 scene names, which doesn't match "8 scenes."
2. What renderer produced the 64 intermediate views for motion blur? If it was a learned reconstruction, its geometric errors are baked into the motion-blur inputs, and some method families may have a "home advantage."
3. What does the per-frame `sharpness` field mean, and how much does it vary **within** one condition?
4. Is the laser mesh already in the same frame and scale as the provided camera poses? The depth maps being rendered from the mesh suggests yes. If so, per-method ICP may be unnecessary, and even unfair.
5. Is RealX3D's evaluation code public?
6. For each of the six methods: is the code public, what licence, and which CUDA/PyTorch versions does it pin?

## 6. Our project as proposed

**Title:** "Structural Failure Diagnosis and Structure-Aware Densification for 3D Gaussian Splatting under Real Blur"

**Research questions:**
- **Q1 (diagnosis):** Does the ranking of blur-aware 3DGS methods change when geometric error is scored per structural stratum instead of per scene?
- **Q2 (intervention):** Can a targeted change to primitive allocation recover the structures that blur destroys?

**Stage 1: stratified geometric evaluation**
- Partition each scene's laser-scan ground truth along structural axes:
  - **local curvature:** local-PCA surface variation λ0/(λ0+λ1+λ2), with strata checked for stability across two neighbourhood radii;
  - **surface proximity:** distance to the nearest independent surface, separating isolated geometry from tightly spaced or thin structure;
  - **thickness** via the Shape Diameter Function: optional, only if the mesh passes watertightness and manifoldness checks (it is frustum-trimmed, so it may not).
- Quantile-bin each axis into 3 strata and analyse each axis separately. A joint 3×3×3 partition would be too sparse over 8 scenes.
- Extract a point cloud from **every** method by one common procedure: render depth from the training views, back-project and fuse. Never use Gaussian centres.
- Align to the laser scan and report Accuracy, Completeness and F-score per stratum at 5 cm, plus 2 cm and 10 cm as a sensitivity check.

**Stage 2: structure-aware densification.** A small decision-rule change inside an existing 3DGS codebase, with no new network:
- **(i)** Weight the densification gradient statistic by a sharpness or reliability estimate, so heavily blurred inputs don't dilute it. The exact weighting is chosen by ablation.
- **(ii)** Lower the effective densification threshold where *reconstruction-time* signals (the local neighbourhood of current Gaussians) indicate structurally fragile geometry.
- **No leakage:** the laser ground truth is used only for diagnosis and scoring, never in training.
- **Ablation:** vanilla ADC, (i) only, (ii) only, (i)+(ii), and 3DGS-MCMC.
- **Stage 2 is guided by Stage 1 but does not depend on it.** If thin structures turn out *not* to be disproportionately degraded, the same instrumentation tests the converse allocation hypothesis.

**Supporting hypotheses:**
- **H1:** ADC interacts with blur. Tested by within-method ablation, since cross-method differences can't isolate ADC.
- **H2:** blur-model mismatch against RealX3D's known blur parameters. Reach goal.
- **H3:** trajectory-optimising methods drift away from correct poses (absolute trajectory error against the provided poses). Reach goal.

**Evaluation grid as presented:**
- Methods: vanilla 3DGS, Deblurring-3DGS (strongest motion baseline), BAGS (strongest defocus baseline), and ours.
- 8 scenes × {motion, defocus} × {mild, strong}, giving **128 runs**, plus clean reference captures as controls.
- The full six-method grid is an extension.
- Primary metrics are geometric and per stratum. PSNR / SSIM / LPIPS are secondary, and we will reproduce RealX3D's Table 2 to within about 0.3 dB as a pipeline-correctness check.
- Statistics: paired per-scene differences, bootstrap confidence intervals, Wilcoxon signed-rank, every scene shown.

**Success criteria:**
1. Reproduction within tolerance.
2. Establish whether geometric error and method ranking depend on the stratum.
3. Improve per-stratum F-score in the fragile strata **without** regressing overall quality, or document why the change fails.

A null Stage 1 result (rankings stable under stratification) is still reportable.

**Timeline as presented:**
- Weeks 1–2: reproduce one PSNR row.
- Weeks 3–7: Stage 1.
- Weeks 8–11: Stage 2 and ablations.
- Week 12: sensitivity analysis and write-up.
- Week-3 checkpoint: drop any baseline that won't build.
- If Stage 0 slips badly, Stage 1 alone becomes the project.

**Earlier idea we haven't committed to.** A **heterogeneous-degradation protocol**: replace a fraction of the blurred training views with their pixel-aligned sharp references and sweep the sharp:blurred ratio. This is realistic (handheld capture always includes some sharp frames) and might pair well with Stage 2.

## 7. Problems found since the proposal. Please address them in the plan.

1. **"Real blur" overclaims.** Only defocus is physically captured; motion blur is trajectory-rendered. The title and claims need honest wording, and motion and defocus results may need to be interpreted differently. Note also that the synthetic motion blur follows almost exactly the constant-speed trajectory model BAD-Gaussians assumes, yet BAD-Gaussians still scores below vanilla 3DGS. Is that worth investigating, and does it change H2?
2. **Per-view sharpness weighting (factor i) may be close to a no-op on RealX3D.** Within one condition every view gets the same blur (same path length, or same misfocus), so per-view sharpness barely varies. Candidate fixes:
   - (a) per-pixel or per-region reliability weighting. Motion blur does vary within an image with depth, which ties back to Lee & Lee.
   - (b) the mixed sharp/blurred protocol above, to create variation between views.

   Decide which, or propose something better.
3. **Alignment.** Per-method ICP can absorb part of a method's systematic error. If the mesh is already in the pose frame, prefer no ICP, or one fixed transform per scene shared by all methods.
4. **Depth extraction.** Alpha-blended mean depth creates floaters between surfaces, which land exactly in our thin and close-proximity strata. Consider median depth and/or TSDF fusion, with a sensitivity check against the extractor choice and against RealX3D's 16-bit depth.
5. **Claims and citations still to fix:**
   - Drop the unverified claim in our proposal that "under mild blur plain 3DGS attains the highest PSNR of all six." The main table covers strong blur only.
   - Soften "RealX3D is the *first* real-capture benchmark."
   - Cite 3DGS-MCMC directly, not GeMS.
   - Add Pixel-GS, AbsGS, EA-3DGS and BSGS to the references.
6. **How fragility is estimated at training time (factor ii) is still vague.** We need a concrete, implementable definition, for example local PCA over neighbouring Gaussian centres, Gaussian scale anisotropy, opacity, or rendered-depth discontinuities.

## 8. Compute

- **University HPC (newly available, shared across the university, so expect queues):**
  - 2 × NVIDIA RTX 6000 (48 GB): most likely RTX 6000 Ada, compute capability sm_89; please confirm what this means.
  - 2 × NVIDIA H200 (141 GB): Hopper, sm_90.
  - Many CPU cores.
  - Probably a SLURM-style scheduler, not yet confirmed.
- **A GPU workstation** available on request.
- **Occasional access to a friend's RTX 5060/5090-class GPU** (Blackwell, sm_120). Treat it as unreliable, and as the highest build risk, since it needs CUDA 12.8+.
- **Main build risk:** 2024-era research repos with custom CUDA rasterisers (diff-gaussian-rasterization, simple-knn, etc.) pinned to old CUDA/PyTorch. Please plan:
  - environment strategy (conda vs containers, CUDA 11.8 vs 12.x, `TORCH_CUDA_ARCH_LIST`);
  - whether to standardise on one codebase (e.g. gsplat/nerfstudio) and port the methods' ideas into it, rather than running heterogeneous repos;
  - rough GPU-hour estimates per run at quarter resolution, and whether the grid fits.

## 9. Our status, dates and final deliverables

- **Today:** 8 October 2026.
- **Mid-term evaluation:** **9 October 2026, 10–11 am**, about 34 hours from now. Expectations:
  - Mainly that we clearly understand the problem, our proposed work, the domain and the dataset.
  - Ideally also that one available GitHub implementation runs (the exact requirement wasn't specified).
  - The professor's mid-term feedback may change the scope; the plan should be easy to adjust afterwards.
- **Our laptops have no NVIDIA GPU.** Saksham's is an HP Envy x360 with an Intel Core i7 U-series processor and integrated Intel graphics only, and Aditya's is similar. CUDA-based 3DGS code **cannot** run locally. Anything that "runs" must be on the HPC, or on a free cloud GPU (Google Colab / Kaggle T4) as a fallback if HPC access isn't set up in time. Laptops are for code editing, viewing results (e.g. a web-based splat viewer) and point-cloud analysis on CPU (Open3D).
- **End-term presentation:** sometime between **27 Nov and 3 Dec 2026** (exact date not yet announced). Plan for **27 Nov as the worst case**.
  - 10–12 minutes; no fixed slide count or format.
  - A demo may be shown, but only inside the presentation itself.
- **Required end-term submissions besides the slides:**
  1. Full code in a GitHub repository.
  2. A **6-page report**. The format will be announced later, so plan for an IEEE-style conference paper, like our proposal.
  3. Multimedia, including a demo video if we choose to include one.
- **Progress so far: none on implementation.** We haven't installed any method, downloaded the data or reproduced any number. We plan to use AI assistance heavily to move fast.
- **Time available:**
  - Each of us can guarantee **at least 5 hours/week**, and up to about 10 in a week that needs it.
  - That is roughly 70–140 person-hours in total until the end-term, including the report, slides and video. Use this to order and tier the work, not to cap ambition.
  - Aditya's skills are similar to Saksham's.

**How to handle scope. This matters to us.** Aim big now. We'll scope down later, based on the professor's mid-term feedback and on how the work progresses. Do **not** shrink the project to what we're sure we can finish; a thin project won't get a good grade or make a good project. The end-term work has to look substantial, so plan **ambitiously**: both stages, the full evaluation, and the reach hypotheses included. It's fine if the plan holds more work than we end up doing. We will cut as time goes on.

What we need is for the plan to be **layered**, so cutting is easy and nothing breaks:
- Order the work so each completed layer is a presentable result on its own.
- Mark every item Core / Extended / Stretch.
- Say what the showcase looks like if we stop after each layer.
- Use AI-assisted coding and the HPC to push throughput, rather than lowering ambition.

## 10. What I want from you

Produce a complete plan from today to the end-term as a structured document, in this order:

**A. Mid-term sprint (now until 9 Oct, 10 am).** This is a real work target, not just a talk. Our thinking so far is below; improve on it.

- **Goal 1: understanding, shown with real data.** We present the topic, domain, problem, our planned work and the dataset to the professor and TAs, using concrete examples from one RealX3D blur scene: blurred vs sharp reference image pairs (mild/strong, motion/defocus), camera poses, the laser-scanned mesh, and the per-frame `sharpness` values.
- **Goal 2: run the RealX3D baseline.** Reproduce what the RealX3D paper already did, on one scene, on the HPC or Colab: vanilla 3DGS first (via gsplat), then one blur-aware method (e.g. Deblurring-3DGS) if it builds. Compare PSNR with RealX3D's Table 2.
- **Goal 3: start the novel part.** Begin Stage 1, the geometry side, even if it isn't finished:
  - compute curvature and surface-proximity strata on the laser mesh and visualise them;
  - if Goal 2 produced a trained model: render depth, fuse a point cloud, and compute a first per-stratum accuracy/completeness/F-score for one scene.

**Our first idea was "if the baseline is hard, only do the baseline; if it's easy, move to the novel part." We now think they should run in parallel**, because the stratification work runs on CPU and doesn't depend on the baseline. One of us takes the GPU baseline, the other takes data exploration and stratification. Confirm or improve this split.

**For the mid-term sprint, give us:**
- exact steps, commands and expected run times for each goal;
- what is realistic in about 30 hours given Colab/HPC setup time and dataset download size (check the quarter-resolution scene size and how to download a single scene with `huggingface_hub`);
- a fallback for each goal;
- the 4–5 points to say, plus the Section 7 corrections framed as "what we learned since the proposal";
- which tasks Claude Code cloud sessions can do for us (no GPU, CPU only, Hugging Face may need allowlisting).

**B. The full plan:**

0. **Layered scope.** Keep the full, ambitious scope (see "How to handle scope" in Section 9), and organise it into layers:
   - **Core:** must exist for a credible end-term.
   - **Extended:** what we're aiming for.
   - **Stretch:** what makes it exceptional.

   For each layer, give the result we could present if we stopped there, and a rough effort estimate. Be honest about which layers fit ~7 weeks, but don't delete anything. Make sure dropping a later layer never invalidates an earlier one. Stage 2 should stay in the plan.
1. **Critical review of the current plan.** What is strong, what is weak or risky, and what you would change or strengthen. Fix weak parts rather than removing them. This includes whether Stage 2 as designed can plausibly work on this dataset, and whether the novelty claim holds against BSGS, Pixel-GS, AbsGS and EA-3DGS. Be blunt.
2. **The end-term showcase, designed first.** Work backwards from the final presentation:
   - the one-sentence story;
   - the 4–6 key figures and tables that must exist (describe each concretely: axes, what it compares, what a "good" result looks like);
   - qualitative visuals (per-stratum error heatmaps on the laser mesh, before/after renders, Gaussian-density maps);
   - a live or recorded demo if sensible.
3. **Week-by-week plan from today to the end-term date.** For each week: concrete tasks, an owner (Saksham, Aditya or both), deliverables, and a "done when…" test. Include explicit go/no-go gates and what we switch to if a gate fails.
4. **Experiment matrix** prioritised Must / Should / Could. For each experiment: run count, estimated GPU-hours, which GPU type, and which question (Q1/Q2/H1–H3) it answers. Fit it to our compute, including queue contention.
5. **Day-1 / week-1 technical checklist:**
   - data download and the verification checks in Section 5;
   - environment builds per GPU type;
   - which repo or codebase to standardise on;
   - directory and logging conventions;
   - experiment tracking;
   - SLURM job templates.
6. **Exact Stage 1 protocol:**
   - geometry extraction (depth type, fusion, voxel size);
   - alignment;
   - stratification (radii, binning, stability check);
   - metric computation;
   - how to validate the harness before trusting it (e.g. reproduce a known RealX3D geometry number, or sanity-check on the clean reference reconstruction).
7. **Stage 2 design in implementable detail:**
   - 1–3 concrete candidate rules, with pseudo-code for the modified ADC step;
   - the training-time fragility estimator;
   - the sharpness/reliability signal, resolving Problem 2;
   - hyperparameters to sweep;
   - the ablation ladder;
   - the "no regression" criterion.
8. **Analysis and statistics plan** suited to 8 scenes.
9. **Risk register and decision tree.** For example: "if no blur-aware baseline builds by week X, then …"; "if Stage 1 finds no stratum dependence, then …"; "if HPC queues are bad, then …".
10. **Final deliverables plan.** When and how to build:
    - the **GitHub repo** (structure, README, reproducibility);
    - the **6-page report** (section outline, page budget per section, which figures);
    - the **demo / multimedia** (what to record and how to embed it in a 10–12 min talk);
    - the **slides** (rough slide-by-slide outline with time per slide).

    Leave the final 1–1.5 weeks for writing and slides, not experiments.
11. **Showcase at each layer.** Include the minimum viable version, but also what the Extended and Stretch versions add. What we can still present convincingly if things go badly from here.
12. **Wording fixes** for the title, abstract and claims (Section 7.5).
13. **Questions for us** whose answers would change the plan.

**Style:**
- Precise, but in simple language. We are strong undergraduates, not 3DGS experts; define any jargon the first time you use it.
- No padding, and no generic advice like "communicate well."
- Use tables where they help.
- Mark clearly what you verified by search versus what you are assuming.
