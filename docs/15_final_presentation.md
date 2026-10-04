# 15 — Team presentation: the Animate Anyone core model

*15 slides, ~20 minutes. Each slide: headline, what to say, evidence pointer.*

---

### Slide 1 — Scope of this work

**Headline:** I own the *generative core*: ReferenceNet, Pose Guider, Denoising
UNet, VAE, CLIP, and the four attention roles.
**Say:** Not the pose detector, not datasets, not multi-GPU training. Everything
is grounded in the paper (arXiv:2311.17117v3, CVPR 2024) plus the Apache-2.0
`Moore-AnimateAnyone` reproduction vendored under `reference/`.
**Evidence:** `README.md`, `docs/01_architecture.md`.

### Slide 2 — The pipeline in one picture

**Headline:** Four information channels enter one sampling loop.
**Say:** reference image → VAE latent → **ReferenceNet** (appearance, stored);
reference image → **CLIP** (semantics); pose maps → **Pose Guider** (motion);
random noise → **Denoising UNet** (appearance via spatial attention, semantics via
cross-attention, continuity via temporal attention); latents → VAE → video.
**Evidence:** `docs/01_architecture.md` §1 (full diagram + tensor directions).

### Slide 3 — ReferenceNet: the appearance channel

**Headline:** A second UNet that only looks at the reference image.
**Say:** same structure as the denoising UNet *minus temporal layers*, SD weights,
timestep 0, run once; its per-block hidden states are banked and replayed as
extra K/V in every spatial-attention call — Q = video tokens, K/V = `[video ‖ ref]`.
That is what preserves face, clothing and texture that CLIP cannot.
**Evidence:** `docs/02_module_reference_net.md` (cited line-by-line).

### Slide 4 — Pose Guider: the motion channel

**Headline:** ~1.5M parameters, stride-2 convs, features **added** after `conv_in`.
**Say:** pose maps are resized, `/255`, *not* normalised; the Guider downsamples
×8 to the latent grid and its output is added to `conv_in(sample)` — verified in
the reference code (`unet_3d.py:503-505`). Concatenation (SD-image-variation
style) also exists and is implemented as ablation E13.
**Evidence:** `docs/03_module_pose_guider.md`.

### Slide 5 — Denoising UNet: where everything fuses

**Headline:** SD UNet inflated to video, conditioned on four channels.
**Say:** 2D convs applied per frame (`InflatedConv3d`), timestep embedding in
every ResBlock, AnimateDiff motion modules after the spatial blocks at all
scales, skip connections for spatial detail (they are *not* the appearance path),
CFG by batch doubling with a **zero** unconditional CLIP embedding.
**Evidence:** `docs/04_module_denoising_unet.md`.

### Slide 6 — VAE + CLIP: the frozen bookends

**Headline:** Latent space for compute; CLIP for semantics; both frozen.
**Say:** VAE compresses ×8 and scales latents by 0.18215 (÷0.18215 before decode);
decoding is **frame-by-frame** for memory, and there is *no* temporal model inside
the VAE. CLIP contributes exactly one 1024-d token per reference image — global
semantics, never pixel detail.
**Evidence:** `docs/05_module_vae.md`, `docs/06_module_clip.md`.

### Slide 7 — Attention: four roles, one formula

**Headline:** Same maths, four different Q/K/V sources.
**Say:** spatial self-attention (structure), reference attention (appearance),
cross-attention (semantics), temporal attention (continuity). Reference attention
doubles the key count → memory pressure exactly where SD is already tight.
**Evidence:** `docs/07_attention_deep_dive.md` (table + complexity + failure modes).

### Slide 8 — Shapes, verified not asserted

**Headline:** Every module boundary logs shape/dtype/device/grad.
**Say:** `ShapeLogger` records 18 tensors per run; the documentation's tensor
table is therefore checkable against `outputs/run_small/shape_log.json`.
**Evidence:** `docs/08_tensor_shapes.md`, `scripts/validate_build.py`.

### Slide 9 — Prototype: what we built and ran

**Headline:** CPU-only, 1.98M parameters total, no downloads.
**Say:** tiny random VAE + tiny random CLIP + ReferenceNet + Pose Guider + video
UNet + DDIM/CFG loop; 13/13 self-checks pass; a full 8-frame 96×96 clip runs in
~0.8–1.4 s with ~390 MB RSS. **The output is noise by design** — the point is the
architecture, shapes and cost.
**Evidence:** `src/aan/`, `outputs/run_small/run_log.json`.

### Slide 10 — Training: two stages, frozen bookends

**Headline:** Stage 1 images → stage 2 video; VAE and CLIP never train.
**Say:** stage 1 (768², bs 4, 30k steps) trains ReferenceNet + denoising UNet +
Pose Guider on single frames with a ≥30-frame "identity hard-negative" margin;
stage 2 (512², 24-frame clips, stride 4, 10k steps) adds AnimateDiff motion
modules; loss = SNR-γ-weighted MSE with zero-SNR v-prediction.
**Evidence:** `docs/09_training_and_inference.md`.

### Slide 11 — Ablations: what the numbers say

**Headline:** Cost is dominated by *branch count*, not conditioning strength.
**Say:** removing temporal attention −39 % runtime, CLIP −33 %, pose guider −25 %,
ReferenceNet −12 %, weakening reference attention to 0.5 only −8 %; runtime is
linear in steps; CPU fp16 costs +1115 %.
**Evidence:** `docs/10_experiments_ablation.md`, `outputs/ablations/*`.

### Slide 12 — Ablations: what they cannot say

**Headline:** With random weights, quality metrics are meaningless — and we say so.
**Say:** five ablations produce *identical* metrics by construction (temporal
modules are zero-initialised = identity); identity/pose/flicker need trained
checkpoints. Our reports mark those cells `n/a` instead of inventing numbers.
**Evidence:** `docs/10` §3, `docs/11_evaluation_metrics.md`.

### Slide 13 — Three bugs the instrumentation caught

**Headline:** Shape assertions are not bureaucracy.
**Say:** (1) CFG doubles the UNet batch but the reference bank was batch 1 →
broadcast, now asserted; (2) SD's zero-init output conv made ε̂ ≡ 0 with random
weights → the sampler was inert; (3) per-frame latents stacked without a frame
axis → `[1,4,96,12]`. Each is now a documented failure mode.
**Evidence:** `docs/12_debugging_guide.md`.

### Slide 14 — Honest status

**Headline:** Implemented and verified / inferred / simplified / open.
**Say:** the *mechanics* are implemented and tested; the *quality claims* are the
paper's and require trained weights; add–vs-concat, ε-vs-v and `full`-vs-`midup`
remain open; 15 open questions are listed with the smallest experiment for each.
**Evidence:** `docs/16_final_report.md` §2, `docs/14_open_questions.md`.

### Slide 15 — Ask / next steps

**Headline:** Two things would turn this from structural into empirical.
**Say:** (1) run the ablation matrix on a GPU with `clip.mode: pretrained` and add
a pose metric (DWPose on outputs) — open question #15; (2) confirm the original
checkpoint's pose fusion and prediction target from the released weights.
Keep the disclaimer practice: synthetic media, no impersonation, consent for
real likenesses.
**Evidence:** `docs/00_setup_guide.md` §6, `docs/14_open_questions.md`.