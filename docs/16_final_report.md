# 16 — Final report (Team Member 3)

Scope: the **core diffusion/animation model** of Animate Anyone — ReferenceNet,
Pose Guider, Denoising UNet, VAE, CLIP image encoder, and the four attention
roles. Deliverables: `README.md` (index) + `docs/00`–`docs/15`, a runnable
educational prototype in `src/aan/`, and 19 measured ablation configurations.

---

## 1. The architecture in plain language

* **ReferenceNet** answers: *"What does the character look like?"*
  A copy of the denoising UNet without temporal layers, run **once** on the
  reference image at timestep 0. Its per-block hidden states are stored (the
  "bank") and injected into the denoising UNet as **extra keys/values** in every
  spatial-attention call, so each generated frame can query the reference's
  appearance token by token.

* **Pose Guider** answers: *"How should the character move?"*
  A deliberately small stride-2 convolutional encoder (≈1.5M parameters, ~0.2 %
  of the model) mapping every frame's pose map to the **latent grid**
  (H/8×W/8). Its output is **added** to the denoising UNet's features right after
  `conv_in` — addition, not channel concatenation (production behaviour).

* **CLIP image encoder** answers: *"What is the high-level visual meaning of the
  reference?"* A frozen ViT produces one projected embedding per reference image;
  it enters every transformer block through cross-attention. Global semantics
  ("what kind of character is this"), never pixel-level detail.

* **Denoising UNet** answers: *"How can noisy latent frames be transformed into a
  coherent animation?"* The only learned module inside the sampling loop: 20–30
  DDIM steps take pure noise to structured latents, conditioned on pose,
  reference appearance, CLIP semantics and the timestep.

* **Spatial (reference) attention** answers: *"Which appearance details should be
  transferred?"* Queries come from the video being generated; keys/values are the
  concatenation of the video's own tokens and the ReferenceNet bank, so the
  softmax decides, per location, how much reference information to import.

* **Cross-attention** answers: *"Which semantic features should guide
  generation?"* Queries from the UNet, keys/values from the CLIP embedding.

* **Temporal attention** answers: *"How should neighbouring frames remain
  consistent?"* AnimateDiff-style motion modules attend across frames at each
  spatial position (with temporal positional encoding), suppressing flicker.

* **VAE** answers: *"How do we move between compressed latent space and visible
  video?"* Encodes the reference image to 4-channel latents at 1/8 resolution
  (×0.18215) and decodes the final latent frame-by-frame into RGB. Frozen, and a
  hard ceiling on sharpness.

One sentence: **the model copies the reference's appearance through attention,
follows the pose through a small additive encoder, keeps frames coherent with
temporal attention, and moves between pixels and latents with a frozen VAE.**

---

## 2. Evidence status (required distinction)

### 2.1 Directly implemented *and* verified here

| item | verification |
|---|---|
| Reference attention = Q from frames, K/V = `[frames ‖ bank]` | `validate_build.py` (bank changes output; `strength=0` ≡ no bank) |
| ReferenceNet writes LayerNorm-ed hidden states per block; reader consumes them | `reference_net.py`; validated bank ladder `[144,36,9,36,144]×[32,64,64,64,32]` |
| Pose Guider produces H/8 features **added after `conv_in`** | `validate_build.py` (grid, zeros-at-init, signal when init disabled) |
| Concat pose fusion as an alternative (8-channel `conv_in`) | `validate_build.py`, ablation E13 |
| Cross-attention with CLIP context; identity when context is `None` | `validate_build.py` |
| Temporal attention over frames with sinusoidal PE, zero-init ⇒ identity | `validate_build.py` |
| CFG = batch doubling + `ε_u + g(ε_c − ε_u)` with zero uncond context; bank broadcast | `validate_build.py` (broadcast), `pipeline.py` |
| DDIM sampling, β 0.00085→0.012, `steps_offset=1`, ε-prediction | `pipeline.py`; matches `configs/inference/inference_v1.yaml` |
| Frame-wise VAE decode, `×0.18215` / `÷0.18215`, `[-1,1] ↔ [0,1]` | `validate_build.py`, `vae.py` |
| Complete CFG + DDIM + decode + mp4 pipeline with timing/shape logs | `scripts/run_prototype.py`, `outputs/run_small/*` |
| Ablation harness for the 12 required experiments (+7 extra) | `scripts/run_ablations.py`, `outputs/ablations/*` |
| Cost behaviour vs steps / frames / resolution / CFG / precision | measured (`docs/10_experiments_ablation.md` §2) |

### 2.2 Inferred from the architecture / paper + reference code (not executable here)

* the **quality** effects of each component (identity preservation, pose accuracy,
  flicker reduction) — the paper's claims, verified there with trained weights and
  *not* measurable with our random weights;
* that ReferenceNet's own cross-attention on the CLIP embedding materially affects
  appearance (the code passes it; the effect is unquantified);
* the exact latent grid / resolution used by the original checkpoints.

### 2.3 Simplified for education (deliberate)

* 3 spatial scales instead of 4; 32–64 base channels instead of 320–1280;
* 8 frames instead of 24; no long-video windowing;
* writer/reader bank pairing by construction instead of sorting channel widths;
* the reference bank is broadcast to **both** CFG halves (production applies it
  to the conditional half via `uc_mask`);
* zero-init output convolutions disabled by default (see §2.5);
* manual softmax instead of fused SDPA/xformers kernels (identical mathematics).

### 2.4 Experimentally validated (numbers, not claims)

* 13/13 self-checks pass; 19 ablation configurations ran end-to-end;
* removing temporal attention saves **39 %** runtime and changes metrics **not at
  all** (it is identity-at-init by construction);
* removing CLIP saves **33 %**, removing the Pose Guider **25 %**, removing
  ReferenceNet **12 %**, weakening reference attention to 0.5 **8 %**;
* runtime is linear in steps (0.90 / 1.40 / 2.59 s for 2 / 8 / 16);
* CFG costs only **+14 %** wall-clock at this model size (77.4 vs 68.0 ms/step),
  not 2×, because per-layer overhead dominates small models;
* CPU fp16 VAE costs **+1115 %** runtime;
* `vae_psnr ≈ 18.2 dB` for the random tiny VAE — the only quality metric that is
  meaningful without training.

### 2.5 Uncertainty / requires further work

* whether the *original* Animate Anyone fuses pose features by addition or by
  concatenation (we follow the Moore reproduction: addition);
* whether the released checkpoints use ε- or v-prediction (both appear in the
  reference repo's v1/v2 configs);
* `fusion_blocks="full"` vs `"midup"` — no public ablation;
* whether our structural metrics correlate at all with perceived quality
  (open question #15, the most important one for the team);
* all 15 open questions are enumerated in `docs/14_open_questions.md`.

---

## 3. Deliverables index

| # | deliverable | file |
|---|---|---|
| 1 | architecture diagram | `docs/01_architecture.md` |
| 2 | module-by-module explanation | `docs/02`–`docs/06` |
| 3–7 | ReferenceNet / Pose Guider / UNet / VAE / CLIP notes | `docs/02`…`docs/06` |
| 8 | attention document | `docs/07_attention_deep_dive.md` |
| 9 | tensor-shape table | `docs/08_tensor_shapes.md` |
| 10 | training & inference flowcharts | `docs/09_training_and_inference.md` |
| 11 | minimal working prototype | `src/aan/`, `scripts/`, `configs/` |
| 12 | configuration files | `configs/prototype_cpu_small.yaml`, `prototype_cpu_quick.yaml`, `ablation_matrix.yaml` |
| 13 | reproducible setup guide | `docs/00_setup_guide.md` |
| 14 | experiment & ablation report | `docs/10_experiments_ablation.md` + `outputs/ablations/` |
| 15 | debugging guide | `docs/12_debugging_guide.md` |
| 16 | final presentation | `docs/15_final_presentation.md` (this file is the written report) |
| 17 | glossary | `docs/13_glossary.md` |
| 18 | open research questions | `docs/14_open_questions.md` |
| — | evaluation metrics | `docs/11_evaluation_metrics.md` |

---

## 4. Handover notes for the team

1. **Start with** `docs/01_architecture.md` (diagram) and
   `docs/07_attention_deep_dive.md` (the four attention roles) — they carry most
   of the intuition; the module notes carry the exact code references.
2. **If you inherit the trained checkpoint path**, the three things that will bite
   first are: (a) pose/latent grid alignment, (b) reference bank channel pairing,
   (c) `×0.18215` scaling — all three are asserted in code here so they fail loudly
   instead of silently degrading output.
3. **When you can afford a GPU**, re-run `scripts/run_ablations.py` with
   `clip.mode: pretrained` and add a pose metric; that converts this study from
   "structural" to "empirical", which is open question #15.
4. **Ethics**: keep the disclaimer practice; do not run real people's likenesses
   without consent; do not ship generated media as real footage.