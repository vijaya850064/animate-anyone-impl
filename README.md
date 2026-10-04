# Animate Anyone — Core Diffusion & Animation Model (Team Member 3 study)

Implementation-oriented study of the **core generative components** of
*Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for
Character Animation* (Hu et al., CVPR 2024, [arXiv:2311.17117](https://arxiv.org/abs/2311.17117)):

**ReferenceNet · Pose Guider · Denoising UNet · VAE · CLIP image encoder ·
spatial / reference / cross / temporal attention.**

---

## 1. What this repository is

| Layer | Status | Where |
|---|---|---|
| **Production architecture** (what the paper / reference implementation do) | Documented from the paper + `Moore-AnimateAnyone` source, with file:line citations | `docs/01`–`docs/10`, `reference/` |
| **Research reproduction of the mechanics** (bank transfer, pose addition inside the UNet, frame-wise VAE decode, DDIM+CFG loop) | Implemented, shape-asserted, runnable on CPU | `src/aan/`, `scripts/`, `configs/` |
| **Educational prototype** (tiny *randomly initialised* modules) | Implemented and measured (13/13 self-checks pass, 19 ablation configs executed) | `src/aan/pipeline.py`, `outputs/` |
| Trained production checkpoints / paper-level video quality | **Out of scope** — needs the full datasets and multi-GPU training | — |

> **Honesty statement (please read before using any output).**
> The prototype uses *randomly initialised* modules (a few hundred K parameters
> each) and *procedurally drawn* inputs. Its output video is **noise**, and it is
> **not** a reproduction of Animate Anyone quality. What it does reproduce is the
> **architecture, data flow, tensor shapes, and cost behaviour**. Any claim about
> identity preservation, pose accuracy or flicker in this repo is explicitly
> marked as *not measurable without trained weights*.

Ethics & compliance (project requirements):
* No private/unauthorised datasets, no user media, no scraping. Demo inputs are
  drawn procedurally by `scripts/make_demo_inputs.py`.
* Not for impersonation, fraud, or non-consensual identity manipulation.
* Generated files carry a `disclaimer` field in `run_log.json` / report headers:
  **do not present generated media as real footage.**
* Pretrained checkpoints are not redistributed here; `docs/00_setup_guide.md`
  lists the licences of every optional download.

---

## 2. Quickstart (CPU, ~1 minute, no downloads)

```bash
python -m pip install -r requirements.txt          # or: pip install -e .

# 0) self-checks: every module boundary, attention math, end-to-end run
python scripts/validate_build.py                   # -> 13/13 checks passed

# 1) procedural demo inputs (reference image + 16-frame pose sequence)
python scripts/make_demo_inputs.py --out inputs/demo --frames 16 --size 256 --seed 7

# 2) full pipeline run -> outputs/run_small/{video.mp4,contact_sheet.png,shape_log.json,...}
python scripts/run_prototype.py --config configs/prototype_cpu_small.yaml --steps 8

# 3) the 19-configuration ablation matrix -> outputs/ablations/ablation_report.{csv,md}
python scripts/run_ablations.py
```

Measured on this machine (CPU-only, torch 2.14.1+cpu, 96×96, 8 frames, 8 DDIM steps,
seed 1234 — the E00 baseline configuration):

```
reference encode 33 ms | CLIP 8 ms | ReferenceNet 16 ms | Pose Guider 35 ms
denoise 619 ms (77 ms/step) | VAE decode 187 ms | total 1.40 s | peak RSS 386 MB
parameters: VAE 316,695 | CLIP 44,032 | ReferenceNet 681,920 | PoseGuider 182,832 | UNet 756,164
```

Full per-stage timings and tensor shapes land in
`outputs/run_small/{run_log.json, shape_log.json}`; the 19-configuration sweep is in
`outputs/ablations/ablation_report.md`.

---

## 3. Documentation map (all 18 deliverables)

| # | Deliverable | File |
|---|---|---|
| 1 | Complete architecture diagram | [`docs/01_architecture.md`](docs/01_architecture.md) |
| 2 | Module-by-module technical explanation | `docs/02`–`docs/06` |
| 3 | ReferenceNet implementation note | [`docs/02_module_reference_net.md`](docs/02_module_reference_net.md) |
| 4 | Pose Guider implementation note | [`docs/03_module_pose_guider.md`](docs/03_module_pose_guider.md) |
| 5 | Denoising UNet implementation note | [`docs/04_module_denoising_unet.md`](docs/04_module_denoising_unet.md) |
| 6 | VAE implementation note | [`docs/05_module_vae.md`](docs/05_module_vae.md) |
| 7 | CLIP implementation note | [`docs/06_module_clip.md`](docs/06_module_clip.md) |
| 8 | Attention-mechanism document | [`docs/07_attention_deep_dive.md`](docs/07_attention_deep_dive.md) |
| 9 | Tensor-shape reference table | [`docs/08_tensor_shapes.md`](docs/08_tensor_shapes.md) |
| 10 | Training & inference flowcharts | [`docs/09_training_and_inference.md`](docs/09_training_and_inference.md) |
| 11 | Minimal working prototype | `src/aan/`, `scripts/` (this README §2) |
| 12 | Configuration files | `configs/*.yaml` |
| 13 | Reproducible setup guide | [`docs/00_setup_guide.md`](docs/00_setup_guide.md) |
| 14 | Experiment & ablation report | [`docs/10_experiments_ablation.md`](docs/10_experiments_ablation.md) + `outputs/ablations/` |
| 15 | Debugging guide | [`docs/12_debugging_guide.md`](docs/12_debugging_guide.md) |
| 16 | Final presentation for the team | [`docs/15_final_presentation.md`](docs/15_final_presentation.md) |
| 17 | Glossary of diffusion/attention terms | [`docs/13_glossary.md`](docs/13_glossary.md) |
| 18 | Open research questions | [`docs/14_open_questions.md`](docs/14_open_questions.md) |
| — | Evaluation metrics | [`docs/11_evaluation_metrics.md`](docs/11_evaluation_metrics.md) |
| — | Final report (plain-language + evidence status) | [`docs/16_final_report.md`](docs/16_final_report.md) |

---

## 4. Code map

```
src/aan/
  config.py          typed YAML configs + cross-module validation (fails fast)
  layers.py          InflatedConv3d, ResBlock3D, TimeEmbedding, SelfAttention2D, up/down
  attention.py       ReferenceSpatialAttention | CrossAttention | TemporalAttention
  vae.py             diffusers AutoencoderKL wrapper (encode x0.18215, decode /0.18215)
  clip_encoder.py    frozen CLIPVisionModelWithProjection + preprocessing
  reference_net.py   2D trunk that *writes* the feature bank (5 tensors)
  pose_guider.py     production-shaped pose encoder (H/8, 3 stride-2 stages)
  denoising_unet.py  UNet composing spatial/reference/cross/temporal attention + skips
  pipeline.py        preprocessing, CFG, DDIM loop, frame-wise VAE decode, mp4 export
  metrics.py         PSNR/SSIM/temporal delta/flicker/hist-correlation (+ limitations)
  shape_log.py       module-boundary tensor logging -> shape_log.json
  utils.py           seeding, device/dtype, assert_shape, RSS tracking
scripts/
  make_demo_inputs.py  procedural reference image + COCO-18 pose sequence
  validate_build.py    13 self-checks (shape contracts, attention math, e2e)
  run_prototype.py     one full run -> video, contact sheet, logs, metrics
  run_ablations.py     experiment matrix -> CSV + Markdown report
reference/Moore-AnimateAnyone/   Apache-2.0 sources kept locally for citation
```

## 5. One-paragraph summary of the architecture

A reference image is encoded once by a frozen **VAE** (→ 4-channel latent, ×0.18215)
and once by a frozen **CLIP** image encoder (→ one semantic embedding token).
The latent additionally passes through **ReferenceNet**, a copy of the denoising
UNet without temporal layers, whose per-block hidden states are *stored* (the
"bank"). Every diffusion step feeds the noisy latent plus **Pose Guider** features
(added after `conv_in`) into the **Denoising UNet**, where each transformer block
performs (1) *spatial attention whose keys/values are the concatenation of the
current frame tokens and the ReferenceNet bank* — this is what preserves identity,
(2) *cross-attention* against the CLIP embedding — global semantics, and (3)
*temporal attention* across frames (AnimateDiff-style motion modules) — continuity.
After 20–30 DDIM steps the clean latent is decoded frame-by-frame by the VAE into RGB.