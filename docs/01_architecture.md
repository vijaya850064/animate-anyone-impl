# 01 — Architecture of the Animate Anyone core model

> Scope: the **core diffusion/animation model** (ReferenceNet, Pose Guider,
> Denoising UNet, VAE, CLIP, attention) — not the pose detector (DWPose/OpenPose),
> not datasets, not multi-GPU training engineering.
>
> Sources: paper arXiv:2311.17117v3 (CVPR 2024) and the Apache-2.0 reproduction
> `Moore-AnimateAnyone` vendored under `reference/Moore-AnimateAnyone/`
> (citations below are `path:line` into that folder).

---

## 1. End-to-end data flow (inference)

```
            ┌──────────────── FROZEN / SHARED (SD checkpoints) ────────────────┐
            │                                                                 │
 REFERENCE   │  ┌──────────────┐                            ┌───────────────┐   │
 IMAGE       │  │   VAE ENC.   │                            │ CLIP VISION   │   │
 (1xRGB,     ├─▶│  E(x0)       │                            │ ENCODER       │   │
  H x W)     │  │  ↓ z_ref     │                            │ ↓             │   │
            │  │ [1,4,h,w]    │                            │ [1,D] →[1,1,D]│  │
            │  │ ×0.18215     │                            └───┬───────┬───┘   │
            │  └──────┬───────┘                          (2) cross │       │    │
            │         │                                       │    K/V│       │
            │         │ (1) scale                     ┌────────▼───────▼──┐   │
            │         ▼                                │  REFERENCE  BANK   │   │
            │  ┌────────────────────────────────────┐  │ (stored once per   │   │
            │  │ REFERENCE NET  (2D UNet, NO       │  │  generation, K/V   │   │
            │  │ temporal layers) forward(z_ref,    │  │  augmentation)     │   │
            │  │ t = 0); WRITE bank[i]=norm(h_i)    │  └────────┬───────────┘   │
            │  └────────────────────────────────────┘           │               │
            │                                                       │               │
 POSE SEQ.  │  ┌───────────────┐                                    │               │
 (F x RGB   ├─▶│  POSE GUIDER  │  (lightweight conv stack)           │               │
  pose maps,│  │  H/8 x W/8    │                                    │               │
  H x W,    │  │  [0,1] input  │                                    │               │
  [0,1])    │  └───────┬───────┘                                    │               │
            │          │ (3) ADD after conv_in                      │               │
            │          │                                           ▼               │
            │          │         ┌───────────────────────────────────────────────┐   │
 RANDOM     │          │         │            DENOISING UNET                     │   │
 NOISE      │          │         │  conv_in ──(+pose feat)──┐                     │   │
 x_T~N(0,I) ┼──────────┤         │  down0  attn0 ──────────┤ ref bank0 [144,32] │   │
 [1,4,F,h,w]│          │         │  down1  attn1 ──────────┤ ref bank1  [36,64] │   │
            │          │         │  mid    attn2 ──────────┤ ref bank2   [9,64] │   │
            │          │         │  up0    attn3 ──────────┤ ref bank3  [36,64] │   │
            │          │         │  up1    attn4 ──────────┤ ref bank4 [144,32] │   │
            │          │         │  conv_out → [B,4,F,h,w] │                     │   │
            │          │         └──────────┬───────────────┘                     │   │
            │          │                    │ (4) predicted ε̂ (or v̂)              │   │
            │          │                    ▼                                    │   │
            │          │            ┌──────────────────┐                          │   │
            │          │            │ scheduler.step   │  DDIM, β .00085→.012,     │   │
            │          │            │ (DDIM, steps_offset=1)  CFG: ε̂ = ε_u + g(ε_c−ε_u)│
            │          │            └────────┬─────────┘                          │   │
            │          │                     │ x_{t-1}  (20–30 steps)              │   │
            │          │                     └── loop ──┘                         │   │
            │          │                                                        │   │
            │  ┌───────┴───────┐                                                │   │
            │  │   VAE DEC. D  │◀── x_0 [B,4,F,h,w], decoded FRAME-BY-FRAME      │   │
            │  │  ÷0.18215     │    (memory saver, see 05_module_vae.md)         │   │
            │  └───────┬───────┘                                                │   │
            └──────────┼────────────────────────────────────────────────────────┘   │
                       ▼                                                           
            GENERATED VIDEO  [F,3,H,W] in [-1,1]  →  mp4                            │
                       ▲                                                            │
                       └── x_t  (noisy latent video, "random noise → structure") ──┘
```

**Tensor direction summary** (prototype values for 96×96, F=8, latent 12×12):

| stage | tensor | producer → consumer |
|---|---|---|
| reference image | `[1,3,96,96]` | user → VAE encoder, CLIP, ReferenceNet |
| reference latent | `[1,4,12,12]` | VAE encoder → ReferenceNet |
| CLIP context | `[1,1,64]` | CLIP → every cross-attention |
| reference bank | 5 × `[1,{144,36,9,36,144},{32,64,64,64,32}]` | ReferenceNet → spatial attention K/V |
| pose maps | `[1,3,8,96,96]` | pose detector (or demo generator) → Pose Guider |
| pose features | `[1,32,8,12,12]` | Pose Guider → UNet `conv_in` output |
| noisy latent | `[1,4,8,12,12]` | sampler → UNet |
| UNet input (CFG) | `[2,4,8,12,12]` | CFG batch duplication → UNet |
| predicted noise | `[1,4,8,12,12]` | UNet → scheduler |
| denoised latent | `[1,4,8,12,12]` | scheduler → VAE decoder |
| decoded video | `[1,3,8,96,96]` | VAE decoder → mp4 writer |

### The four information channels

| channel | carrier | injection point | consumer |
|---|---|---|---|
| **Appearance (spatial detail)** | ReferenceNet bank `[B, h·w, C]` | every spatial-attention call | K/V augmentation |
| **Semantics (global)** | CLIP embedding `[B,1,D]` | every block, every step | cross-attention K/V |
| **Motion (spatially aligned)** | Pose Guider output `[B,C_inner,F,h,w]` | **added** to `conv_in(sample)` | ResNet/attention stack |
| **Temporal continuity** | motion modules, frames as sequence | after each spatial block | temporal attention |

---

## 2. Who processes what (responsibility table)

| Data | Module(s) that process it | Reference implementation | This prototype |
|---|---|---|---|
| **Reference image** | VAE encoder, CLIP image encoder, ReferenceNet | `pipeline_pose2vid.py:362-369` (VAE), `:318-324` (CLIP), `:399-406` (ReferenceNet write + `reader.update`) | `pipeline.py:generate()` steps 1–3 |
| **Pose sequence** | resize → ÷255 → Pose Guider → addition inside UNet | `pipeline_pose2vid.py:371-386`, `pose_guider.py`, `unet_3d.py:503-505` | `pose_guider.py`, `denoising_unet.py:forward` |
| **Noisy latent video** | Denoising UNet (+ scheduler for the time axis) | `pipeline_pose2vid.py:412-432` | `pipeline.py` loop |
| **Semantic image features** | CLIP once → cross-attention every block/step | `mutual_self_attention.py:236-250` (`attn2`) | `attention.py:CrossAttention` |
| **Spatial appearance features** | ReferenceNet once → spatial attention K/V every block/step | `mutual_self_attention.py:139-159`, `:304-341` | `reference_net.py`, `attention.py:ReferenceSpatialAttention` |
| **Temporal frame relations** | motion modules (temporal self-attn + temporal PE) | `motion_module.py`, `inference_v1.yaml` (`motion_module_resolutions: [1,2,4,8]`) | `attention.py:TemporalAttention` |
| **Final decoded video** | VAE decoder, frame-by-frame | `pipeline_pose2vid.py:decode_latents` | `vae.py:decode_video_framewise` |
| **Timestep** | sinusoidal embedding + MLP in every ResBlock | `unet_3d.py:482-488` | `layers.py:TimeEmbedding` |

---

## 3. Block-level view of the Denoising UNet

```
x_t  [B,4,F,h,w]
 │
 ├─ conv_in (InflatedConv3d 4→C0) ──► x  [B,C0,F,h,w]
 ├─ (+) pose features [B,C0,F,h,w]        (production: unet_3d.py:503-505)
 │
 ├── down level 0 ──────────────────────────────────────────────┐
 │     ResBlock3D(C0) + temb                                     │ skip0
 │     ┌─ spatial attn : Q=frame tokens, K/V=[frame ‖ bank0]   │
 │     ├─ cross attn   : Q=frame tokens, K/V=CLIP [1,1,D]        │
 │     └─ temporal attn: over F at each spatial position         │
 │     Downsample (stride 2, C0→C1)                             │
 ├── down level 1  (same pattern, bank1) ────────────────┐        │ skip1
 │     Downsample (stride 2, C1→C2)                    │        │
 ├── mid          (bank2)                               │        │
 │     up0: Upsample → cat(skip1) → ResBlock(C2+C1→C1) → attn(bank3)
 │     up1: Upsample → cat(skip0) → ResBlock(C1+C0→C0) → attn(bank4)
 └─ GroupNorm → SiLU → conv_out (C0→4)  →  ε̂ / v̂  [B,4,F,h,w]
```

Two distinct routes carry information to the decoder — a frequent source of confusion:

* **Skip connections** (`torch.cat([x, skip], dim=1)`) carry *the noisy-video
  features* from encoder to decoder at the same resolution: spatial detail of the
  video being generated.
* **Reference attention** carries *the reference image's* features (different
  origin: ReferenceNet, timestep 0). It is the only path by which the reference
  character's appearance reaches the decoder.

---

## 4. Production vs this prototype (explicit deltas)

| Aspect | Production (paper + reference impl) | This prototype | Reason |
|---|---|---|---|
| Channels | UNet `(320,640,1280,1280)`, 4 scales | `(32,64,64)`, 3 scales | CPU runtime |
| Frames | 24 per clip (PE window 24/32) | 8 (max_frames 32) | CPU runtime |
| Pose fusion | add after `conv_in` (320 ch) | add after `conv_in` (32 ch) | same mechanism |
| Reference attention | all blocks (`fusion_blocks="full"`), writer/reader paired by sorting channel width | 5 stacks paired by construction | same mechanism, explicit wiring |
| CFG + reference | reference attention only in the **conditional** half (`uc_mask`, `mutual_self_attention.py:168-181`) | bank broadcast to both halves | documented simplification |
| Uncond context | `torch.zeros_like(embeds)` | same | identical |
| UNet output init | `zero_module(conv_out)` | configurable; `false` in educational configs | zero-init + random weights ⇒ inert sampler |
| Scheduler | DDIM (v1) / DDIM + v-prediction + zero-SNR (v2) | DDIM, ε-prediction, same betas | v1 config |
| ReferenceNet training | trainable except `up_blocks.3` (`train_stage_1.py:309-313`) | inference-only | no training here |
| Long videos | windowed pipeline (`pipeline_pose2vid_long.py`) | single clip | scope |

All deltas are repeated in [`docs/16_final_report.md`](16_final_report.md) with their
evidence status (implemented / inferred / simplified / validated / open).