# 08 — Tensor-shape reference (symbolic + production + measured)

```
B = batch size          F = number of frames
H, W = input height/width in pixels      h, w = latent height/width (= H/8, W/8)
C = latent channels (4)                  C_ctx = CLIP projection width (1024 prod / 64 ours)
C_i  = UNet channels at level i          N_i = h_i · w_i  (tokens per frame at level i)
```

Layout convention used everywhere in this project and in the reference
implementation: **video tensors are `[B, C, F, H, W]` (channels before frames)**;
per-frame images are `[B, C, H, W]`; a batch of frames folded for attention is
`[B·F, N, C]`; attention contexts are `[B, S, C_ctx]`.

---

## 1. Master table — every major stage

| # | stage | symbolic | production (784×512, F=24, SD 1.5) | measured prototype (96×96, F=8) | dtype / device | grad |
|---|---|---|---|---|---|---|
| 1 | reference image (preprocessed) | `[B,3,H,W]` | `[1,3,784,512]` | `[1,3,96,96]` (−0.80 … +0.90) | f32 / cpu | no |
| 2 | CLIP pixel input | `[B,3,224,224]` | `[1,3,224,224]` | `[1,3,32,32]` | f32 / cpu | no |
| 3 | CLIP image embedding | `[B,D]` | `[1,1024]` | `[1,64]` (−2.74 … +2.25) | f32 / cpu | no (frozen) |
| 4 | cross-attention context | `[B,1,D]` | `[1,1,1024]` | `[1,1,64]` | f32 / cpu | no |
| 5 | reference latent | `[B,4,h,w]` | `[1,4,98,64]` | `[1,4,12,12]` (−0.30 … +0.09) | f32 / cpu | no |
| 6 | ReferenceNet bank (per block) | `[B,N_i,C_i]` | 4 scales, e.g. `[1,6272,320]`, `[1,1568,640]`, … | `[1,144,32]`, `[1,36,64]`, `[1,9,64]`, `[1,36,64]`, `[1,144,32]` | f32 (f16 for the prod bank copy) | no |
| 7 | pose map sequence | `[B,3,F,H,W]` | `[1,3,24,784,512]` | `[1,3,8,96,96]` (0 … 1) | f32 / cpu | no |
| 8 | pose features (Pose Guider out) | `[B,C_inner,F,h,w]` | `[1,320,24,98,64]` | `[1,32,8,12,12]` (−0.036 … +0.040) | f32 / cpu | no (trainable in prod) |
| 9 | UNet input after `conv_in` (+pose) | `[B,C_0,F,h,w]` | `[1,320,24,98,64]` | `[1,32,8,12,12]` | f32 / cpu | no |
| 10 | noisy video latent | `[B,4,F,h,w]` | `[1,4,24,98,64]` | `[1,4,8,12,12]` (−4.07 … +3.90) | f32 | no |
| 11 | UNet input under CFG | `[2B,4,F,h,w]` | `[2,4,24,98,64]` | `[2,4,8,12,12]` | f32 | no |
|  12 | attention tokens (per block, folded) | `[B·F,N_i,C_i]` | `[24,6272,320]` (level 0) | `[8,144,32]` | f32 | no |
| 13 | reference K/V (concatenated) | `[B·F, 2N_i, C_i]` | `[24,12544,320]` | `[8,288,32]` | f32 | no |
| 14 | temporal attention tokens | `[B·N_i,F,C_i]` | `[6272,24,320]` | `[144,8,32]` | f32 | no |
| 15 | predicted noise (ε̂ or v̂) | `[B,4,F,h,w]` | `[1,4,24,98,64]` | `[1,4,8,12,12]` (−3.59 … +3.57) | f32 | no |
| 16 | denoised latent x₀ | `[B,4,F,h,w]` | `[1,4,24,98,64]` | `[1,4,8,12,12]` (±~50 — see §4) | f32 | no |
| 17 | decoded video (pre-normalisation) | `[B,3,F,H,W]` | `[1,3,24,784,512]` | `[1,3,8,96,96]` (−2.74 … +1.85) | f32 | no |
| 18 | output frames (saved) | `[F,3,H,W]` | `[24,3,784,512]` in `[0,1]` | `[8,3,96,96]` in `[0,1]` | f32→uint8 | no |

`B=1` everywhere except CFG, which doubles the batch *for the UNet call only*
(latents, CLIP context and pose features are duplicated; the reference bank is
broadcast — `attention.py:137-141`).

---

## 2. Module-boundary contracts (what the code asserts)

| boundary | assertion | failure message |
|---|---|---|
| VAE encode | 4-D `[B,3,H,W]`, `H%8==0` | `VaeWrapper.encode expects [B,3,H,W]; got …` |
| VAE decode (video) | 5-D `[B,4,F,h,w]` | `expected [B,4,F,h,w]; got …` |
| Pose Guider | 5-D `[B,3,F,H,W]`, `H%8==0` | `PoseGuider expects [B,3,F,H,W]; got …` |
| ReferenceNet | 4-D `[B,4,h,w]` | `ReferenceNet expects [B,4,h,w]; got …` |
| Reference attention | batch + channel match | `reference batch 1 != frame batch 2` / `reference channel width 8 != frame width 16` |
| Cross-attention | 4-D `[B,F,N,C]` | `CrossAttention expects [B,F,N,C]; got …` |
| Temporal attention | 5-D `[B,C,F,h,w]`, `F ≤ max_frames` | `frame count 24 exceeds temporal max_frames 8; increase motion.max_frames …` |
| UNet | pose batch/frame/grid match | `pose features … must match batch/frames of sample …`, `pose feature grid … != latent grid …` |
| config | cross-module contracts | `clip.projection_dim … must equal unet.context_dim …`, `reference_net.base_channels must equal unet.base_channels …` |

---

## 3. Value ranges

| tensor | expected range | why |
|---|---|---|
| reference image | `[-1, 1]` | VAE convention (`pixel*2-1`) |
| pose maps | `[0, 1]` | `np.array(resize(img))/255`, **no** normalisation |
| CLIP pixels | ≈ `[-1.8, +2.1]` after CLIP normalisation | CLIP mean/std |
| latent (post-scaling) | ≈ `[-4, +4]` with a trained VAE | 0.18215 scaling |
| latent (our tiny random VAE) | `[-0.3, +0.1]` measured | random encoder, not a trained prior |
| ε̂ prediction | ≈ `[-4, +4]` | matches the noise prior |
| decoded RGB (raw) | ≈ `[-1, 1]` | decoder output |
| saved frames | `[0, 1]` → uint8 | `(x/2 + 0.5).clamp(0,1)` |

---

## 4. Measured anomalies and what they teach

Two observations from the measured runs that look like bugs but are not:

1. **`predicted_noise` was exactly 0 in the first prototype version.**
   Cause: SD's convention `zero_module(conv_out)` combined with *random* weights
   ⇒ the denoiser outputs zeros, the scheduler never moves, and every
   conditioning channel (pose, reference, CLIP) has literally no effect.
   Production keeps zero-init because training follows; the educational configs
   therefore expose `unet.zero_init_out` / `pose_guider.zero_init_out` (default
   **false**), and `validate_build.py` asserts the mechanism in *both* states
   (`PoseGuider(zero_init_out=True)` → all zeros; `False` → non-zero signal).
2. **final latents reach ±50 with the tiny random VAE/UNet.** Cause: DDIM
   computes `x̂₀ = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t`; at the first steps
   `ᾱ_t ≈ 4e−5` (t ≈ 999), so any error in an *untrained* ε̂ is amplified by
   `1/√ᾱ_t ≈ 150`. A trained model keeps latents O(1); an untrained one does not.
   Diagnostic rule for the team: **unbounded latent magnitudes ⇒ ε-prediction is
   not calibrated** (untrained weights, wrong β schedule, wrong scaling factor,
   or a timestep/ε̂ sign-convention bug).

---

## 5. How to regenerate this table

```bash
python scripts/run_prototype.py --config configs/prototype_cpu_small.yaml --steps 6
python -c "import json;d=json.load(open('outputs/run_small/shape_log.json'));\
[print(r['name'], r['shape'], r['dtype'], r['min'], r['max']) for r in d['records'] if r['dtype']!='params']"
```

Every module boundary calls `ShapeLogger.record(...)`, which stores shape, dtype,
device, `requires_grad`, min/max and `numel` — so this table is checkable against
a real run instead of being documentation folklore.