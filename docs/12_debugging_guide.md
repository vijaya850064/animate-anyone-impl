# 12 — Debugging guide (shape errors, quality problems, performance)

Every entry: **symptom → likely cause → diagnostic → fix → trade-off**. Items
marked *(hit in this repo)* are bugs we actually hit while building the
prototype, with the assertion that caught them.

---

## A. Shape errors

| symptom | likely cause | diagnostic | fix | trade-off |
|---|---|---|---|---|
| `expected [B,3,H,W]; got 5 dims` *(hit)* | passing `[B,C,F,H,W]` to a 2D module | print the shape at the boundary; `ShapeLogger` records it | `rearrange(x, "b c f h w -> (b f) c h w")` (frame-wise) | frame-wise loses temporal ops — use motion modules for that |
| `expected 5 dims` on VAE decode | frame axis lost when stacking per-frame latents | check `cat(..., dim=2)` — per-frame latents are 4-D | `unsqueeze(2)` before stacking *(hit in `run_ablations.py`)* | none |
| `pose feature grid (96,96) != latent grid (12,12)` | pose features taken at pixel resolution, or wrong number of stride-2 stages | compare `pose_fea.shape[-2:]` with `latents.shape[-2:]` | Pose Guider levels so `down_factor == 8`; config validation warns | one extra stage costs a little compute |
| `pose features … must match batch/frames of sample` | pose clip length ≠ latent clip length | compare `len(pose_images)` with `cfg.input.frames` | trim/extend the pose list *before* the loop; assert in the UNet | truncating loses motion |
| `reference batch 1 != frame batch 2` *(hit)* | CFG doubles the UNet batch; the bank has batch 1 | compare `model_in.shape[0]` with `bank[0].shape[0]` | broadcast the bank (`attention.py:137-141`); production does it via `uc_mask` | none |
| `reference channel width 8 != frame width 16` *(hit)* | ReferenceNet and UNet ladders diverge | `[b.shape[-1] for b in bank]` vs UNet level widths | keep `reference_net.base_channels == unet.base_channels` (enforced by `validate_config`) | couples the two configs |
| `clip.projection_dim … must equal unet.context_dim` | CLIP width ≠ UNet cross-attention width | message prints both numbers | set `unet.context_dim = clip.projection_dim` | none |
| `frame count F exceeds temporal max_frames` | PE buffer too small for the clip | message names both values | raise `motion.max_frames` (prod 24/32) or shorten the clip | longer windows cost quadratic temporal attention |
| `[F,C,H,W]` silently read as `[C,F,H,W]` | layout confusion between per-frame and video tensors | `assert x.shape[0] == cfg.input.frames` | convert once at the boundary: `.permute(1,0,2,3)` *(hit in `pipeline.py`)* | one extra copy; make it explicit and logged |
| off-by-8 grid artefacts in the decoder | `H`/`W` not divisible by 8 | `H % 8`, `W % 8` | use `H = 98·8 = 784` (as the reference script does) or pad and unpad | padding bookkeeping |

---

## B. Quality problems

| symptom | likely cause | diagnostic | fix | trade-off |
|---|---|---|---|---|
| **identity changes across frames** | reference attention too weak / bank not applied in decoder blocks | A/B E00 vs E01 vs E05b; bank shapes in the log | restore the bank; `fusion_blocks="full"`; raise `strength` | more attention memory |
| **face distorted, body fine** | CLIP too coarse at the finest scale | compare with/without ReferenceNet at `down_blocks.0` | include all blocks in reference attention | cost |
| **clothing/hairstyle changes** | over/under-conditioning; temporal drift | per-frame contact sheet; flicker index | temporal modules + reference attention; check pose alignment | speed |
| **pose not followed** | pose features zero (zero-init `conv_out`!) *(hit)* or wrong fusion | is `pose_fea` all zeros? see `shape_log.json` | `pose_guider.zero_init_out=false` for ablation, `true` for training | training stability |
| **frozen motion** | pose branch ignored; guidance too low; temporal-window mismatch | compare driving-pose motion vs output `temporal_delta` | raise guidance/pose strength; check detector output | stiff, over-committed motion |
| **flicker** | weak/absent temporal attention; short window | flicker index + per-frame crops | enable motion modules, longer window | memory |
| **blurry frames** | VAE 8× ceiling; too few steps; missing `÷0.18215` | VAE round-trip PSNR; decoded min/max | fix scaling; more steps; `sd-vae-ft-mse` | speed |
| **colour shift / grey output** | missing `video/2 + 0.5`, or wrong latent scaling | `ref_latent.std()` ≈ 1; decoded range ≈ `[-1,1]` | apply/remove the conversion where it belongs | none |
| **background deformation** | over-conditioning (reference background copied) | attention weights; `strength` 0.5 vs 1.0 | fewer reference blocks, lower strength | identity fidelity |
| **unwanted reference objects** | bank carries reference background/props | visual diff ref vs output | tight crops on references; `fusion_blocks="midup"` | may lose appearance |
| **output is pure noise** *(hit)* | zero-init output conv + random weights ⇒ ε̂ ≡ 0 | `predicted_noise_first_step` min=max=0 | set `unet.zero_init_out: false` for ablations | production keeps `true` **after** training |
| **latents explode (±50)** *(hit)* | untrained ε̂ amplified by `1/√ᾱ_t` at large t | `denoised_video_latents` min/max | expected with random weights; with a trained model it signals schedule/scaling bugs | — |
| **NaNs in VAE decode** | fp16 CPU conv overflow | `torch.isfinite(video)` | `runtime.dtype: float32` on CPU (E11 quantifies fp16) | speed |

---

## C. Performance problems

| symptom | likely cause | diagnostic | fix | trade-off |
|---|---|---|---|---|
| **CUDA OOM at 512×512×24 frames** | `N²` attention weights + 24-frame activations | per-stage memory; `torch.cuda.max_memory_allocated()` | xformers/SDPA; `enable_attention_slicing()`; gradient checkpointing (training); shorter clips | slower |
| **OOM in the VAE decode** | decoder activations at full resolution | confirm frame-wise decode is used | `decode_video_framewise`; `enable_vae_tiling()` | seam artefacts with tiling |
| **slow attention** | reference attention doubles keys; manual softmax materialises `N²` | per-stage timings in `run_log.json` | SDPA/xformers instead of the manual formula | memory vs speed |
| **excessive latency** | steps × frames × resolution, ×2 with CFG | `denoise_per_step_mean` (78 ms/step measured) | fewer steps; CFG off (halves it); lower resolution | quality |
| **large intermediate feature storage** | bank + attention maps retained across steps | bank dtype (prod copies to fp16) | store banks in fp16; `clear()` after generation | small precision cost |
| **video too long** | windowing needed | `F > temporal max_frames` error | chunk into windows and blend (`pipeline_pose2vid_long.py`) | seam artefacts |

---

## D. Environment / reproducibility issues

| symptom | likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: aan` | `src/` layout | `pip install -e .` or add `src` to `sys.path` (the scripts do) |
| different results after reinstall | torch/diffusers drift | pin `requirements.txt`; `run_log.json` records resolved versions |
| CPU fp16 crash or slowness | fp16 convolutions on CPU | `runtime.dtype: float32` (E11 measures fp16 explicitly) |
| seed ignored | generator created outside the seeded scope | `seed_everything()` + explicit `torch.Generator` (`pipeline.py` step 5) |
| outputs missing from git | `.gitignore` covers `outputs/`, `inputs/` | keep generated media out of the repo (privacy + size) |

---

## E. Fast triage order

1. `python scripts/validate_build.py` — 13 contract checks in ~1 s;
2. `outputs/*/shape_log.json` — look for `min == max == 0` (an inactive branch)
   or absurd ranges (uncalibrated ε̂);
3. `outputs/*/run_log.json` — `timings` and `peak_rss_mb` separate "too slow"
   from "wrong";
4. only then look at the video — most visual complaints above have a numerical
   signature first.

---

## F. Audit conditioning branches at the *injection point*, not at the output

Measured on this prototype (same weights, same seed, one factor varied):

| perturbation | mean abs Δ of the decoded clip | conclusion |
|---|---|---|
| swap the reference image (seed 7 → 99) | 3.4e−3 | branch live |
| disable the ReferenceNet bank | 2.6e−2 | branch live |
| `reference_attention_strength = 0.0` | **2.6e−2 (identical to the row above)** | equivalence property holds to 6 decimals |
| disable CLIP cross-attention | 6.7e−2 (largest) | branch live |
| disable the Pose Guider | 1.6e−3 | branch live |
| disable temporal attention | **0.0 exactly** | expected: motion modules are zero-initialised ⇒ identity at init |
| disable CFG (`g=1`) | 4.7e−2 | branch live |
| swap the pose sequence (reversed) | 2e−6 | *see below* |

The pose row looks alarming (reversing every pose barely moves the output), so
measure where the signal enters:

```
pose maps (reversed sequences) differ by        1.1 % of the [0,1] pixel range
pose features after the (random-weight) Guider   0.039 % relative difference
pose features injected at the injection point    21 % of the activation scale  → branch is LIVE
final decoded RGB                                2e−6
```

Interpretation: injection happens exactly as production does it, but with
**untrained** weights the conv stack acts as a strong smoother, so a
high-frequency stroke displacement shrinks by ~28× before entering the UNet, and
the remaining difference is then crushed by the random denoiser plus VAE
smoothing. Practical rules:

* an output-level "does perturbation X change the video?" test is only valid for
  branches with a *multiplicative/projection* path (attention K/V). For
  **additive** conditioning, audit at the injection point (or the module output);
* a near-zero output delta for the pose channel is **not** evidence of a wiring
  bug here — the wiring assertion in `denoising_unet.py` and the zero-init
  behaviour test in `validate_build.py` are the authoritative checks;
* to make the pose channel numerically dominant in a demo you would have to
  rescale it — deliberately *not* done here, because it would hide the very
  effect (small additive conditioning) that the production design relies on
  training to learn the right scale for.