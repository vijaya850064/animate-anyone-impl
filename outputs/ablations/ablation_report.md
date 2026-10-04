# Animate Anyone study — ablation report (automatically generated)

* matrix: `C:\Users\manik\Downloads\newone\configs\ablation_matrix.yaml`
* base config: `96x96`, seed 1234, CPU-only run
* baseline run: `E00_baseline` (1.404 s)

> **Reading guide.** All numbers below come from an *educational prototype with randomly initialised
> weights*: they measure the data path, shapes, runtime and memory of each configuration — **not**
> animation quality. Identity/pose/flicker quality requires the trained production checkpoints.
> `temporal_delta` and `flicker_index` are structural proxies computed on the decoded frames;
> `vae_psnr`/`vae_ssim` measure VAE round-trip fidelity (the one quality metric that is meaningful
> even with random weights, because it compares the decoder against its own encoder).

| id | frames | res | steps | CFG | runtime (s) | Δruntime | peak RSS (MB) | params | temporal_delta | flicker | hist_corr | VAE PSNR | VAE SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E00_baseline | 8 | 96x96 | 8 | 3.5 | 1.40 | +0.0% | 386 | 1,981,643 | 0.2084 | 0.0071 | 0.565 | 18.23 | 0.040 |
| E01_no_reference | 8 | 96x96 | 8 | 3.5 | 1.24 | -11.9% | 388 | 1,981,643 | 0.2087 | 0.0058 | 0.577 | 18.19 | 0.035 |
| E02_no_clip | 8 | 96x96 | 8 | 3.5 | 0.94 | -33.1% | 414 | 1,981,643 | 0.2177 | 0.0084 | 0.532 | 18.36 | 0.066 |
| E03_no_pose | 8 | 96x96 | 8 | 3.5 | 1.05 | -24.9% | 413 | 1,981,643 | 0.2084 | 0.0072 | 0.564 | 18.24 | 0.040 |
| E04_no_temporal | 8 | 96x96 | 8 | 3.5 | 0.85 | -39.4% | 404 | 1,981,643 | 0.2084 | 0.0071 | 0.565 | 18.23 | 0.040 |
| E05_weak_reference_0p5 | 8 | 96x96 | 8 | 3.5 | 1.29 | -8.0% | 404 | 1,981,643 | 0.2084 | 0.0065 | 0.572 | 18.21 | 0.040 |
| E05b_zero_reference_0p0 | 8 | 96x96 | 8 | 3.5 | 1.12 | -19.9% | 403 | 1,981,643 | 0.2087 | 0.0058 | 0.577 | 18.19 | 0.035 |
| E06_steps_2 | 8 | 96x96 | 2 | 3.5 | 0.90 | -35.6% | 393 | 1,981,643 | 0.1770 | 0.0070 | 0.472 | 19.16 | 0.076 |
| E06b_steps_16 | 8 | 96x96 | 16 | 3.5 | 2.59 | +84.5% | 393 | 1,981,643 | 0.2097 | 0.0070 | 0.567 | 18.23 | 0.042 |
| E07_guidance_1p0 | 8 | 96x96 | 8 | 1.0 | 1.36 | -3.1% | 393 | 1,981,643 | 0.2170 | 0.0076 | 0.544 | 18.38 | 0.069 |
| E07b_guidance_7p5 | 8 | 96x96 | 8 | 7.5 | 1.36 | -3.3% | 415 | 1,981,643 | 0.1906 | 0.0089 | 0.583 | 18.08 | 0.013 |
| E08_pose_degraded | 8 | 96x96 | 8 | 3.5 | 0.99 | -29.4% | 417 | 1,981,643 | 0.2084 | 0.0071 | 0.565 | 18.23 | 0.040 |
| E09_frames_4 | 4 | 96x96 | 8 | 3.5 | 0.61 | -56.8% | 403 | 1,981,643 | 0.2051 | 0.0087 | 0.578 | 18.19 | 0.064 |
| E09b_frames_16 | 16 | 96x96 | 8 | 3.5 | 1.84 | +30.8% | 406 | 1,981,643 | 0.2068 | 0.0094 | 0.568 | 18.23 | 0.057 |
| E10_res_64 | 8 | 64x64 | 8 | 3.5 | 0.70 | -49.9% | 400 | 1,981,643 | 0.2018 | 0.0103 | 0.587 | 18.12 | 0.051 |
| E10b_res_128 | 8 | 128x128 | 8 | 3.5 | 1.55 | +10.3% | 478 | 1,981,643 | 0.2080 | 0.0035 | 0.568 | 18.24 | 0.060 |
| E11_vae_fp16 | 8 | 96x96 | 8 | 3.5 | 17.07 | +1115.5% | 412 | 1,981,643 | 0.2084 | 0.0071 | 0.565 | 18.23 | 0.040 |
| E12_slow_motion | 8 | 96x96 | 8 | 3.5 | 1.03 | -26.9% | 439 | 1,981,643 | 0.2084 | 0.0071 | 0.565 | 18.23 | 0.040 |
| E13_pose_fusion_concat | 8 | 96x96 | 8 | 3.5 | 0.95 | -32.1% | 419 | 1,950,511 | 0.2117 | 0.0090 | 0.393 | 18.65 | 0.088 |

## Per-experiment interpretation

### E00_baseline — All components enabled (reference + CLIP + pose + temporal, CFG 3.5, 8 steps).

* config changes: `{}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.40 s (+0.0% vs baseline), peak RSS 386 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Reference configuration. All structural numbers are baselines; identity/pose quality needs trained weights.

### E01_no_reference — ReferenceNet disabled -> spatial attention falls back to plain self-attention.

* config changes: `{"runtime.use_reference": false}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.24 s (-11.9% vs baseline), peak RSS 388 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Removing ReferenceNet removes appearance injection. Trained models show identity drift/clothing distortion; here we can only verify the code path (bank=None) and the runtime/memory delta.

### E02_no_clip — CLIP conditioning disabled -> cross-attention skipped in all blocks.

* config changes: `{"runtime.use_clip": false}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.94 s (-33.1% vs baseline), peak RSS 414 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Cross-attention is skipped entirely (context=None is an exact identity in our CrossAttention). Trained models lose semantic/global appearance consistency (e.g. character category, palette).

### E03_no_pose — Pose guider disabled -> no motion conditioning at all.

* config changes: `{"runtime.use_pose": false}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.05 s (-24.9% vs baseline), peak RSS 413 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: No pose conditioning: a trained model ignores the driving signal and produces pose-unrelated motion. Here the pose tensor is simply not computed (timing/param delta is the measurable effect).

### E04_no_temporal — Temporal (motion) modules disabled -> no cross-frame attention.

* config changes: `{"runtime.use_temporal": false}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.85 s (-39.4% vs baseline), peak RSS 404 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Temporal (motion) modules disabled. In production they are zero-initialised, so an untrained model shows no flicker difference; with trained weights this experiment is the primary flicker measurement.

### E05_weak_reference_0p5 — Reference attention strength 0.5 (blend with plain self-attention).

* config changes: `{"runtime.reference_attention_strength": 0.5}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.29 s (-8.0% vs baseline), peak RSS 404 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Reference attention strength 0.5 / 0.0 linearly blends reference-augmented and plain self-attention outputs. 0.0 must be numerically equivalent to removing the bank (property verified by validate_build.py).

### E05b_zero_reference_0p0 — Reference attention strength 0.0 (equivalent to disabling the bank, keeps module shapes).

* config changes: `{"runtime.reference_attention_strength": 0.0}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.12 s (-19.9% vs baseline), peak RSS 403 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Reference attention strength 0.5 / 0.0 linearly blends reference-augmented and plain self-attention outputs. 0.0 must be numerically equivalent to removing the bank (property verified by validate_build.py).

### E06_steps_2 — Only 2 DDIM steps (speed vs quality).

* config changes: `{"diffusion.num_inference_steps": 2}` | options: `{}`
* latent grid 12x12, 2 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.90 s (-35.6% vs baseline), peak RSS 393 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Fewer DDIM steps trade quality for speed (compute scales ~linearly with steps).

### E06b_steps_16 — 16 DDIM steps (quality vs speed).

* config changes: `{"diffusion.num_inference_steps": 16}` | options: `{}`
* latent grid 12x12, 16 steps, CFG 3.5 (active=True), seed 1234
* runtime 2.59 s (+84.5% vs baseline), peak RSS 393 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Fewer DDIM steps trade quality for speed (compute scales ~linearly with steps).

### E07_guidance_1p0 — Guidance 1.0 -> classifier-free guidance disabled (single forward pass).

* config changes: `{"diffusion.guidance_scale": 1.0}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 1.0 (active=False), seed 1234
* runtime 1.36 s (-3.1% vs baseline), peak RSS 393 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Guidance 1.0 disables CFG (single forward pass -> ~2x faster); high guidance increases conditioning strength and can over-saturate colours in trained models.

### E07b_guidance_7p5 — Guidance 7.5 -> stronger conditioning signal (over-saturation risk).

* config changes: `{"diffusion.guidance_scale": 7.5}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 7.5 (active=True), seed 1234
* runtime 1.36 s (-3.3% vs baseline), peak RSS 415 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Guidance 1.0 disables CFG (single forward pass -> ~2x faster); high guidance increases conditioning strength and can over-saturate colours in trained models.

### E08_pose_degraded — Blurred + jittered + partially occluded pose maps (noisy detector output).

* config changes: `{}` | options: `{"pose_blur": 3.0, "pose_jitter": 4, "pose_occlude": 0.25}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.99 s (-29.4% vs baseline), peak RSS 417 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Degraded pose maps (blur/jitter/occlusion) mimic detector failures: trained models show pose lag, distorted limbs or ignored poses.

### E09_frames_4 — Short clip: 4 frames.

* config changes: `{}` | options: `{"frames": 4}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.61 s (-56.8% vs baseline), peak RSS 403 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Frame-count sweep: cost is ~linear in F (temporal attention and VAE decode dominate); production v1/v2 use 24-frame windows with positional encodings sized for that window.

### E09b_frames_16 — Longer clip: 16 frames.

* config changes: `{}` | options: `{"frames": 16}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.84 s (+30.8% vs baseline), peak RSS 406 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Frame-count sweep: cost is ~linear in F (temporal attention and VAE decode dominate); production v1/v2 use 24-frame windows with positional encodings sized for that window.

### E10_res_64 — Lower resolution: 64x64.

* config changes: `{}` | options: `{"resolution": 64}`
* latent grid 8x8, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.70 s (-49.9% vs baseline), peak RSS 400 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Resolution sweep: latent grid scales with H/8; attention cost grows quadratically in the token count, so cost grows ~4x per doubling.

### E10b_res_128 — Higher resolution: 128x128.

* config changes: `{}` | options: `{"resolution": 128}`
* latent grid 16x16, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.55 s (+10.3% vs baseline), peak RSS 478 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Resolution sweep: latent grid scales with H/8; attention cost grows quadratically in the token count, so cost grows ~4x per doubling.

### E11_vae_fp16 — VAE runs in float16 on CPU (precision/quality check).

* config changes: `{"vae.dtype": "float16"}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 17.07 s (+1115.5% vs baseline), peak RSS 412 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: VAE precision: float16 changes decode speed and introduces rounding; with an untrained VAE this only measures numerics, not the colour/detail artefacts seen with trained VAEs.

### E12_slow_motion — Same 8 frames but half-speed motion (slow sequence).

* config changes: `{}` | options: `{"motion_speed": 0.5}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 1.03 s (-26.9% vs baseline), peak RSS 439 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Motion-speed sweep on a fixed frame budget: slower motion means smaller inter-frame deltas (metrics scale with true motion, which is why temporal metrics must be read together with the pose sequence).

### E13_pose_fusion_concat — Pose fused by channel concatenation instead of addition (SD-image-variation style).

* config changes: `{"runtime.pose_fusion": "concat", "unet.in_channels": 8}` | options: `{}`
* latent grid 12x12, 8 steps, CFG 3.5 (active=True), seed 1234
* runtime 0.95 s (-32.1% vs baseline), peak RSS 419 MB
* identity quality: n/a (random weights - needs trained model + real reference); pose accuracy: n/a (random weights - needs trained model + pose detector); temporal smoothness: structural proxy only (see flicker_index)
* interpretation: Pose fusion by channel concatenation (SD-image-variation style, 8-channel UNet input) instead of the production addition after conv_in; conv_in parameter count grows.
