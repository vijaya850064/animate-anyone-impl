# 09 — Training and inference: flows, settings, and what they control

All production facts below are cited from `reference/Moore-AnimateAnyone/`
(`configs/train/stage1.yaml`, `configs/train/stage2.yaml`, `train_stage_1.py`,
`train_stage_2.py`, `src/dataset/*.py`, `scripts/pose2vid.py`).

---

## 1. Dataset structure

```
<dataset root>/
  <name>_meta.json          # list of records, produced by tools/extract_meta_info.py
    [{"video_path": "...mp4", "kps_path": "..._kps.mp4", "other": ...}, ...]
  videos/*.mp4              # driving videos
  pose_videos/*_kps.mp4     # DWPose renderings, frame-aligned with the videos
```

Preprocessing step (`tools/extract_dwpose_from_vid.py`) renders the pose maps
once; training then never runs a detector.

**Stage 1 sampling — a reference/target pair from the *same* video**
(`src/dataset/dance_image.py:73-121`):

```
video_length = len(reader)
margin       = min(sample_margin=30, video_length)
ref_idx      = random in [0, video_length-1]
tgt_idx      = random in [ref_idx+margin, video_length-1]   (or the mirrored interval)
ref_img      = frame[ref_idx]          -> VAE path + CLIPImageProcessor
tgt_img      = frame[tgt_idx]          -> VAE path (Normalize(0.5,0.5))
tgt_pose     = kps[tgt_idx]            -> cond path (ToTensor, NO normalisation)
```

The `sample_margin: 30` is the **identity hard-negative trick**: the target frame
is at least 30 frames away from the reference, so the model cannot copy the
reference frame and must transfer appearance while producing a *different* pose.

**Stage 2 sampling — a video clip** (`src/dataset/dance_video.py:79-134`):

```
clip_length = min(video_length, (n_sample_frames-1) * sample_rate + 1)      # 24 frames, stride 4 -> 93 frames
start_idx   = random in [0, video_length - clip_length]
batch_index = linspace(start_idx, start_idx+clip_length-1, n_sample_frames) # -> 24 frame indices
frames      = video[batch_index]        # -> [F,C,H,W]
poses       = kps[batch_index]          # -> [F,3,H,W]
ref_img     = random frame of the same video
```

Augmentation (identical RNG state for image/pose/ref — `torch.set_rng_state`,
so the pose map stays aligned with the frame):
`RandomResizedCrop((H,W), scale=(1.0,1.0), ratio=(0.9,1.0), BILINEAR)` +
`ToTensor()` + `Normalize([0.5],[0.5])` for pixels; the same crop **without**
normalisation for the pose map.

---

## 2. Stage 1 — spatial pretraining (single frames)

```
for each batch:
  ref_latents  = vae.encode(ref_img).mean * 0.18215        # VAE frozen, no grad
  clip_embeds  = image_encoder(clip_processor(ref_img))    # CLIP frozen, no grad
  pose_fea     = pose_guider(tgt_pose)                      # [B,320,1,h,w]
  if uncond_fwd (p = uncond_ratio = 0.1):                  # CFG training
      skip the ReferenceNet pass entirely (no reference attention)
  else:
      reference_unet(ref_latents, t=0, encoder_hidden_states=clip_embeds)   # WRITE bank
      reference_control_reader.update(reference_control_writer)
  model_pred   = denoising_unet(noisy_latents, t, pose_cond_fea=pose_fea,
                               encoder_hidden_states=clip_embeds)           # READ bank
  loss         = SNR-γ-weighted MSE(model_pred, target)
```

Config (`configs/train/stage1.yaml`): 768×768, batch 4, 30k steps, lr 1e-5
constant, fp16, xformers attention, gradient checkpointing **off**,
`sample_margin: 30`, noise: `scaled_linear` β 0.00085→0.012, `steps_offset: 1`,
zero-SNR + v-prediction, `snr_gamma: 5.0`, `noise_offset: 0.05`,
`uncond_ratio: 0.1`, `seed: 12580`, base model **sd-image-variations-diffusers**,
Pose Guider initialised from ControlNet-OpenPose (`pose_guider_pretrain: True`).
The denoising UNet is built with `use_motion_module: False, unet_use_temporal_attention: False`
(`train_stage_1.py:270-278`) — **no temporal modelling in stage 1**.

Loss (`train_stage_1.py:571-593`):

```python
if cfg.snr_gamma == 0:
    loss = F.mse_loss(model_pred.float(), target.float())
else:
    snr = compute_snr(train_noise_scheduler, timesteps)
    if prediction_type == "v_prediction":
        snr = snr + 1                                    # velocity objective
    mse_loss_weights = torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], 1).min(1)[0] / snr
    loss = (F.mse_loss(model_pred.float(), target.float(), reduction="none")
            .mean(dim=list(range(1, loss.ndim))) * mse_loss_weights).mean()
```

i.e. **Min-SNR-γ weighting** (down-weights very-low-SNR timesteps) on top of the
standard diffusion objective. No auxiliary reconstruction, perceptual or
identity loss is used in this reference implementation — worth knowing when
comparing with papers that add one.

---

## 3. Stage 2 — video pretraining (temporal)

```
for each batch (24-frame clip):
  same conditioning pipeline, plus:
  denoising_unet = UNet3DConditionModel.from_pretrained_2d(sd15, mm_sd_v15_v2.ckpt)
                    # 2D SD weights + AnimateDiff motion modules, zero-init proj_out
  loss = same SNR-γ-weighted MSE over all 24 frames
```

Config (`configs/train/stage2.yaml`): 512×512, batch 1, 10k steps, lr 1e-5,
`n_sample_frames: 24`, `sample_rate: 4`, gradient checkpointing **on**, 8-bit
Adam, noise `linear` β, zero-SNR v-prediction, `stage1_ckpt_step` initialises
from the stage-1 run.

Checkpointing: stage 1 saves per-module `.pth` state dicts (`reference_unet-N.pth`,
`denoising_unet-N.pth`, `pose_guider-N.pth`, `total_limit=3`) plus an accelerate
state; stage 2 additionally saves a **motion-module-only** checkpoint by filtering
keys containing `"motion_module"` — which is exactly what stage 2 of a *new*
video model needs to bootstrap from.

Memory strategies: fp16 autocast + xformers memory-efficient attention (both
stages) and gradient checkpointing (stage 2, where 24-frame activations
dominate). VAE and CLIP are never in the optimiser.

---

## 4. Inference flow

```
  reference image ──► VaeImageProcessor.preprocess(height, width) → [-1,1] ──► VAE encode
        │                                              (×0.18215)
        ├──► CLIPImageProcessor.preprocess → CLIPVisionModelWithProjection → [1,1,1024]
        └──► ReferenceNet(z_ref, t=0, context) ──► banks (fp16 copies)

  pose frames  ──► resize(W,H) /255 ──► [1,3,F,H,W] ──► PoseGuider → [1,320,F,h,w]

  latents = randn([1,4,F,h,w], generator=seed)
  context = cat([zeros, clip_context]) if guidance>1 else clip_context
  pose_fea = cat([pose_fea, pose_fea]) if guidance>1 else pose_fea

  scheduler.set_timesteps(steps)            # DDIM: β 0.00085→0.012 linear, steps_offset 1
  for i, t in enumerate(timesteps):
      if i == 0: reference_unet(...) ; reference_control_reader.update(writer)   # ONCE
      x_in = cat([latents]*2) if CFG else latents
      e    = denoising_unet(x_in, t, context, pose_cond_fea=pose_fea)
      if CFG: e_u, e_c = e.chunk(2); e = e_u + guidance * (e_c - e_u)
      latents = scheduler.step(e, t, latents).prev_sample

  video = VAE.decode(latents / 0.18215) frame-by-frame → [1,3,F,H,W] → (x/2+0.5) → mp4
```

Production defaults (`scripts/pose2vid.py`): `W=512, H=784, L=24 frames,
seed=42, cfg=3.5, steps=30` (DDIM). Note `H=784 = 98·8`, so the latent grid is
`98×64` and **no padding is needed** — a common source of off-by-8 artefacts when
the height is not divisible by 8.

---

## 5. Inference settings and what they control

| setting | identity | pose | smoothness | sharpness | flicker | time | memory |
|---|---|---|---|---|---|---|---|
| **more steps** (30 → 50) | ↑ slightly | ↑ slightly | ↑ | ↑ | ↓ | ×1.7 | ~same |
| few steps (10) | ↓ (details drop) | ↓ (limbs soften) | ~ | ↓ | ↑ | ×0.33 | ~same |
| **guidance 1.0 (no CFG)** | ↓ (generic faces) | ↓ | ~ | ↓ | ↑ | **×0.5** (one pass) | ~same |
| guidance 3.5 (default) | good | good | good | good | good | baseline | baseline |
| guidance 7.5+ | ↑ "hardened", can over-saturate | ↑ over-committed | ↓ stiff | ↓ clipped | ↑ | baseline | baseline |
| **more frames** | ~ | ~ | ↑ (longer context) | ~ | ↓ | **linear in F** | linear in F |
| **higher resolution** | ↑ (more detail tokens) | ↑ | ~ | ↑ | ~ | **~4× per doubling** (O(N²) attention) | ~4× |
| **fp16 VAE** | ~ | ~ | ~ | ↓ tiny rounding | ~ | ↓ (faster) | ↓ |
| VAE slicing / frame-wise decode | identical | identical | identical | identical | identical | slightly slower | ↓↓ (the reason it exists) |
| seed change | *which* identity variant, not its fidelity | same | same | same | same | same | same |
| temporal PE window (24 vs 32) | ~ | ~ | ↑ with more frames | ~ | ↓ | ~ | ~ |

The structural predictions (steps → linear time, CFG off → ~half UNet work,
resolution → steepest) are confirmed by the measured ablation runs
(`outputs/ablations/ablation_report.md`); the quality columns there are honestly
marked `n/a` because the weights are untrained.

---

## 6. Post-processing and export

* `video = (video/2 + 0.5).clamp(0,1)` then `× 255` → `uint8`;
* `save_videos_grid(..., n_rows=3)` concatenates **reference row / pose row /
  generated row** for human review — our harness mirrors this with
  `outputs/*/contact_sheet.png`;
* `imageio`/ffmpeg writes H.264 mp4; the prototype sets `macro_block_size=None`
  so frame sizes like 96×96 are not silently rescaled;
* **labelling**: generated clips are synthetic — our runs write a `disclaimer`
  string into `run_log.json`. Keep that practice for anything leaving the team.