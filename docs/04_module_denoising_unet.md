# 04 — Denoising UNet: the main generative model

**Question it answers:** *"How can noisy latent frames be transformed into a
coherent animation?"*

---

## 1. Role in latent diffusion

The UNet is the only *learned* component inside the sampling loop. Everything
else is a fixed transform: the VAE defines pixel↔latent space, CLIP the semantic
context, the scheduler the noise schedule; the Pose Guider and ReferenceNet are
learned but run **outside** the loop. The UNet is where all conditioning is
fused and where the image is actually synthesised.

**Forward (training) process** — a fixed Markov chain that destroys structure:

```
x_0 ~ q(x_0|x)                                  (VAE-encoded real video latents)
x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε ,  ε ~ N(0, I),   t ~ U{0,…,999}
loss = E_{t,ε} ‖ v̂_θ(x_t, t, c) − v ‖²          (or ‖ ε̂ − ε ‖² for epsilon-prediction)
```

The reproduction trains **v-prediction with zero-SNR**
(`train_stage_1.py:253-258`, `inference_v2.yaml`): `rescale_betas_zero_snr=True`,
`timestep_spacing="trailing"`, `prediction_type="v_prediction"`, with an SNR-γ
(γ=5) weighted MSE (`train_stage_1.py:571-593`). Stage-1 additionally uses
`scaled_linear` betas; inference uses `linear` with the same endpoints
(β: 0.00085 → 0.012, `steps_offset: 1`).

**Reverse (sampling) process** — the loop our pipeline runs:

```
x_T ~ N(0, I)                                            # [B,4,F,h,w]
for t = T … 1:
    ε̂ = UNet_θ(x_t, t, pose, ref bank, CLIP)              # one or two CFG passes
    x_{t-1} = Scheduler.step(ε̂, t, x_t)                   # DDIM in the reproduction
video = VAE.decode(x_0)                                   # frame by frame
```

How does "random noise become a structured video"? Each step the UNet receives
features that already contain *pose structure* (added after `conv_in`),
*appearance* (reference-attention K/V) and *temporal context*; the ε prediction
is only correct if those conditions explain the image, so repeated refinement
progressively locks skeletons, colours and frame-to-frame consistency into `x_t`.

---

## 2. Timestep conditioning

```python
t_emb = self.time_proj(timesteps)          # sinusoidal, [B, block_out_channels[0]]
emb   = self.time_embedding(t_emb)         # 2-layer MLP -> [B, time_embed_dim]
# inside every ResBlock (SD "default" norm):
h = conv1(silu(norm1(x)))
h = h + time_proj(silu(emb))[:, :, None, None, None]
```

Prototype: `layers.TimeEmbedding` (sinusoidal + Linear-SiLU-Linear) injected in
`layers.ResBlock3D.forward`. The embedding is the *only* place `t` enters —
everything else is spatial or temporal.

---

## 3. From a 2D image UNet to a video UNet

| aspect | 2D image UNet (SD 1.5) | video denoising UNet (Animate Anyone) |
|---|---|---|
| tensor | `[B,4,h,w]` | `[B,4,F,h,w]` |
| convs | 2D | `InflatedConv3d`: **2D conv applied per frame** (`resnet.py:9-17`) — weights keep the 2D shape so pretrained SD weights load unchanged |
| normalisation | GroupNorm over `[B,C,H,W]` | `InflatedGroupNorm` per frame (`resnet.py:20-28`) — statistics are per-frame, not per-clip |
| up/down sampling | 2D nearest + conv | `Upsample3D`/`Downsample3D` resample only H,W, never F (`resnet.py:31-113`) |
| self-attention | over `h·w` tokens | over `h·w` tokens **per frame** (frames folded into the batch) |
| extra axis | — | **temporal attention** over `F` at each spatial position |
| output | `[B,4,h,w]` | `[B,4,F,h,w]` |

Temporal module placement (`configs/inference/inference_v1.yaml`, `motion_module.py`):
`motion_module_resolutions: [1, 2, 4, 8]` → a `VanillaTemporalModule` after the
spatial blocks of **every** encoder and decoder block (v1: mid-block off,
v2: `motion_module_mid_block: true`), 8 heads, `num_transformer_block: 1`,
sinusoidal temporal positional encoding (max_len 24 in v1, 32 in v2), and a
**zero-initialised** output projection
(`zero_module(self.temporal_transformer.proj_out)`) so the motion branch starts
as an identity on top of a pretrained image model. Motion weights come from
AnimateDiff (`mm_sd_v15_v2.ckpt`, `stage2.yaml:mm_path`).

---

## 4. Conditioning entry points (all inside the UNet)

| condition | entry point | code |
|---|---|---|
| pose | `sample = conv_in(sample); sample += pose_cond_fea` | `unet_3d.py:503-505` |
| reference appearance | spatial-attention K/V augmentation per block | `mutual_self_attention.py:149-159` |
| CLIP semantics | cross-attention `attn2(..., encoder_hidden_states=embeds.unsqueeze(1))` | `mutual_self_attention.py:236-250`, `pipeline_pose2vid.py:321-324` |
| timestep | additive embedding in every ResBlock | `unet_3d.py:482-488` |
| frame continuity | temporal attention + temporal positional encoding | `motion_module.py:VersatileAttention.forward` |

**Skip connections** are the plain UNet kind — `torch.cat([x, skip], dim=1)`
before each decoder ResBlock (`[C2+C1 → C1]`, `[C1+C0 → C0]`). They carry
*noisy-video* detail and are unrelated to ReferenceNet.

---

## 5. Classifier-free guidance (CFG)

Production (`pipeline_pose2vid.py:309, 327-330, 412-432`):

```python
do_classifier_free_guidance = guidance_scale > 1.0
uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)
latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
noise_pred = self.denoising_unet(latent_model_input, t, encoder_hidden_states=…, pose_cond_fea=pose_fea)
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

Details that matter for correctness:

* the **unconditional** branch uses a **zero CLIP embedding** (there is no text
  prompt in this architecture), so "CFG" means "conditioned vs not conditioned on
  the reference image + CLIP";
* pose features are duplicated for both halves (`torch.cat([pose_fea] * 2)`);
* the reference bank is applied **only to the conditional half**, via `uc_mask`
  inside the patched attention (`mutual_self_attention.py:168-181`);
* training must match inference — the reference repo trains with
  `uncond_ratio: 0.1` and skips the ReferenceNet pass for those samples
  (`Net.forward(..., uncond_fwd=True)`, `train_stage_1.py:74-85`).

Prototype: identical structure, with the documented simplification that the bank
is broadcast to both halves (`attention.py:137-141`).

---

## 6. The three attention mechanisms inside one block

```
AttentionStack.forward(x [B,C,F,h,w], bank, clip_context, ref_strength, use_temporal)
 ├─ spatial(reference) attention : Q = frame tokens
 │                                K/V = [frame tokens ; ReferenceNet bank]   (attn1)
 ├─ cross-attention              : Q = frame tokens, K/V = CLIP [1,1,D]      (attn2)
 └─ temporal attention           : sequence = F at each (b,h,w)               (motion module)
```

Conceptual pseudocode from the task brief, mapped onto the real code:

```python
# CONCEPTUAL (task brief)                    # THIS REPOSITORY (src/aan/denoising_unet.py)
x = noisy_latents + pose_features            x = self.conv_in(sample)
for block in unet.blocks:                    if pose_cond_fea is not None: x = x + pose_cond_fea
    x = block.spatial_attention(             x = self.down0_res(x, temb)
              x, reference_features)         x = self.down0_attn(x, bank_at(0), context,
    x = block.cross_attention(                            ref_strength, use_temporal)
              x, encoder_hidden_states)     ... down1 / mid / up0 / up1 ...
    x = block.temporal_attention(x)          return self.conv_out(self.act(self.norm_out(x)))
predicted_noise = unet.output_layer(x)
```

Differences worth stating explicitly:

1. the brief's `x = noisy_latents + pose_features` is **not** what the production
   code does (it adds *after* `conv_in`, i.e. in inner-feature space) — we follow
   the production behaviour and keep the literal version as `pose_fusion: concat`;
2. the brief's plain `spatial_attention(x, reference_features)` hides that the
   reference features are **appended to the K/V sequence**, not summed into the
   query — our implementation shows the concatenation explicitly;
3. real UNet blocks also contain ResBlocks, skip connections, norms and dropout,
   which the conceptual snippet omits.

---

## 7. Frozen vs trainable parameters (production)

| module | stage 1 (image) | stage 2 (video) | evidence |
|---|---|---|---|
| VAE | frozen | frozen | `vae.requires_grad_(False)` — `train_stage_1.py:303` |
| CLIP image encoder | frozen | frozen | `image_enc.requires_grad_(False)` — `train_stage_1.py:304` |
| Pose Guider | **trained** (optionally init from ControlNet-OpenPose) | trained | `:315`, `:284-296` |
| Denoising UNet | trained, **without** motion modules | trained **with** motion modules | `:307`, `:270-278` (`use_motion_module: False`) |
| ReferenceNet | trained **except** `up_blocks.3` | trained | `:309-313` |
| Motion modules | absent | trained, init from AnimateDiff | `stage2.yaml: mm_path` |
| text encoder / tokenizer | unused (`image_proj_model=None`, no text branch) | unused | `pipeline_pose2vid.py:__init__` |

Precision/memory in training: `mixed_precision: 'fp16'`, xformers memory-efficient
attention enabled, `gradient_checkpointing: True` for stage 2 (24 frames at
512×512), 8-bit Adam, `max_grad_norm: 1.0`.

In **this** repository all modules are inference-only: `requires_grad_(False)` for
every module in `pipeline.py`, documented in `docs/16_final_report.md`.