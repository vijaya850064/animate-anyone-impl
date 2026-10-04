# 13 — Glossary of diffusion and attention terminology

Terms are defined in the context of *this* project (Animate Anyone core model).

## Diffusion

**Latent diffusion (LDM)** — running the diffusion process in the compressed
space produced by an autoencoder instead of in pixels; makes video synthesis
tractable (Rombach et al. 2020; the foundation of Stable Diffusion and of
Animate Anyone).

**Forward process / noising** — the fixed Markov chain that adds Gaussian noise:
`x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`.

**Reverse process / denoising** — the learned chain the model approximates; our
sampler approximates it with 20–30 DDIM steps.

**ε-prediction vs v-prediction** — which target the network regresses: the noise
`ε` or the velocity `v = √ᾱ_t·ε − √(1-ᾱ_t)·x_0`. The reproduction's inference-v2
config uses v-prediction with zero-SNR.

**β schedule** — the noise-level schedule (`beta_start: 0.00085 → beta_end: 0.012`
here, `linear` or `scaled_linear`); `steps_offset: 1` shifts discretisation.

**ᾱ_t (alpha-bar)** — cumulative product of `(1-β)`; controls how much signal
survives. At `t≈999`, `ᾱ ≈ 4e−5`, which is why ε-prediction errors are amplified
into huge latents when the model is untrained.

**DDIM** — a deterministic sampler for the reverse process; allows `eta=0` and
few-step sampling. Used in production and here.

**Classifier-free guidance (CFG)** — running the model twice per step (with and
without conditioning) and extrapolating: `ε = ε_u + g·(ε_c − ε_u)`. Here the
"unconditional" branch means *no reference image, zero CLIP embedding*. Costs 2×.

**Zero-SNR / v-prediction re-scaling** — rescaling β so the terminal SNR is 0
(fully noised), improving v-prediction training and low-step sampling
(`rescale_betas_zero_snr: True`, `timestep_spacing: trailing`).

**Min-SNR-γ loss weighting** — down-weighting the loss at very low SNR during
training (`snr_gamma: 5.0` here); stabilises diffusion fine-tuning.

**Latent scaling factor (0.18215)** — the constant that maps SD latents to
roughly unit scale so the `N(0,I)` noise prior is well matched; multiplied on
encode, divided before decode.

## Architectures

**UNet** — encoder–decoder with skip connections that predicts the diffusion
target from a noisy latent; the only learned module inside the sampling loop.

**Skip connection** — channel concatenation of an encoder feature map into the
matching decoder block; carries spatial detail of the *generated* video.

**ResBlock** — residual unit with GroupNorm/SiLU/conv and a timestep embedding
added to the features.

**Inflated conv (2D→3D)** — applying a 2D convolution frame-wise to a
`[B,C,F,H,W]` tensor so pretrained 2D weights load unchanged.

**ReferenceNet** — a copy of the denoising UNet **without temporal layers**, run
on the reference image at timestep 0; its per-block hidden states are stored
("banked") and injected into the denoising UNet as extra attention keys/values.

**Reference attention / mutual self-attention** — attention whose queries are the
current video tokens and whose keys/values are the concatenation of current and
reference tokens.

**Pose Guider** — a small stride-2 convolutional encoder mapping pose maps to
features at latent resolution; the features are **added** after the denoising
UNet's `conv_in`.

**Motion module / temporal attention** — an AnimateDiff-style transformer block
that attends across frames at each spatial position (with sinusoidal temporal
positional encoding), inserted after spatial blocks; zero-initialised so it
starts as an identity.

**CLIP image encoder** — frozen ViT producing a projected global image embedding
used as cross-attention context.

**Frozen / trainable** — frozen modules keep pretrained weights and no gradients
(VAE, CLIP); trainable modules are optimised (Pose Guider, ReferenceNet,
denoising UNet, motion modules).

**Two-stage training** — stage 1 trains the spatial model on single frames; stage 2
adds temporal modules and trains on video clips.

## Attention

**Query / Key / Value (Q/K/V)** — the three learned projections of attention;
"query" asks, "key" advertises, "value" carries content.

**Self-attention** — Q, K and V all come from the same tensor.

**Cross-attention** — Q from the UNet, K/V from another tensor (here: CLIP, or
the reference bank in the "reference attention" formulation).

**Multi-head attention** — several independent attention maps over sub-spaces
(`heads: 4` here, 8 in production) concatenated back.

**Scaled dot-product** — `softmax(QKᵀ/√d_k)V`; the `1/√d_k` keeps the softmax
gradients well-scaled.

**Softmax weights as a distribution** — since weights sum to 1 over keys, the
output is a convex combination of values: that is what makes "which detail gets
transferred" interpretable.

**Residual connection** — adding the block input back to its output, so attention
refines rather than replaces.

**LayerNorm (pre-norm)** — normalising features before Q/K/V; the normalised
tensor is exactly what ReferenceNet stores in its bank.

**Attention mask** — a bias that hides keys (e.g. additive `-10000`); used here
only for CFG bookkeeping.

**Attention slicing** — chunking attention calls to save memory at the cost of
recomputation.

**xformers / memory-efficient attention** — fused kernels that avoid
materialising the `N×N` weight matrix.

**Token / sequence length `N`** — for spatial attention, `N = h·w`; for temporal
attention, `N = F`.

## Video / data

**Pose map (OpenPose / DWPose rendering)** — an RGB image drawing the detected
skeleton; the model's actual motion input (not raw keypoints).

**Keypoints** — 2D/3D joint locations (COCO-18 layout); converted to a drawing
before reaching the network.

**Occlusion / missing keypoints** — detector failure where a body part is not
drawn; the model sees an unconstrained region.

**Frame stride / sample rate** — temporal subsampling of the driving video
(`sample_rate: 4` in stage-2 training).

**Window / PE max length** — the temporal positional encoding limits how many
frames the temporal attention can address (24 in v1, 32 in v2).

**Flicker** — frame-to-frame appearance instability; mitigated by temporal
attention.

**Fidelity vs diversity trade-off** — in diffusion sampling, more guidance/steps
buy fidelity (detail, identity) at the cost of diversity and speed.