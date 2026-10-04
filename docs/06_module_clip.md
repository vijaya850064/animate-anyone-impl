# 06 — CLIP image encoder: the semantic channel

**Question it answers:** *"What is the high-level visual meaning of the
reference?"*

---

## 1. Role, class, and freeze status

Animate Anyone uses a **frozen** CLIP *vision* encoder, not the text tower:

```python
from transformers import CLIPVisionModelWithProjection
image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path)  # train_stage_1.py:280-282
image_enc.requires_grad_(False)                                                     # train_stage_1.py:304

# inference (pipeline_pose2vid.py:318-325)
clip_image_embeds = self.image_encoder(clip_image.to(...)).image_embeds            # [1, D]
encoder_hidden_states = clip_image_embeds.unsqueeze(1)                              # [1, 1, D]
uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)              # CFG
```

| question | answer |
|---|---|
| which class? | `transformers.CLIPVisionModelWithProjection` — the vision tower **with** its `visual_projection` head, so the output is already projected |
| which checkpoint? | the SD **image-variations** image encoder (`stage1.yaml: image_encoder_path`), a CLIP ViT-H/14 variant whose projection width is **1024**; the UNet built from the same repo has `cross_attention_dim = 1024`, so no extra trainable projection is needed (`image_proj_model=None`) |
| frozen? | yes, in both stages (`requires_grad_(False)`) |
| text embeddings? | none in this architecture — the pipeline has `tokenizer`/`text_encoder` slots but they are unused; SD-1.5's 768-d text width is irrelevant here |
| output shape | `image_embeds`: `[B, D]` (prototype `[1, 64]`); cross-attention context: `[B, 1, D]` (prototype `[1, 1, 64]`) |
| measured range | prototype tiny CLIP: min −2.74, max +2.25 (`outputs/run_small/shape_log.json`) |

---

## 2. Image vs text embeddings, and why CLIP alone is not enough

| | CLIP **image** embedding (here) | CLIP **text** embedding (not used) |
|---|---|---|
| input | reference image (RGB, resized/cropped) | tokenised prompt |
| output | `[B, D]` projected visual vector | `[B, 77, D]` per-token sequence |
| trained for | aligning images with text (contrastive) | aligning text with images |
| in cross-attention | 1 token → a *global* bias per block | 77 tokens → fine-grained semantic control |
| why CLIP alone cannot preserve identity | a single 1024-d vector trained for *semantic similarity* has no incentive to encode exact colour/texture/garment detail; two different people of the same class can map to nearly the same vector |

Concretely: CLIP alone tells the model "a person in a red top"; it does not know
*this* person's stripe pattern, hairstyle or face shape. That gap is exactly what
ReferenceNet fills (`docs/02_module_reference_net.md` §5).

---

## 3. Preprocessing contract

Real CLIP preprocessing (what `CLIPImageProcessor` does, triggered by
`CLIPImageProcessor().preprocess(...)` in the reference pipeline):

| step | value | why it matters |
|---|---|---|
| colour format | convert to **RGB** | grayscale/RGBA silently shifts statistics |
| resize | shortest side → **224** (bicubic) | the ViT has a fixed 224×224 grid |
| centre crop | **224×224** | crops the sides of non-square references — a *semantic* crop, composition matters |
| rescale | `× 1/255` → `[0,1]` | |
| normalise | mean `(0.48145466, 0.4578275, 0.40821073)`, std `(0.26862954, 0.26130258, 0.27577711)` | CLIP's own statistics, **not** the VAE's `0.5/0.5` |
| batch dim | `[B, 3, 224, 224]` (B = 1) | |
| device / dtype | moved to model device; our implementation keeps CLIP in **fp32** even when the UNet runs fp16 (`pipeline.py`) | fp16 attention on CPU is fragile |
| projection layer | applied implicitly by `…WithProjection`; the pipeline adds **no** extra projection | |

How wrong preprocessing shows up:

* wrong normalisation → washed-out or over-saturated output, weak conditioning;
* no centre crop / stretching → the embedding describes a distorted character;
* RGB→BGR, or uint8 floats without rescale → near-random embeddings;
* feeding the VAE tensor (`[-1,1]`) instead of the PIL image → wrong statistics.

---

## 4. Implementation note (prototype): `src/aan/clip_encoder.py`

```python
class ClipImageEncoder(nn.Module):
    """Frozen CLIP vision tower with its built-in projection head."""
    def __init__(self, cfg: ClipConfig) -> None:
        if cfg.mode == "tiny":                       # no downloads: tiny random CLIPVisionConfig
            vision_config = CLIPVisionConfig(hidden_size=32, intermediate_size=64,
                                             num_hidden_layers=2, num_attention_heads=2,
                                             image_size=32, patch_size=16, projection_dim=64)
            self.model = CLIPVisionModelWithProjection(vision_config)
        else:                                        # e.g. openai/clip-vit-base-patch32
            self.model = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_id)
            self.processor = _try_load_processor(cfg.pretrained_id)
        self.model.eval(); self.model.requires_grad_(False)

    def preprocess(self, images) -> torch.Tensor:      # -> [B,3,S,S]
        ...                                           # HF processor, or the explicit CLIP recipe

    def forward(self, pixel_values) -> torch.Tensor:  # [B,3,S,S] -> [B, D]
        return self.model(pixel_values=pixel_values).image_embeds
```

The tiny fallback reimplements the *same contract* (bicubic shortest-side resize
→ centre crop → `/255` → CLIP mean/std) at `image_size = 32`, so the interface,
normalisation and projection semantics are identical even though the weights are
random. `config.py:validate_config` hard-fails when
`clip.projection_dim != unet.context_dim` — exactly the "cross-attention K/V
width mismatch" bug that check prevents.

Measured: 8 ms for the tiny encoder, 44,032 parameters.

---

## 5. Task-brief pseudocode vs implementation

```python
# brief
clip_inputs   = clip_processor(images=reference_image, return_tensors="pt")   # pixel_values
clip_features = clip_image_encoder(**clip_inputs).image_embeds                 # [1, D]
clip_features = clip_projection(clip_features)                                # extra projection
```

Implemented (and what production does):

```python
clip_pixels = self.clip.preprocess(ref_pil)     # [1,3,32,32] (tiny) / [1,3,224,224] (pretrained)
clip_emb    = self.clip(clip_pixels)            # [1, D]  (projection already applied inside CLIP)
context     = clip_emb.unsqueeze(1)             # [1,1,D] -> cross-attention K/V
# CFG: context = torch.cat([torch.zeros_like(context), context], dim=0)   # pipeline.py
```

Differences: production applies **no additional projection** (the model is
already `…WithProjection`, `image_proj_model=None`), and the unconditional
context under CFG is a **zero vector of the same shape**, not a learned "null"
embedding.

---

## 6. How these features are used (and not used)

* **used**: as K/V in every cross-attention layer, every block, every diffusion
  step (`mutual_self_attention.py:236-250` → `attn2`);
* **also used**: ReferenceNet receives the same embedding as its own
  cross-attention context (`train_stage_1.py:82`), so appearance extraction is
  CLIP-conditioned too;
* **never used**: never decoded into pixels, never concatenated to the latent,
  and never used to localise anything.

Key intuition for the team: CLIP answers *"what kind of scene/character is
this?"* globally; ReferenceNet answers *"which pixels exactly?"* spatially; the
Pose Guider answers *"where does each body part go?"*.