"""End-to-end educational pipeline: reference image + pose sequence -> video.

This assembles every component exactly in the production order
(``pipeline_pose2vid.py``):

1. reference image → VAE encoder (× 0.18215) → reference latent
2. reference latent → ReferenceNet → banked features (stored once)
3. reference image → CLIP image encoder → ``[1, 1, D]`` cross-attention context
4. pose frames → resize/÷255 → Pose Guider → per-frame pose features
5. random latent ``[1, 4, F, h, w]`` → DDIM loop:
       UNet(noisy latent, t, pose, reference bank, CLIP context) → predicted noise
       (+ classifier-free guidance) → scheduler step
6. final latent → VAE decoder (frame-wise) → RGB frames → mp4

Deviations from production that are *deliberate* and documented
---------------------------------------------------------------
* CFG: production applies reference attention only in the conditional half via
  a ``uc_mask`` inside the hacked attention.  We pass the bank to both halves
  (simpler, readable), which slightly changes the unconditional prediction; see
  docs/04_module_denoising_unet.md.
* CFG context: production repeats ``torch.zeros_like(embeds)`` as the
  unconditional context; we do the same (``clip.use_null_embedding``).
* Everything else (pose addition inside the UNet, DDIM betas, scaling factor,
  frame-wise decode) matches the reference mechanics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DDIMScheduler
from einops import rearrange

from .clip_encoder import ClipImageEncoder
from .config import PrototypeConfig, config_to_dict, resolve_out_channels, validate_config
from .denoising_unet import TinyVideoUNet
from .pose_guider import PoseGuider, preprocess_pose_frames
from .reference_net import SimpleReferenceNet
from .shape_log import ShapeLogger, count_parameters, timer_ms
from .utils import resolve_device, resolve_dtype, seed_everything
from .vae import VaeWrapper

__all__ = ["AnimateAnyoneLite"]


class AnimateAnyoneLite(nn.Module):
    """CPU-friendly educational version of the Animate Anyone inference graph."""

    def __init__(
        self,
        cfg: PrototypeConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        logger: Optional[ShapeLogger] = None,
    ) -> None:
        super().__init__()
        validate_config(cfg)  # raises on structural mismatches
        self.cfg = cfg
        self.device = device or resolve_device(cfg.runtime.device)
        self.dtype = dtype or resolve_dtype(cfg.runtime.dtype, self.device)
        self.log = logger if logger is not None else ShapeLogger(cfg.runtime.log_shapes)

        # The VAE may be run in a different precision than the UNet (experiment E11):
        # vae.dtype: null -> use runtime.dtype, otherwise explicit.
        vae_dtype = resolve_dtype(cfg.vae.dtype, self.device) if cfg.vae.dtype else self.dtype
        self.vae = VaeWrapper(cfg.vae).to(device=self.device, dtype=vae_dtype)
        self.clip = ClipImageEncoder(cfg.clip).to(device=self.device)  # CLIP stays fp32
        self.reference_net = SimpleReferenceNet(
            in_channels=cfg.vae.latent_channels,
            base_channels=tuple(cfg.reference_net.base_channels),
            heads=cfg.reference_net.heads,
            store_all_blocks=cfg.reference_net.store_all_blocks,
        ).to(device=self.device, dtype=self.dtype)
        self.pose_guider = PoseGuider(
            out_channels=resolve_out_channels(cfg),
            block_out_channels=tuple(cfg.pose_guider.block_out_channels),
            zero_init_out=cfg.pose_guider.zero_init_out,
        ).to(device=self.device, dtype=self.dtype)
        self.unet = TinyVideoUNet(cfg).to(device=self.device, dtype=self.dtype)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            beta_schedule=cfg.diffusion.beta_schedule,
            steps_offset=cfg.diffusion.steps_offset,
            clip_sample=False,
            prediction_type=cfg.diffusion.prediction_type,
        )

        for module in (self.vae, self.clip, self.reference_net, self.pose_guider, self.unet):
            module.eval()
            module.requires_grad_(False)   # prototype is inference-only
        self.log.log_module("vae", self.vae)
        self.log.log_module("clip", self.clip)
        self.log.log_module("reference_net", self.reference_net)
        self.log.log_module("pose_guider", self.pose_guider)
        self.log.log_module("unet", self.unet)

    # ------------------------------------------------------------------ info
    def parameter_report(self) -> Dict[str, int]:
        return {
            "vae": count_parameters(self.vae),
            "clip": count_parameters(self.clip),
            "reference_net": count_parameters(self.reference_net),
            "pose_guider": count_parameters(self.pose_guider),
            "denoising_unet": count_parameters(self.unet),
        }

    @property
    def latent_hw(self) -> tuple[int, int]:
        f = self.vae.down_factor
        return self.cfg.input.ref_height // f, self.cfg.input.ref_width // f

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        reference_image: Image.Image,
        pose_images: Sequence[Image.Image],
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run the full pipeline for one clip.

        Returns a dict with: ``video01`` ``[F,3,H,W]`` float CPU, ``ref01``
        ``[3,H,W]``, ``latents``, ``timings`` (ms), ``meta`` (run settings).
        """
        cfg = self.cfg
        log = self.log
        F = cfg.input.frames
        H, W = cfg.input.ref_height, cfg.input.ref_width
        if len(pose_images) < F:
            raise ValueError(f"need >= {F} pose frames, got {len(pose_images)}")
        steps = int(steps or cfg.diffusion.num_inference_steps)
        guidance = float(cfg.diffusion.guidance_scale if guidance_scale is None else guidance_scale)
        seed = int(cfg.runtime.seed if seed is None else seed)
        seed_everything(seed)
        use_cfg = bool(cfg.diffusion.use_cfg) and guidance > 1.0
        timings: Dict[str, float] = {}

        # 1. reference image -------------------------------------------------
        with timer_ms("preprocess_reference", timings):
            ref_pil = reference_image.convert("RGB").resize((W, H), Image.BILINEAR)
            ref_arr = np.asarray(ref_pil, dtype=np.float32) / 255.0
            ref_tensor = torch.from_numpy(ref_arr.transpose(2, 0, 1)).unsqueeze(0)
            ref_tensor = (ref_tensor * 2 - 1).to(device=self.device, dtype=self.dtype)
        log.record("reference_image", ref_tensor, "[B,3,H,W] in [-1,1]")

        with timer_ms("vae_encode_reference", timings):
            # cast back to the UNet dtype: the VAE may run in another precision (E11)
            ref_latent = self.vae.encode(ref_tensor).to(self.dtype)
        log.record("reference_latent", ref_latent, "[B,4,h,w] = VAE(ref)*0.18215")

        # 2. CLIP semantic context -------------------------------------------
        if cfg.runtime.use_clip:
            with timer_ms("clip_encode_reference", timings):
                clip_pixels = self.clip.preprocess(ref_pil).to(self.device)
                clip_emb = self.clip(clip_pixels)               # [1, D]
                context = clip_emb.unsqueeze(1).to(self.dtype)  # [1, 1, D]
            log.record("clip_image_embedding", clip_emb, "[B,D] projected CLIP image embedding")
            log.record("clip_context", context, "[B,1,D] cross-attention K/V")
        else:
            context = None
            timings["clip_encode_reference"] = 0.0

        # 3. ReferenceNet (feature bank, stored once) -------------------------
        if cfg.runtime.use_reference:
            with timer_ms("reference_net", timings):
                bank = self.reference_net(ref_latent)
            for i, b in enumerate(bank):
                log.record(f"reference_bank_{i}", b, f"bank tensor #{i} [B,N_i,C_i] from ReferenceNet")
        else:
            bank = None
            timings["reference_net"] = 0.0

        # 4. Pose guider -------------------------------------------------------
        pose_maps = preprocess_pose_frames(pose_images[:F], W, H)  # [1,3,F,H,W] in [0,1]
        log.record("pose_maps", pose_maps, "[1,3,F,H,W] in [0,1], no normalization")
        if cfg.runtime.use_pose:
            with timer_ms("pose_guider", timings):
                pose_fea = self.pose_guider(pose_maps.to(device=self.device, dtype=self.dtype))
            log.record("pose_features", pose_fea, "[1,C_inner,F,h,w] added after conv_in")
        else:
            pose_fea = None
            timings["pose_guider"] = 0.0

        # 5. latent initialisation --------------------------------------------
        h, w = self.latent_hw
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(1, self.vae.latent_channels, F, h, w, generator=generator)
        latents = latents.to(device=self.device, dtype=self.dtype)
        log.record("noisy_video_latents", latents, "[B,4,F,h,w] initial pure noise x_T")

        # 6. denoising loop -----------------------------------------------------
        self.scheduler.set_timesteps(steps, device=self.device)
        step_times: List[float] = []
        for i, ts in enumerate(self.scheduler.timesteps):
            if cfg.runtime.pose_fusion == "concat" and pose_fea is not None:
                model_in = torch.cat([latents, pose_fea], dim=1)
            else:
                model_in = latents
            if use_cfg:
                model_in = torch.cat([model_in, model_in], dim=0)
                ctx = torch.cat([torch.zeros_like(context), context], dim=0) if context is not None else None
                pose_in = torch.cat([pose_fea, pose_fea], dim=0) if pose_fea is not None else None
            else:
                ctx, pose_in = context, pose_fea
            model_in = self.scheduler.scale_model_input(model_in, ts)

            t0 = time.perf_counter()
            noise_pred = self.unet(
                model_in,
                ts,
                context=ctx,
                pose_features=pose_in,
                reference_bank=bank,
                ref_strength=cfg.runtime.reference_attention_strength,
                use_temporal=cfg.runtime.use_temporal,
            )
            if use_cfg:
                uncond, cond = noise_pred.chunk(2, dim=0)
                noise_pred = uncond + guidance * (cond - uncond)
            latents = self.scheduler.step(noise_pred, ts, latents, eta=cfg.diffusion.eta).prev_sample
            step_times.append((time.perf_counter() - t0) * 1000.0)
            if i == 0:
                log.record("unet_input_first_step", model_in, "[B,4,F,h,w] (CFG batch doubled)")
                log.record("predicted_noise_first_step", noise_pred, "[B,4,F,h,w]")

        timings["denoise_total"] = float(np.sum(step_times))
        timings["denoise_per_step_mean"] = float(np.mean(step_times))
        log.record("denoised_video_latents", latents, "[B,4,F,h,w] final x_0 latent")

        # 7. VAE decode ---------------------------------------------------------
        with timer_ms("vae_decode_all_frames", timings):
            video = self.vae.decode_video_framewise(latents)   # [1,3,F,H,W] in [-1,1]
        log.record("decoded_video", video, "[B,3,F,H,W] RGB in [-1,1]")
        # video is [B, C, F, H, W] (channels-first, as produced by the VAE);
        # the prototype API works with [F, C, H, W] per-frame layout.
        frames01 = ((video[0].float() / 2 + 0.5).clamp(0, 1)).permute(1, 0, 2, 3).cpu()
        ref01 = ((ref_tensor[0].float() / 2 + 0.5).clamp(0, 1)).cpu()

        timings["total"] = float(np.sum([v for k, v in timings.items() if k != "total"]))
        meta = {
            "frames": F, "height": H, "width": W, "latent_hw": [h, w],
            "steps": steps, "guidance_scale": guidance, "use_cfg": use_cfg,
            "seed": seed, "device": str(self.device), "dtype": str(self.dtype),
            "pose_fusion": cfg.runtime.pose_fusion,
            "use_reference": cfg.runtime.use_reference,
            "use_clip": cfg.runtime.use_clip,
            "use_pose": cfg.runtime.use_pose,
            "use_temporal": cfg.runtime.use_temporal,
            "reference_attention_strength": cfg.runtime.reference_attention_strength,
        }
        return {
            "video01": frames01,
            "ref01": ref01,
            "latents": latents.detach().cpu(),
            "timings": timings,
            "meta": meta,
            "config": config_to_dict(cfg),
        }

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def save_video(frames01: torch.Tensor, path: str | Path, fps: int = 8) -> Path:
        """Save ``[F,3,H,W]`` frames in [0,1] as an H.264 mp4 via imageio."""
        import imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = (frames01.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()  # [F,H,W,3]
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
        try:
            for frame in arr:
                writer.append_data(frame)
        finally:
            writer.close()
        return path
