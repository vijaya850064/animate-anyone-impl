"""Attention mechanisms used by Animate Anyone, implemented explicitly.

This module is intentionally *readable* rather than fast: every mechanism is a
hand-written scaled dot-product block with shape assertions, so the formula

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

can be traced in the code.  Production code uses fused kernels (xformers /
``torch.nn.functional.scaled_dot_product_attention``); the mathematics is the
same and the mapping is documented per class.

The four attention roles in the architecture (see docs/07_attention_deep_dive.md):

=====================  ==================  ====================  ===============================
Mechanism              Query source        Key/Value source      Purpose
=====================  ==================  ====================  ===============================
Spatial self-attn      frame tokens        same frame tokens     intra-frame structure
Reference attention    frame tokens        concat(frame, ref)    appearance transfer (ReferenceNet)
Cross-attention        frame tokens        CLIP embeddings       semantic conditioning
Temporal attention     frame tokens        same location, other  motion continuity / flicker
                                           frames
=====================  ==================  ====================  ===============================
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .layers import sinusoidal_timestep_embedding, zero_module

__all__ = [
    "scaled_dot_product_attention",
    "ReferenceSpatialAttention",
    "CrossAttention",
    "TemporalAttention",
]


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Manual multi-head attention core.

    Args:
        q, k, v: ``[..., heads, seq, head_dim]`` tensors (any leading batch dims).
    Returns:
        Same shape as ``q``.
    """
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5
    attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)  # weights over keys
    return attn @ v



class ReferenceSpatialAttention(nn.Module):
    """Spatial attention whose keys/values are augmented with ReferenceNet tokens.

    Production equivalent (``mutual_self_attention.py``, read mode):

    .. code-block:: python

        bank_fea = [rearrange(d.unsqueeze(1).repeat(1, video_length, 1, 1),
                              "b t l c -> (b t) l c") for d in self.bank]
        modify_norm_hidden_states = torch.cat([norm_hidden_states] + bank_fea, dim=1)
        hidden_states = self.attn1(norm_hidden_states,
                                   encoder_hidden_states=modify_norm_hidden_states)

    i.e. **Q = current frame tokens** and **K/V = [current frame tokens; reference
    tokens]** (the reference feature map is repeated once per frame, so every
    frame can attend to the reference).  The paper describes the same operation
    as concatenating the denoising features with *t* copies of the static
    reference feature map and keeping only the denoising half of the output.

    ``strength`` is an ablation knob: intermediate values linearly interpolate
    between the reference-augmented output (1.0) and a plain self-attention
    output (0.0), which is how experiment E05 "weaker reference conditioning"
    is implemented.  The production model always uses strength = 1.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)          # production norm1 (LayerNorm)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B*F, N, C] -> [B*F, heads, N, head_dim]
        bf, n, c = x.shape
        return x.view(bf, n, self.heads, self.head_dim).transpose(1, 2)

    def _plain_self(self, normed: torch.Tensor) -> torch.Tensor:
        # normed: [B*F, N, C] -> [B*F, N, C]
        q, k, v = self._heads(self.to_q(normed)), self._heads(self.to_k(normed)), self._heads(self.to_v(normed))
        out = scaled_dot_product_attention(q, k, v)
        bf, h, n, dh = out.shape
        return out.transpose(1, 2).reshape(bf, n, h * dh)

    def _with_reference(self, normed: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # normed: [B*F, N, C]; ref: [B*F, Nr, C]
        kv = torch.cat([normed, ref], dim=1)                       # K/V token count = N + Nr
        q = self._heads(self.to_q(normed))                         # Q: N tokens (denoising half)
        k = self._heads(self.to_k(kv))
        v = self._heads(self.to_v(kv))
        out = scaled_dot_product_attention(q, k, v)                # [B*F, heads, N, dh]
        bf, h, n, dh = out.shape
        return out.transpose(1, 2).reshape(bf, n, h * dh)

    def forward(
        self,
        x: torch.Tensor,
        ref_tokens: Optional[torch.Tensor] = None,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, F, N, C]`` tokens of the *current* (noisy video) features,
               N = h*w spatial positions of one frame at this UNet level.
            ref_tokens: ``[B, Nr, C]`` banked ReferenceNet features (same channel
               width as x — guaranteed by config validation), or None.
            strength: 0..1 blend between plain self-attention and
               reference-augmented attention.
        Returns:
            ``[B, F, N, C]`` tensor with a residual connection applied.
        """
        assert x.dim() == 4, f"ReferenceSpatialAttention expects [B,F,N,C]; got {tuple(x.shape)}"
        b, f, n, c = x.shape
        if ref_tokens is not None:
            if ref_tokens.shape[0] == 1 and b > 1:
                # Classifier-free guidance doubles the UNet batch (uncond + cond) while a
                # single reference image produces one bank entry: broadcast it. The production
                # code gets the same effect via its `uc_mask` bookkeeping.
                ref_tokens = ref_tokens.expand(b, -1, -1)
            assert ref_tokens.shape[0] == b, (
                f"reference batch {ref_tokens.shape[0]} != frame batch {b}"
            )
            assert ref_tokens.shape[2] == c, (
                f"reference channel width {ref_tokens.shape[2]} != frame width {c}; "
                "ReferenceNet and denoising UNet blocks are mismatched."
            )
        normed = self.norm(x)
        xf = normed.reshape(b * f, n, c)

        use_ref = ref_tokens is not None and strength > 0.0
        if not use_ref:
            out = self._plain_self(xf)
        else:
            ref = ref_tokens[:, None, :, :].expand(b, f, ref_tokens.shape[1], c)
            ref = ref.reshape(b * f, ref_tokens.shape[1], c)
            out_ref = self._with_reference(xf, ref)
            if strength >= 1.0:
                out = out_ref
            else:
                out_self = self._plain_self(xf)
                out = strength * out_ref + (1.0 - strength) * out_self

        out = self.to_out(out).view(b, f, n, c)
        return x + out


class CrossAttention(nn.Module):
    """Cross-attention: Q from UNet frame tokens, K/V from the CLIP embedding.

    Production equivalent: ``attn2(norm_hidden_states, encoder_hidden_states=
    clip_image_embeds.unsqueeze(1))`` — the CLIP image embedding is a single
    context token (sequence length 1), so this layer mainly injects a global
    "what the character is" bias, not spatial detail (that is ReferenceNet's
    job).  Returning ``x`` unchanged when ``context is None`` implements the
    "no CLIP conditioning" ablation (E02) without changing shapes.
    """

    def __init__(self, dim: int, context_dim: int, heads: int = 4) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)          # production norm2
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, F, N, C]`` UNet tokens.
            context: ``[B, S, Cctx]`` conditioning tokens (CLIP), or None to skip.
        """
        if context is None:
            return x
        assert x.dim() == 4, f"CrossAttention expects [B,F,N,C]; got {tuple(x.shape)}"
        b, f, n, c = x.shape
        s = context.shape[1]
        normed = self.norm(x).reshape(b * f, n, c)
        ctx = context[:, None, :, :].expand(b, f, s, context.shape[2]).reshape(b * f, s, context.shape[2])

        def heads(t: torch.Tensor) -> torch.Tensor:
            bf, seq, ch = t.shape
            return t.view(bf, seq, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.to_q(normed)), heads(self.to_k(ctx)), heads(self.to_v(ctx))
        out = scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b * f, n, self.heads * self.head_dim)
        out = self.to_out(out).view(b, f, n, c)
        return x + out


class TemporalAttention(nn.Module):
    """Frame-mixing attention (AnimateDiff-style motion module core).

    Each spatial position (b, h, w) becomes a sequence over the frame dimension,
    positional encodings are added, and self-attention mixes information across
    frames — this is what keeps motion smooth and suppresses flicker.

    Production equivalent (``motion_module.py``): ``VersatileAttention`` with
    ``rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)``,
    sinusoidal temporal positional encoding, and a zero-initialised output
    projection (``zero_module(self.temporal_transformer.proj_out)``).  The
    zero-init makes the module an identity at initialisation, exactly like the
    reference code, so training starts from a pretrained image model.
    """

    def __init__(self, dim: int, heads: int = 4, max_frames: int = 32, zero_init_out: bool = True) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.max_frames = max_frames
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        if zero_init_out:
            zero_module(self.to_out)
        # temporal positional encoding buffer: [max_frames, dim]
        self.register_buffer(
            "pos_encoding",
            sinusoidal_timestep_embedding(torch.arange(max_frames, dtype=torch.float32), dim),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, C, F, h, w]`` video features -> same shape."""
        assert x.dim() == 5, f"TemporalAttention expects [B,C,F,h,w]; got {tuple(x.shape)}"
        b, c, f, h, w = x.shape
        if f > self.max_frames:
            raise ValueError(
                f"frame count {f} exceeds temporal max_frames {self.max_frames}; "
                "increase motion.max_frames in the config (production v1 uses 24, v2 uses 32)."
            )
        tokens = rearrange(x, "b c f h w -> (b h w) f c")
        tokens = tokens + self.pos_encoding[:f].to(tokens.dtype)[None]
        normed = self.norm(tokens)
        bf = normed.shape[0]

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bf, f, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.to_q(normed)), heads(self.to_k(normed)), heads(self.to_v(normed))
        out = scaled_dot_product_attention(q, k, v)             # [B*h*w, heads, F, dh]
        out = out.transpose(1, 2).reshape(bf, f, self.heads * self.head_dim)
        out = self.to_out(out)
        out = rearrange(out, "(b h w) f c -> b c f h w", b=b, h=h, w=w)
        return x + out

