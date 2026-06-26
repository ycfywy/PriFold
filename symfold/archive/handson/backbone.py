# -*- coding: utf-8 -*-
"""PriFold-SymFlow v4 DiT Backbone (self-contained).

Combines all building blocks:
  - Base modules from v3: SinusoidalTimeEmbedding, AxialRoPE, RMSNorm,
    TriangleMultiplicativeUpdate, GatedFFN, PatchEmbed2D, UnPatchify2D,
    OutputRefineConv, DensityHead, MultiLayerMarsFusion, MarsAttentionProj
  - v4 upgrades: CondAttentionBias, ControlInjectMLP, DilatedAxialAttentionV4,
    DASEDiTBlockV4, DASEDiT_MARS_v4
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ======================================================================
# Base modules (from v3)
# ======================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AxialRoPE(nn.Module):
    """2D axial rotary position embedding (per-axis, applied along the rolled axis)."""

    def __init__(self, dim_head: int):
        super().__init__()
        self.dim_head = dim_head
        half = dim_head // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache_len = 0
        self._cos_cache = None
        self._sin_cache = None

    def _update_cache(self, n: int, device):
        if n <= self._cache_len and self._cos_cache is not None:
            return
        pos = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        freqs = freqs.repeat(1, 2)
        self._cos_cache = freqs.cos().unsqueeze(0).unsqueeze(0)
        self._sin_cache = freqs.sin().unsqueeze(0).unsqueeze(0)
        self._cache_len = n

    @staticmethod
    def _rotate(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, q, k):
        n = q.shape[2]
        self._update_cache(n, q.device)
        cos = self._cos_cache[:, :, :n, :q.shape[-1] // 2]
        sin = self._sin_cache[:, :, :n, :q.shape[-1] // 2]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class RMSNorm(nn.Module):
    """Root mean square layernorm. Equivalent to torch.nn.RMSNorm (PyTorch 2.4+)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms) * self.weight


class DilatedAxialAttention(nn.Module):
    """Symmetry-equivariant row+col attention with shared QKV + dilation + RoPE + QK-Norm."""

    def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0,
                 dilation: int = 1, use_qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dilation = dilation
        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.rope = AxialRoPE(dim_head)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)

    @staticmethod
    def _dilate_gather(tokens, dilation):
        if dilation == 1:
            return tokens, None
        BH, W, D = tokens.shape
        pad_w = (dilation - W % dilation) % dilation
        if pad_w > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_w))
        W_pad = W + pad_w
        tokens = tokens.view(BH, dilation, W_pad // dilation, D)
        tokens = tokens.reshape(BH * dilation, W_pad // dilation, D)
        return tokens, (BH, W, pad_w, dilation)

    @staticmethod
    def _dilate_scatter(tokens, info):
        if info is None:
            return tokens
        BH, W, pad_w, dilation = info
        W_pad = W + pad_w
        _, Wd, D = tokens.shape
        tokens = tokens.view(BH, dilation, Wd, D)
        tokens = tokens.reshape(BH, W_pad, D)
        if pad_w > 0:
            tokens = tokens[:, :W, :]
        return tokens

    def _attn(self, tokens):
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(q, k)
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = F.softmax(a, dim=-1)
        out = a @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward(self, tokens):
        B, H, W, D = tokens.shape
        d = self.dilation
        # Row attention
        row = rearrange(tokens, 'b h w d -> (b h) w d')
        row_d, info_r = self._dilate_gather(row, d)
        row_out = self._attn(row_d)
        row_out = self._dilate_scatter(row_out, info_r)
        tokens = tokens + rearrange(row_out, '(b h) w d -> b h w d', b=B)
        # Col attention
        col = rearrange(tokens, 'b h w d -> (b w) h d')
        col_d, info_c = self._dilate_gather(col, d)
        col_out = self._attn(col_d)
        col_out = self._dilate_scatter(col_out, info_c)
        tokens = tokens + rearrange(col_out, '(b w) h d -> b h w d', b=B)
        return tokens


class TriangleMultiplicativeUpdate(nn.Module):
    """AF2-style triangle update on patch grid (B, H, W, D)."""

    def __init__(self, dim: int, tri_dim: int = 64):
        super().__init__()
        self.tri_dim = tri_dim
        self.proj_left = nn.Linear(dim, tri_dim)
        self.proj_right = nn.Linear(dim, tri_dim)
        self.gate_left = nn.Sequential(nn.Linear(dim, tri_dim), nn.Sigmoid())
        self.gate_right = nn.Sequential(nn.Linear(dim, tri_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(tri_dim)
        self.out_proj = nn.Linear(tri_dim, dim)
        self.out_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        # Zero-init out_proj so triangle delta starts as 0.
        # The outer block owns the residual connection; returning z here would
        # make `x = x + tri_update(norm(x))` become `x + norm(x)` at init.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z):
        left = self.proj_left(z) * self.gate_left(z)
        right = self.proj_right(z) * self.gate_right(z)
        tri = torch.einsum('bild,bljd->bijd', left, right)
        tri = self.norm(tri)
        tri = self.out_proj(tri)
        gate = self.out_gate(z)
        return gate * tri


class GatedFFN(nn.Module):
    """SwiGLU."""

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class DASEDiTBlock(nn.Module):
    """v2 block: AdaLN-Zero × (Dilated Axial Attn + optional Triangle + Gated FFN)."""

    def __init__(self, dim, num_heads=4, dim_head=64, mlp_ratio=4,
                 dropout=0.0, dilation: int = 1,
                 use_triangle: bool = False, tri_dim: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DilatedAxialAttention(dim, num_heads, dim_head, dropout, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = GatedFFN(dim, mlp_ratio, dropout)
        # AdaLN-Zero (6 outputs: shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        self.use_triangle = use_triangle
        if use_triangle:
            self.tri_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.tri_update = TriangleMultiplicativeUpdate(dim, tri_dim=tri_dim)

    def forward(self, x, cond):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)

        def expand(t):
            return t.view(t.shape[0], 1, 1, t.shape[-1])

        # Attention
        h = self.norm1(x) * (1 + expand(sc1)) + expand(sh1)
        h = self.attn(h)
        x = x + expand(g1) * h
        # Triangle Update
        if self.use_triangle:
            x = x + self.tri_update(self.tri_norm(x))
        # FFN
        h = self.norm2(x) * (1 + expand(sc2)) + expand(sh2)
        h = self.ff(h)
        x = x + expand(g2) * h
        return x


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, hidden_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x)


class UnPatchify2D(nn.Module):
    def __init__(self, hidden_dim, out_channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens):
        B, H, W, D = tokens.shape
        P = self.patch_size
        x = self.proj(tokens).view(B, H, W, self.out_channels, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, self.out_channels, H * P, W * P)


class OutputRefineConv(nn.Module):
    """3-layer Conv residual refinement at full L×L resolution (zero-init)."""

    def __init__(self, mid_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, logit):
        return logit + self.net(logit)


class DensityHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, global_feat):
        return self.head(global_feat)


# =====================================================================
# MARS attention 2D 投影模块
# =====================================================================

class MultiLayerMarsFusion(nn.Module):
    """SymFold v4/v5 风格的多层语言模型 hidden 融合。

    输入 MARS 的多个层输出（如 layer 3/6/9/12），通过：
      1) learnable softmax layer weights 做加权平均；
      2) 每层独立投影后 concat + MLP；
      3) 加权平均投影残差；
    输出一个更稳的 1D 表征，后续再 outer concat 成 pair feature。
    """

    def __init__(self, mars_dim: int = 1056, out_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mars_dim, out_dim * 2),
                nn.GELU(),
                nn.Linear(out_dim * 2, out_dim),
            ) for _ in range(num_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * num_layers, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.avg_proj = nn.Sequential(
            nn.Linear(mars_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, layer_reprs: list[torch.Tensor]) -> torch.Tensor:
        # If caller provides fewer layers (e.g. fallback), pad by repeating last.
        if len(layer_reprs) == 0:
            raise ValueError('layer_reprs must not be empty')
        if len(layer_reprs) < self.num_layers:
            layer_reprs = layer_reprs + [layer_reprs[-1]] * (self.num_layers - len(layer_reprs))
        elif len(layer_reprs) > self.num_layers:
            layer_reprs = layer_reprs[-self.num_layers:]
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = sum(w * rep for w, rep in zip(weights, layer_reprs))
        projs = [proj(rep) for proj, rep in zip(self.layer_projs, layer_reprs)]
        fused = self.fuse(torch.cat(projs, dim=-1))
        return fused + self.avg_proj(weighted)


class MarsAttentionProj(nn.Module):
    """把 MARS 的 attention stack (B, n_layer, n_head, L, L) 投影到 (B, out_dim, L, L)。

    流程：
      1) 展平 layer×head → (B, K, L, L)，K = n_layer*n_head
      2) 对称化 + APC 校正（RNA-FM / ESM 标准做法去 background bias）
      3) 1×1 Conv 投影到 out_dim
    """

    def __init__(self, n_attn_layers: int = 6, n_heads: int = 12, out_dim: int = 16):
        super().__init__()
        self.in_dim = n_attn_layers * n_heads
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_dim, out_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, 1),
        )

    @staticmethod
    def _symmetrize(a):
        return 0.5 * (a + a.transpose(-2, -1))

    @staticmethod
    def _apc(a):
        # a: (B, K, L, L) sum-1 over -1
        a_i = a.sum(dim=-1, keepdim=True)
        a_j = a.sum(dim=-2, keepdim=True)
        a_all = a.sum(dim=(-1, -2), keepdim=True)
        return a - a_i * a_j / (a_all + 1e-9)

    def forward(self, attn_stack):
        # attn_stack: (B, n_layers, n_heads, L, L) — note L includes <cls>/<eos> if not stripped
        B, NL, NH, L, _ = attn_stack.shape
        a = attn_stack.reshape(B, NL * NH, L, L)
        a = self._symmetrize(a)
        a = self._apc(a)
        a = self.proj(a)
        return a


# =====================================================================
# 主模型 DASEDiT_MARS_v2
# =====================================================================



# ======================================================================
# v4 specific modules
# ======================================================================

class CondAttentionBias(nn.Module):
    """Project patch-level pair conditions to per-head additive attention bias."""

    def __init__(self, in_channels: int, num_heads: int, zero_init: bool = True):
        super().__init__()
        hidden = max(in_channels * 2, num_heads * 2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_heads, 1),
        )
        if zero_init:
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cond_patch: torch.Tensor) -> torch.Tensor:
        bias = self.proj(cond_patch)
        return 0.5 * (bias + bias.transpose(-2, -1))


class ControlInjectMLP(nn.Module):
    """Zero-init ControlNet-style condition injection on patch tokens."""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond_patch: torch.Tensor) -> torch.Tensor:
        return self.net(cond_patch).permute(0, 2, 3, 1).contiguous()


class DilatedAxialAttentionV4(nn.Module):
    """Row/col axial attention with optional key-position bias from pair conditions."""

    def __init__(self, dim, num_heads=4, dim_head=64, dropout=0.0,
                 dilation: int = 1, use_qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dilation = dilation
        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.rope = AxialRoPE(dim_head)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)

    @staticmethod
    def _dilate_gather(tokens, dilation):
        if dilation == 1:
            return tokens, None
        bh, w, d = tokens.shape
        pad_w = (dilation - w % dilation) % dilation
        if pad_w > 0:
            tokens = F.pad(tokens, (0, 0, 0, pad_w))
        w_pad = w + pad_w
        tokens = tokens.view(bh, dilation, w_pad // dilation, d)
        tokens = tokens.reshape(bh * dilation, w_pad // dilation, d)
        return tokens, (bh, w, pad_w, dilation)

    @staticmethod
    def _dilate_scatter(tokens, info):
        if info is None:
            return tokens
        bh, w, pad_w, dilation = info
        w_pad = w + pad_w
        _, wd, d = tokens.shape
        tokens = tokens.view(bh, dilation, wd, d)
        tokens = tokens.reshape(bh, w_pad, d)
        if pad_w > 0:
            tokens = tokens[:, :w, :]
        return tokens

    def _attn(self, tokens, key_bias=None):
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(q, k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_bias is not None:
            # key_bias: (B_axis, N, heads), added as per-key logit bias.
            attn = attn + key_bias.permute(0, 2, 1).unsqueeze(2)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def forward(self, tokens, attn_bias=None):
        b, h, w, d_model = tokens.shape
        dilation = self.dilation

        row = rearrange(tokens, 'b h w d -> (b h) w d')
        row_bias = None
        if attn_bias is not None:
            # attn_bias: (B, num_heads, H, W). For row attention each token row i
            # attends along j, so bias[b, head, i, j] is the additive key bias.
            row_bias = attn_bias.permute(0, 2, 3, 1).contiguous().reshape(b * h, w, self.num_heads)
        row_d, info_r = self._dilate_gather(row, dilation)
        row_bias_d = None
        if row_bias is not None:
            row_bias_d, _ = self._dilate_gather(row_bias, dilation)
        row_out = self._attn(row_d, row_bias_d)
        row_out = self._dilate_scatter(row_out, info_r)
        tokens = tokens + rearrange(row_out, '(b h) w d -> b h w d', b=b)

        col = rearrange(tokens, 'b h w d -> (b w) h d')
        col_bias = None
        if attn_bias is not None:
            # For column attention each token column j attends along i, so bias is
            # rearranged to (b, j, i_k, head).
            col_bias = attn_bias.permute(0, 3, 2, 1).contiguous().reshape(b * w, h, self.num_heads)
        col_d, info_c = self._dilate_gather(col, dilation)
        col_bias_d = None
        if col_bias is not None:
            col_bias_d, _ = self._dilate_gather(col_bias, dilation)
        col_out = self._attn(col_d, col_bias_d)
        col_out = self._dilate_scatter(col_out, info_c)
        tokens = tokens + rearrange(col_out, '(b w) h d -> b h w d', b=b)
        return tokens


class DASEDiTBlockV4(nn.Module):
    """AdaLN-Zero × biased axial attention + triangle + SwiGLU."""

    def __init__(self, dim, num_heads=4, dim_head=64, mlp_ratio=4,
                 dropout=0.0, dilation: int = 1,
                 use_triangle: bool = False, tri_dim: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = DilatedAxialAttentionV4(dim, num_heads, dim_head, dropout, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = GatedFFN(dim, mlp_ratio, dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        self.use_triangle = use_triangle
        if use_triangle:
            self.tri_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.tri_update = TriangleMultiplicativeUpdate(dim, tri_dim=tri_dim)

    def forward(self, x, cond, attn_bias=None):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)

        def expand(t):
            return t.view(t.shape[0], 1, 1, t.shape[-1])

        h = self.norm1(x) * (1 + expand(sc1)) + expand(sh1)
        h = self.attn(h, attn_bias=attn_bias)
        x = x + expand(g1) * h
        if self.use_triangle:
            x = x + self.tri_update(self.tri_norm(x))
        h = self.norm2(x) * (1 + expand(sc2)) + expand(sh2)
        h = self.ff(h)
        x = x + expand(g2) * h
        return x


class DASEDiT_MARS_v4(nn.Module):
    """v4 backbone: v3 features + per-layer pair bias + direct score head."""

    def __init__(self,
                 mars_dim: int = 1056,
                 n_attn_layers: int = 6,
                 n_heads_mars: int = 12,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dim_head: int = 64,
                 num_layers: int = 9,
                 patch_size: int = 4,
                 mars_emb_proj_dim: int = 32,
                 mars_attn_proj_dim: int = 16,
                 mars_hidden_fusion_dim: int = 64,
                 mars_hidden_layers: int = 4,
                 xt_emb_dim: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 dilation_pattern: list | None = None,
                 tri_start_layer: int = 6,
                 tri_dim: int = 64,
                 refine_mid_ch: int = 16,
                 use_seq_oh: bool = True,
                 cond_bias_zero_init: bool = True,
                 control_every: int = 2):
        super().__init__()
        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4]
        assert len(dilation_pattern) == num_layers
        self.patch_size = patch_size
        self.use_seq_oh = use_seq_oh
        self.control_every = int(control_every)

        self.x_t_embedding = nn.Embedding(2, xt_emb_dim)
        self.mars_hidden_fusion = MultiLayerMarsFusion(
            mars_dim=mars_dim, out_dim=mars_hidden_fusion_dim, num_layers=mars_hidden_layers)
        self.mars_emb_proj = nn.Sequential(
            nn.Linear(mars_hidden_fusion_dim, mars_emb_proj_dim * 2),
            nn.GELU(),
            nn.Linear(mars_emb_proj_dim * 2, mars_emb_proj_dim),
        )
        self.mars_attn_proj = MarsAttentionProj(
            n_attn_layers=n_attn_layers, n_heads=n_heads_mars, out_dim=mars_attn_proj_dim)

        in_channels = xt_emb_dim + 2 * mars_emb_proj_dim + mars_attn_proj_dim + 1
        if use_seq_oh:
            in_channels += 8
        self.in_channels = in_channels
        self.patch_embed = PatchEmbed2D(in_channels, hidden_dim, patch_size)

        cond_channels = mars_attn_proj_dim + 1
        self.cond_bias = CondAttentionBias(cond_channels, num_heads, zero_init=cond_bias_zero_init)
        self.control_inject = ControlInjectMLP(cond_channels, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.mars_global = nn.Sequential(
            nn.Linear(mars_emb_proj_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.density_emb = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim))
        self.cond_fuse = nn.Linear(3 * hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DASEDiTBlockV4(
                hidden_dim, num_heads, dim_head, mlp_ratio, dropout,
                dilation=dilation_pattern[i],
                use_triangle=(i >= tri_start_layer),
                tri_dim=tri_dim)
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.unpatch = UnPatchify2D(hidden_dim, 1, patch_size)
        self.refine = OutputRefineConv(mid_ch=refine_mid_ch)
        self.direct_unpatch = UnPatchify2D(hidden_dim, 1, patch_size)
        self.direct_refine = OutputRefineConv(mid_ch=refine_mid_ch)
        self.density_head = DensityHead(hidden_dim)

    @staticmethod
    def _outer_concat(x):
        b, c, l = x.shape
        xi = x.unsqueeze(-1).expand(-1, -1, -1, l)
        xj = x.unsqueeze(-2).expand(-1, -1, l, -1)
        return torch.cat([xi, xj], dim=1)

    @staticmethod
    def _mask_logits(logit, contact_masks):
        logit = 0.5 * (logit + logit.transpose(-2, -1))
        l = logit.shape[-1]
        idx = torch.arange(l, device=logit.device)
        short = (idx.view(l, 1) - idx.view(1, l)).abs() < 3
        logit = logit.masked_fill(short.view(1, 1, l, l), -10.0)
        if contact_masks is not None:
            logit = logit.masked_fill(contact_masks < 0.5, -10.0)
        return logit

    def _build_features(self, x_t, mars_emb_1d, mars_attn_2d, pos_bias, seq_oh=None):
        x_long = x_t.long().squeeze(1)
        x_emb = self.x_t_embedding(x_long).permute(0, 3, 1, 2).contiguous()
        mars_2d = self._outer_concat(mars_emb_1d.permute(0, 2, 1).contiguous())
        parts = [x_emb, mars_2d, mars_attn_2d, pos_bias.unsqueeze(1)]
        if self.use_seq_oh:
            if seq_oh is None:
                b, _, l, _ = x_t.shape
                seq_2d = torch.zeros(b, 8, l, l, device=x_t.device, dtype=x_emb.dtype)
            else:
                seq_2d = self._outer_concat(seq_oh.permute(0, 2, 1).contiguous()).to(dtype=x_emb.dtype)
            parts.append(seq_2d)
        f = torch.cat(parts, dim=1)
        return 0.5 * (f + f.transpose(-2, -1))

    def _global_cond(self, t, mars_emb_1d, density_hint):
        te = self.time_mlp(t)
        me = self.mars_global(mars_emb_1d.mean(dim=1))
        if density_hint is None:
            zero_hint = torch.zeros(te.shape[0], 1, device=te.device, dtype=te.dtype)
            de = self.density_emb(zero_hint)
        else:
            de = self.density_emb(density_hint)
        return self.cond_fuse(torch.cat([te, me, de], dim=-1))

    def forward(self, x_t, t, *, mars_hidden, mars_attn, pos_bias,
                mars_hidden_layers=None, seq_oh=None,
                contact_masks=None, density_hint=None,
                return_density: bool = False,
                return_direct: bool = False):
        if mars_hidden_layers is None:
            mars_hidden_layers = [mars_hidden]
        mars_fused = self.mars_hidden_fusion(mars_hidden_layers)
        mars_emb_1d = self.mars_emb_proj(mars_fused)
        mars_attn_2d = self.mars_attn_proj(mars_attn)

        features = self._build_features(x_t, mars_emb_1d, mars_attn_2d, pos_bias, seq_oh=seq_oh)
        tokens = self.patch_embed(features)
        cond_pair = torch.cat([mars_attn_2d, pos_bias.unsqueeze(1)], dim=1)
        cond_patch = F.avg_pool2d(cond_pair, kernel_size=self.patch_size, stride=self.patch_size)
        attn_bias = self.cond_bias(cond_patch)
        cond = self._global_cond(t, mars_emb_1d, density_hint)

        for idx, blk in enumerate(self.blocks):
            tokens = blk(tokens, cond, attn_bias=attn_bias)
            if self.control_every > 0 and (idx + 1) % self.control_every == 0:
                tokens = tokens + self.control_inject(cond_patch)

        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)

        def expand(x):
            return x.view(x.shape[0], 1, 1, x.shape[-1])

        final_tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self._mask_logits(self.refine(self.unpatch(final_tokens)), contact_masks)

        outs = [logit]
        if return_density:
            outs.append(self.density_head(cond))
        if return_direct:
            direct_logit = self._mask_logits(self.direct_refine(self.direct_unpatch(final_tokens)), contact_masks)
            outs.append(direct_logit)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)


if __name__ == '__main__':
    torch.manual_seed(0)
    b, l = 2, 32
    m = DASEDiT_MARS_v4(
        mars_dim=128, n_attn_layers=2, n_heads_mars=4,
        hidden_dim=64, num_heads=2, dim_head=16, num_layers=9,
        patch_size=4, dilation_pattern=[1, 1, 1, 2, 2, 2, 4, 4, 4],
        tri_start_layer=6, tri_dim=32).eval()
    x_t = (torch.rand(b, 1, l, l) > 0.99).float()
    x_t = torch.maximum(x_t, x_t.transpose(-2, -1))
    t = torch.rand(b)
    mars_hidden = torch.randn(b, l, 128)
    mars_layers = [torch.randn(b, l, 128) for _ in range(4)]
    mars_attn = torch.softmax(torch.randn(b, 2, 4, l, l), dim=-1)
    pos_bias = torch.randn(b, l, l)
    pos_bias = 0.5 * (pos_bias + pos_bias.transpose(-2, -1))
    cm = torch.ones(b, 1, l, l)
    logit, den, direct = m(
        x_t, t, mars_hidden=mars_hidden, mars_hidden_layers=mars_layers,
        mars_attn=mars_attn, pos_bias=pos_bias, contact_masks=cm,
        density_hint=None, return_density=True, return_direct=True)
    print(f'in_channels = {m.in_channels}')
    print(f'logit={tuple(logit.shape)} density={tuple(den.shape)} direct={tuple(direct.shape)}')
    print(f'params={sum(p.numel() for p in m.parameters()):,}')
