# -*- coding: utf-8 -*-
"""PriFold-SymFlow v12 — Generative RNA Folding via Discrete Flow Matching.

核心创新点: 用**生成式离散 Flow Matching** 做 RNA 二级结构预测
（区别于判别式的 v9/v10），并吸收 v9 验证有效的架构改进。

设计融合:
  - 生成式范式 (创新):  离散 Bernoulli Flow Matching (CTMC + τ-leap)   [来自 v6]
  - v6 式加速:          patch_size=4 压缩到 (L/4)×(L/4) space 做主干计算   [来自 v6]
  - 双轨表示 (重构):    single (1D) + pair (2D) 双表示, OuterProductMean 通信 [类 Evoformer]
  - 位置编码:            2D RoPE (v9 消融 +11.9pp)                       [来自 v9]
  - 高效注意力:          Flash Attention (scaled_dot_product_attention)  [来自 v9]
  - 正则化:              DropPath(线性递增) + Dropout                    [来自 v9]
  - 时间条件:            AdaLN-Zero (DiT, 生成式必需)                    [来自 DiT]
  - 损失:                模块化 (Focal BCE + Dice + 结构约束)            [来自 v6]

Pipeline:
  RNA seq → MARS-LX
         → single s + pair z + noisy x_t
         → learned patch embedding to (L/patch)×(L/patch) space
         → DualFlowDiT blocks in patch space:
              single self-attn (1D, RoPE, AdaLN-Zero on t)
              OuterProductMean(single) → pair      (1D→2D 通信)
              pair axial attn (row+col, 2D RoPE, Flash, AdaLN-Zero on t)
         → unpatch + refine → contact logit → τ-leap CTMC sampling → binary contact map
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from prifold.llama2_with_attn import mars_forward_with_attn

# 离散 flow 基础设施 + 模块化损失 (v12 自包含, 不依赖 v3/v4/v6)
from symfold.v12.discrete_flow import (
    sample_x_t_given_x_1,
    symmetrize_binary,
    symmetrize_logit,
    compute_ctmc_rates,
    project_to_valid_contact_map,
    ModularFlowLoss,
)


# ============================================================
# Building blocks
# ============================================================

class DropPath(nn.Module):
    """Stochastic depth / drop path (v9)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal timestep embedding for flow time t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class RoPE2D(nn.Module):
    """Rotary Position Embedding, applied per-axis to axial / 1D attention (v9).

    对 single (1D) 和 pair (row/col) attention 施加 RoPE，使模型感知相对距离 |i-j|，
    对长距离 RNA 配对建模至关重要 (v9 消融: +11.9pp F1)。
    """
    def __init__(self, dim_head: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer('cos_cache', cos, persistent=False)
        self.register_buffer('sin_cache', sin, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(self, q, k, seq_len: int):
        """q, k: (..., S, D). cos/sin broadcast over leading dims."""
        cos = self.cos_cache[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class PatchEmbed2D(nn.Module):
    """v6-style learned 2D patch embedding: (B,C,L,L) -> (B,L/P,L/P,D)."""

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class UnPatchify2D(nn.Module):
    """v6-style learned unpatchify: (B,H,W,D) -> (B,C,H*P,W*P)."""

    def __init__(self, hidden_dim: int, out_channels: int = 1, patch_size: int = 4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.proj = nn.Linear(hidden_dim, out_channels * self.patch_size * self.patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, H, W, D = tokens.shape
        P = self.patch_size
        x = self.proj(tokens).view(B, H, W, self.out_channels, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, self.out_channels, H * P, W * P)


class OutputRefineConv(nn.Module):
    """Small full-resolution refinement head, zero-init so it starts as identity."""

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

    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        return logit + self.net(logit)


class SinglePatchEmbed1D(nn.Module):
    """Compress 1D single representation to the same patch grid length as pair tokens."""

    def __init__(self, dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv1d(dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.transpose(1, 2)).transpose(1, 2).contiguous()
        return self.norm(x)


class SingleAttention(nn.Module):
    """1D multi-head self-attention over the sequence (single track), RoPE + Flash."""

    def __init__(self, dim: int, num_heads: int = 4, dim_head: int = 32,
                 dropout: float = 0.0, use_rope: bool = True, max_len: int = 512):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.attn_dropout = dropout
        self.qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.out = nn.Linear(inner_dim, dim)
        if use_rope:
            self.rope = RoPE2D(dim_head, max_len)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, D); seq_mask: (B, L), 1/True for valid tokens."""
        B, L, _ = x.shape
        h = self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b l (h d) -> b h l d', h=h)
        k = rearrange(k, 'b l (h d) -> b h l d', h=h)
        v = rearrange(v, 'b l (h d) -> b h l d', h=h)
        if self.use_rope:
            q, k = self.rope.apply_rotary(q, k, L)
        dp = self.attn_dropout if self.training else 0.0
        attn_mask = None
        if seq_mask is not None:
            attn_mask = seq_mask.to(torch.bool).view(B, 1, 1, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.out(out)


class OuterProductMean(nn.Module):
    """Single→Pair communication (Evoformer-style outer product).

    单序列场景没有 MSA 维度，故无 "mean"，等价于 single 的外积投影:
        a_i (h) ⊗ b_j (h)  →  (i,j) 的 h*h 维特征  →  Linear → pair_dim
    输出 zero-init，初始不干扰 pair（残差友好）。

    注意显存: 中间张量 (B, L, L, h*h)。L 较大时务必控制 opm_hidden。
    """

    def __init__(self, single_dim: int, pair_dim: int, hidden: int = 16):
        super().__init__()
        self.hidden = hidden
        self.norm = nn.LayerNorm(single_dim)
        self.proj_a = nn.Linear(single_dim, hidden)
        self.proj_b = nn.Linear(single_dim, hidden)
        self.out = nn.Linear(hidden * hidden, pair_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, L, single_dim) → (B, L, L, pair_dim)."""
        B, L, _ = s.shape
        s = self.norm(s)
        a = self.proj_a(s)  # (B, L, h)
        b = self.proj_b(s)  # (B, L, h)
        # outer[b,i,j,h,k] = a[b,i,h] * b[b,j,k]
        outer = torch.einsum('bih,bjk->bijhk', a, b)  # (B, L, L, h, h)
        outer = outer.reshape(B, L, L, self.hidden * self.hidden)
        return self.out(outer)  # (B, L, L, pair_dim)


class DualFlowDiTBlock(nn.Module):
    """Dual-track block: single self-attn + OuterProductMean + pair axial attn.

    融合点:
      - single track:    1D self-attention (RoPE + Flash) + FFN, AdaLN-Zero(t)
      - communication:   OuterProductMean(single) 加到 pair (1D→2D)
      - pair track:       row + col axial attention (2D RoPE + Flash) + FFN, AdaLN-Zero(t)
      - DiT: AdaLN-Zero 注入 flow 时间条件 t (生成式必需)
    """

    def __init__(self, pair_dim: int, single_dim: int,
                 num_heads: int = 8, dim_head: int = 32,
                 single_heads: int = 4, single_dim_head: int = 32,
                 cond_dim: int = 256, ff_mult: int = 4,
                 dropout: float = 0.2, drop_path: float = 0.0,
                 use_rope: bool = True, max_len: int = 512,
                 opm_hidden: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.use_rope = use_rope
        self.attn_dropout = dropout

        # ---------- single track ----------
        self.single_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * single_dim))
        nn.init.zeros_(self.single_adaln[-1].weight)
        nn.init.zeros_(self.single_adaln[-1].bias)
        self.single_attn_norm = nn.LayerNorm(single_dim, elementwise_affine=False, eps=1e-6)
        self.single_attn = SingleAttention(single_dim, single_heads, single_dim_head,
                                            dropout=dropout, use_rope=use_rope, max_len=max_len)
        self.single_ffn_norm = nn.LayerNorm(single_dim, elementwise_affine=False, eps=1e-6)
        self.single_ffn = nn.Sequential(
            nn.Linear(single_dim, single_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(single_dim * ff_mult, single_dim),
            nn.Dropout(dropout),
        )

        # ---------- communication: single → pair ----------
        self.opm = OuterProductMean(single_dim, pair_dim, hidden=opm_hidden)

        # ---------- pair track ----------
        inner_dim = num_heads * dim_head
        self.pair_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * pair_dim))
        nn.init.zeros_(self.pair_adaln[-1].weight)
        nn.init.zeros_(self.pair_adaln[-1].bias)
        self.row_norm = nn.LayerNorm(pair_dim, elementwise_affine=False, eps=1e-6)
        self.row_qkv = nn.Linear(pair_dim, 3 * inner_dim, bias=False)
        self.row_out = nn.Linear(inner_dim, pair_dim)
        self.col_norm = nn.LayerNorm(pair_dim, elementwise_affine=False, eps=1e-6)
        self.col_qkv = nn.Linear(pair_dim, 3 * inner_dim, bias=False)
        self.col_out = nn.Linear(inner_dim, pair_dim)
        self.ffn_norm = nn.LayerNorm(pair_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(pair_dim, pair_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * ff_mult, pair_dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if use_rope:
            self.pair_rope = RoPE2D(dim_head, max_len)

    def _axial_attn(self, x, qkv_proj, out_proj, key_mask: torch.Tensor | None = None):
        """Multi-head axial self-attention with RoPE + Flash Attention (SDPA).

        x: (B, N, S, D) — attention along the last sequence axis S.
        key_mask: (B, N, S), 1/True for valid keys along the S axis.
        """
        B, N, S, _ = x.shape
        h = self.num_heads
        qkv = qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n s (h d) -> (b n) h s d', h=h)
        k = rearrange(k, 'b n s (h d) -> (b n) h s d', h=h)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        if self.use_rope:
            q, k = self.pair_rope.apply_rotary(q, k, S)
        dp = self.attn_dropout if self.training else 0.0
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask.to(torch.bool).reshape(B * N, 1, 1, S)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=N)
        return out_proj(out)

    def forward(self, single: torch.Tensor, pair: torch.Tensor,
                cond: torch.Tensor, seq_mask: torch.Tensor | None = None,
                pair_mask: torch.Tensor | None = None):
        """single: (B, L, Ds); pair: (B, L, L, Dp); cond: (B, cond_dim).

        seq_mask:  (B, L), valid nucleotide positions.
        pair_mask: (B, L, L), valid pair positions before channel expansion.
        """
        B, L, Ds = single.shape
        Dp = pair.shape[-1]
        seq_gate = seq_mask.to(single.dtype).unsqueeze(-1) if seq_mask is not None else None
        pair_gate = pair_mask.to(pair.dtype).unsqueeze(-1) if pair_mask is not None else None

        if seq_gate is not None:
            single = single * seq_gate
        if pair_gate is not None:
            pair = pair * pair_gate

        # ===== single track (modulated by time via AdaLN) =====
        s_sh1, s_sc1, s_g1, s_sh2, s_sc2, s_g2 = self.single_adaln(cond).chunk(6, dim=-1)

        def es(p):  # broadcast (B, Ds) over L → (B, 1, Ds)
            return p.unsqueeze(1)

        h = self.single_attn_norm(single) * (1 + es(s_sc1)) + es(s_sh1)
        h = self.single_attn(h, seq_mask=seq_mask)
        single = single + self.drop_path(es(s_g1) * h)
        if seq_gate is not None:
            single = single * seq_gate

        h = self.single_ffn_norm(single) * (1 + es(s_sc2)) + es(s_sh2)
        h = self.single_ffn(h)
        single = single + self.drop_path(es(s_g2) * h)
        if seq_gate is not None:
            single = single * seq_gate

        # ===== communication: single → pair =====
        pair = pair + self.opm(single)
        if pair_gate is not None:
            pair = pair * pair_gate

        # ===== pair track (modulated by time via AdaLN) =====
        p_sh1, p_sc1, p_g1, p_sh2, p_sc2, p_g2 = self.pair_adaln(cond).chunk(6, dim=-1)

        def ep(p):  # broadcast (B, Dp) over L×L → (B, 1, 1, Dp)
            return p.view(B, 1, 1, Dp)

        row_key_mask = pair_mask if pair_mask is not None else None
        col_key_mask = pair_mask.transpose(1, 2) if pair_mask is not None else None

        h = self.row_norm(pair) * (1 + ep(p_sc1)) + ep(p_sh1)
        h = self._axial_attn(h, self.row_qkv, self.row_out, key_mask=row_key_mask)
        pair = pair + self.drop_path(ep(p_g1) * h)
        if pair_gate is not None:
            pair = pair * pair_gate

        h = (self.col_norm(pair) * (1 + ep(p_sc1)) + ep(p_sh1)).transpose(1, 2)
        h = self._axial_attn(h, self.col_qkv, self.col_out, key_mask=col_key_mask).transpose(1, 2)
        pair = pair + self.drop_path(ep(p_g1) * h)
        if pair_gate is not None:
            pair = pair * pair_gate

        h = self.ffn_norm(pair) * (1 + ep(p_sc2)) + ep(p_sh2)
        h = self.ffn(h)
        pair = pair + self.drop_path(ep(p_g2) * h)
        if pair_gate is not None:
            pair = pair * pair_gate

        return single, pair


# ============================================================
# Main Model: Discrete Flow Matching + Dual-track FlowDiT
# ============================================================

class RNAFlowDiT(nn.Module):
    """Generative RNA Secondary Structure Prediction via Discrete Flow Matching.

    - Condition:  MARS single (1D hidden) + pair (2D attention) 双表示
    - Input:      noisy binary contact map x_t at flow time t (additive injection on pair)
    - Backbone:   patch-space DualFlowDiT (single self-attn + OPM + pair axial attn + AdaLN-Zero)
    - Output:     full-resolution contact logit  p(x_1 = 1 | x_t, t)
    - Training:   modular discrete flow loss (Focal BCE + Dice + structural priors)
    - Inference:  τ-leap CTMC sampling + greedy projection
    """

    def __init__(self,
                 extractor: nn.Module,
                 # MARS
                 freeze_mars: bool = True,
                 mars_dim: int = 1056,
                 mars_n_attn_layers: int = 6,
                 mars_n_heads: int = 12,
                 mars_hidden_layer_indices: list | None = None,
                 # FlowDiT
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dim_head: int = 32,
                 num_layers: int = 8,
                 ff_mult: int = 4,
                 dropout: float = 0.2,
                 drop_path: float = 0.15,
                 use_rope: bool = True,
                 max_len: int = 512,
                 use_gradient_checkpoint: bool = False,
                 patch_size: int = 4,
                 refine_mid_ch: int = 16,
                 # Dual-track (single + pair)
                 single_dim: int = 128,
                 single_heads: int = 4,
                 single_dim_head: int = 32,
                 opm_hidden: int = 16,
                 # Discrete flow
                 rho_0: float = 0.005,
                 loss_config: dict | None = None,
                 ):
        super().__init__()
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.hidden_dim = hidden_dim
        self.single_dim = single_dim
        self.patch_size = int(patch_size)
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.rho_0 = rho_0

        # MARS extractor
        self.extractor = extractor
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        # --- single representation: MARS 1D hidden → single_dim → patch tokens ---
        self.mars_1d_proj = nn.Sequential(
            nn.Linear(mars_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, single_dim),
        )
        self.single_patch = SinglePatchEmbed1D(single_dim, self.patch_size)

        # --- pair representation: MARS 2D attention → v6-style patch tokens ---
        mars_attn_ch = mars_n_attn_layers * mars_n_heads  # 72
        self.mars_2d_proj = nn.Sequential(
            nn.Conv2d(mars_attn_ch, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
        )
        self.pair_patch = PatchEmbed2D(64, hidden_dim, self.patch_size)

        # Noisy x_t injection is also patchified; this is the v6 speed path.
        self.xt_patch = PatchEmbed2D(1, hidden_dim, self.patch_size)

        # Time conditioning
        cond_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmb(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # DualFlowDiT blocks (linear DropPath schedule)
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            DualFlowDiTBlock(pair_dim=hidden_dim, single_dim=single_dim,
                             num_heads=num_heads, dim_head=dim_head,
                             single_heads=single_heads, single_dim_head=single_dim_head,
                             cond_dim=cond_dim, ff_mult=ff_mult,
                             dropout=dropout, drop_path=dpr[i],
                             use_rope=use_rope, max_len=max_len,
                             opm_hidden=opm_hidden)
            for i in range(num_layers)
        ])

        # Output head: unpatch patch-space pair tokens back to full L×L logits.
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
        self.unpatch = UnPatchify2D(hidden_dim, 1, self.patch_size)
        self.refine = OutputRefineConv(mid_ch=refine_mid_ch)

        # Modular discrete flow loss (复用 v6)
        if loss_config is None:
            loss_config = self._default_loss_config()
        self.flow_loss = ModularFlowLoss(loss_config, rho_0=rho_0)

    @staticmethod
    def _default_loss_config() -> dict:
        """单 flow head 的默认损失配置。

        把 flow logit 同时作为 direct_logit 传入、并关闭 'direct' BCE，
        即可让 Dice/PairCount/RatioPenalty 作用在 flow head 上；
        BCE/Stacking/NonCrossing 也作用在同一 flow head。
        """
        return {
            'bce': {'enabled': True, 'pos_weight_base': 99.0, 'pos_weight_min': 10.0,
                    'focal_gamma': 1.0, 'time_weight': True},
            'dice': {'enabled': True, 'weight': 0.5, 'smooth': 1.0},
            'tversky': {'enabled': False, 'weight': 0.5, 'alpha': 0.3, 'beta': 0.7, 'smooth': 1.0},
            'pair_count': {'enabled': True, 'weight': 0.3},
            'ratio_penalty': {'enabled': True, 'weight': 0.2, 'threshold': 1.2},
            'density': {'enabled': False, 'weight': 0.2},
            'direct': {'enabled': False, 'weight': 0.4},
            'stacking': {'enabled': True, 'weight': 0.05},
            'non_crossing': {'enabled': True, 'weight': 0.03},
            'label_smoothing': {'enabled': False, 'epsilon': 0.01},
        }

    # ------------------------------------------------------------------
    # MARS feature extraction
    # ------------------------------------------------------------------

    def _extract_mars(self, input_ids, attention_mask, L):
        if self.freeze_mars:
            self.extractor.eval()
            with torch.no_grad():
                hidden, attn_stack, _ = mars_forward_with_attn(
                    self.extractor, input_ids, attention_mask,
                    n_attn_layers=self.mars_n_attn_layers,
                    hidden_layer_indices=self.mars_hidden_layer_indices,
                    return_hidden_layers=True)
        else:
            hidden, attn_stack, _ = mars_forward_with_attn(
                self.extractor, input_ids, attention_mask,
                n_attn_layers=self.mars_n_attn_layers,
                hidden_layer_indices=self.mars_hidden_layer_indices,
                return_hidden_layers=True)

        hidden = hidden[:, 1:1+L, :]
        attn_stack = attn_stack[:, :, :, 1:1+L, 1:1+L]
        cur_len = hidden.shape[1]
        if cur_len < L:
            hidden = F.pad(hidden, (0, 0, 0, L - cur_len))
            attn_stack = F.pad(attn_stack, (0, L - cur_len, 0, L - cur_len))
        return hidden, attn_stack

    def _build_features(self, mars_hidden, mars_attn):
        """Build compressed single + pair representations.

        v6 speed trick: MARS features are patchified before the expensive FlowDiT
        blocks, so the backbone works on `(L/P)×(L/P)` pair tokens instead of
        full `L×L` pair tokens.
        """
        B, L, _ = mars_hidden.shape

        # single: 1D sequence representation, compressed to patch length L/P.
        single = self.mars_1d_proj(mars_hidden)          # (B, L, single_dim)
        single = self.single_patch(single)               # (B, L/P, single_dim)

        # pair: 2D attention representation, learned patch embedding to hidden_dim.
        attn_flat = mars_attn.reshape(B, -1, L, L)       # (B, 72, L, L)
        pair_2d = self.mars_2d_proj(attn_flat)           # (B, 64, L, L)
        pair = self.pair_patch(pair_2d)                  # (B, L/P, L/P, hidden_dim)
        return single, pair

    # ------------------------------------------------------------------
    # DualFlowDiT forward: predict logit p(x_1=1 | x_t, t)
    # ------------------------------------------------------------------

    def _dit_forward(self, single: torch.Tensor, base_pair: torch.Tensor,
                     x_t: torch.Tensor, t: torch.Tensor,
                     contact_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        single:       (B, L/P, single_dim) — patch-space MARS 1D conditioning
        base_pair:    (B, L/P, L/P, D)     — patch-space MARS 2D conditioning
        x_t:          (B, 1, L, L)         — binary noisy contact map at time t
        t:            (B,)                 — flow time
        contact_mask: (B, 1, L, L)         — full-resolution valid mask
        Returns:      (B, 1, L, L)         — symmetrized and masked contact logit
        """
        B, _, L_full, _ = x_t.shape
        P = self.patch_size

        pair_mask = None
        seq_mask = None
        pair_gate = None
        if contact_mask is not None:
            # Patch is valid if any full-resolution cell inside it is valid.
            pair_mask = F.max_pool2d(contact_mask.float(), kernel_size=P, stride=P).squeeze(1).to(torch.bool)
            seq_mask = pair_mask.any(dim=1)
            pair_gate = pair_mask.to(base_pair.dtype).unsqueeze(-1)

        # Additive injection of noisy state x_t after learned patch embedding.
        pair = base_pair + self.xt_patch(x_t)  # (B, L/P, L/P, D)
        if pair_gate is not None:
            pair = pair * pair_gate
        s = single
        if seq_mask is not None:
            s = s * seq_mask.to(s.dtype).unsqueeze(-1)

        cond = self.time_embed(t)  # (B, cond_dim)

        for block in self.blocks:
            if self.use_gradient_checkpoint and self.training:
                s, pair = torch.utils.checkpoint.checkpoint(
                    block, s, pair, cond, seq_mask, pair_mask, use_reentrant=False)
            else:
                s, pair = block(s, pair, cond, seq_mask, pair_mask)

        # Time-conditioned output head in patch space, then unpatch to full L×L.
        sh, sc = self.final_adaln(cond).chunk(2, dim=-1)
        pair = self.final_norm(pair) * (1 + sc.view(B, 1, 1, -1)) + sh.view(B, 1, 1, -1)
        logit = self.refine(self.unpatch(pair))
        if logit.shape[-1] != L_full:
            logit = logit[:, :, :L_full, :L_full]
        logit = 0.5 * (logit + logit.transpose(-2, -1))

        # Unified invalid-position masking: no short-range pairs and no padded tokens.
        idx = torch.arange(L_full, device=logit.device)
        short = (idx.view(L_full, 1) - idx.view(1, L_full)).abs() < 3
        logit = logit.masked_fill(short.view(1, 1, L_full, L_full), -10.0)
        if contact_mask is not None:
            logit = logit.masked_fill(contact_mask < 0.5, -10.0)
        return logit

    # ------------------------------------------------------------------
    # Training: Discrete (Bernoulli) Flow Matching
    # ------------------------------------------------------------------

    def forward(self, batch: dict):
        """Training forward pass.

        x_1 = symmetrize(GT contact)            (binary)
        x_t ~ Bernoulli( t·x_1 + (1-t)·ρ_0 )    (noising)
        logit = DualFlowDiT(x_t, t, cond)       → p(x_1=1|x_t,t)
        loss = ModularFlowLoss(logit, x_1, t)
        """
        contact = batch['contact']            # (B, 1, L, L)
        contact_mask = batch['contact_mask']  # (B, 1, L, L)
        x_1 = symmetrize_binary(contact) * contact_mask

        B = x_1.shape[0]
        L = x_1.shape[-1]
        device = x_1.device

        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], L)
        single, base_pair = self._build_features(mars_hidden, mars_attn)

        # Sample time + Bernoulli noising
        t = torch.rand(B, device=device)
        x_t = sample_x_t_given_x_1(x_1, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_mask

        # Prediction (logit)
        logit = self._dit_forward(single, base_pair, x_t, t, contact_mask)  # (B, 1, L, L)

        # Modular loss. flow logit 同时作为 direct_logit (direct BCE 已关闭)。
        total_loss, loss_dict = self.flow_loss(
            logit, x_1, t, contact_mask,
            density_pred=None,
            direct_logit=logit)
        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Inference: τ-leap CTMC sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch: dict, num_steps: int = 20, threshold: float = 0.5):
        """Generate contact map via τ-leap CTMC sampling.

        x_0 ~ Bernoulli(ρ_0); 按 CTMC rates 逐步翻转 (cosine schedule);
        最终 greedy projection 到合法接触图 (对称, |i-j|>=3, ≤1 pair/row)。
        """
        contact_mask = batch['contact_mask']  # (B, 1, L, L)
        device = contact_mask.device
        B, _, L, _ = contact_mask.shape

        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], L)
        single, base_pair = self._build_features(mars_hidden, mars_attn)  # 仅算一次

        x_t = (torch.rand(B, 1, L, L, device=device) < self.rho_0).float()
        x_t = symmetrize_binary(x_t) * contact_mask

        raw = [math.sin(math.pi * (k + 0.5) / (2 * num_steps)) for k in range(num_steps)]
        total_raw = sum(raw)
        dt_list = [r / total_raw for r in raw]

        p_last = torch.zeros_like(x_t)
        t_cum = 0.0
        for dt in dt_list:
            t_tensor = torch.full((B,), t_cum, device=device)
            logit = symmetrize_logit(self._dit_forward(single, base_pair, x_t, t_tensor, contact_mask))
            p_x1 = torch.sigmoid(logit)
            p_x1 = 0.5 * (p_x1 + p_x1.transpose(-2, -1))
            p_last = p_x1

            rate_01, rate_10 = compute_ctmc_rates(x_t, p_x1, t_tensor, rho_0=self.rho_0)
            f01 = torch.clamp(rate_01 * dt, max=1.0)
            f10 = torch.clamp(rate_10 * dt, max=1.0)
            flip01 = (torch.rand_like(f01) < f01) & (x_t < 0.5)
            flip10 = (torch.rand_like(f10) < f10) & (x_t > 0.5)
            x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
            x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
            x_t = symmetrize_binary(x_t) * contact_mask
            t_cum += dt

        x_final = project_to_valid_contact_map(
            x_t, p_last, contact_mask,
            min_score=threshold,
            use_sample_mask=False)
        return x_final, p_last

    @torch.no_grad()
    def predict(self, batch, **kwargs):
        """Alias compatible with evaluation interface."""
        return self.sample(batch, **kwargs)
