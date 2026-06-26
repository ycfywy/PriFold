# -*- coding: utf-8 -*-
"""PriFold-SymFlow v12 — Generative RNA Folding via Discrete Flow Matching.

核心创新点: 用**生成式离散 Flow Matching** 做 RNA 二级结构预测
（区别于判别式的 v9/v10），并吸收 v9 验证有效的架构改进。

设计融合:
  - 生成式范式 (创新):  离散 Bernoulli Flow Matching (CTMC + τ-leap)   [来自 v6]
  - 位置编码:            2D RoPE (v9 消融 +11.9pp)                       [来自 v9]
  - 高效注意力:          Flash Attention (scaled_dot_product_attention)  [来自 v9]
  - 正则化:              DropPath(线性递增) + Dropout                    [来自 v9]
  - 碱基配对先验:        seq_pair 特征 (碱基 outer product)              [来自 v9]
  - 时间条件:            AdaLN-Zero (DiT, 生成式必需)                    [来自 DiT]
  - 损失:                模块化 (Focal BCE + Dice + 结构约束)            [来自 v6]

Pipeline:
  RNA seq → MARS-LX → pair features (1D outer + 2D attn + seq_pair)
         → + noisy x_t (additive injection)
         → FlowDiT blocks (axial attn + 2D RoPE + Flash + AdaLN-Zero on t)
         → contact logit  → τ-leap CTMC sampling → binary contact map
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """2D Rotary Position Embedding for axial attention (v9).

    对 row / col attention 分别施加 1D RoPE，使模型感知相对距离 |i-j|，
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
        """q, k: (B*N, H, S, D)."""
        cos = self.cos_cache[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class FlowDiTBlock(nn.Module):
    """Axial attention block: AdaLN-Zero(time) × (row+col Flash attn + RoPE) + FFN.

    融合点:
      - v9: axial attention + 2D RoPE + Flash Attention(SDPA) + DropPath
      - DiT: AdaLN-Zero 注入 flow 时间条件 t (生成式必需)
    """

    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 32,
                 cond_dim: int = 256, ff_mult: int = 4,
                 dropout: float = 0.2, drop_path: float = 0.0,
                 use_rope: bool = True, max_len: int = 512):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.use_rope = use_rope
        self.attn_dropout = dropout

        # AdaLN-Zero: 6 个调制参数 (attn: shift/scale/gate, ffn: shift/scale/gate)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        # Row attention
        self.row_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.row_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.row_out = nn.Linear(inner_dim, dim)
        # Col attention
        self.col_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.col_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.col_out = nn.Linear(inner_dim, dim)
        # Zero-init output projections (stable start)
        nn.init.zeros_(self.row_out.weight); nn.init.zeros_(self.row_out.bias)
        nn.init.zeros_(self.col_out.weight); nn.init.zeros_(self.col_out.bias)

        # FFN
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if use_rope:
            self.rope = RoPE2D(dim_head, max_len)

    def _axial_attn(self, x, qkv_proj, out_proj):
        """Multi-head axial self-attention with RoPE + Flash Attention (SDPA).

        x: (B, N, S, D) — attention along the last sequence axis S.
        """
        B, N, S, _ = x.shape
        h = self.num_heads
        qkv = qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n s (h d) -> (b n) h s d', h=h)
        k = rearrange(k, 'b n s (h d) -> (b n) h s d', h=h)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        if self.use_rope:
            q, k = self.rope.apply_rotary(q, k, S)
        dp = self.attn_dropout if self.training else 0.0
        # Flash Attention: 不存完整 O(S^2) 注意力矩阵 → 省显存/提速
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=N)
        return out_proj(out)

    def forward(self, pair: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """pair: (B, L, L, D); cond: (B, cond_dim)."""
        B, _, _, D = pair.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln(cond).chunk(6, dim=-1)

        def expand(p):
            return p.view(B, 1, 1, D)

        # --- Attention sub-block (row + col), modulated by time via AdaLN ---
        h = self.row_norm(pair) * (1 + expand(sc1)) + expand(sh1)
        h = self._axial_attn(h, self.row_qkv, self.row_out)
        pair = pair + self.drop_path(expand(g1) * h)

        h = (self.col_norm(pair) * (1 + expand(sc1)) + expand(sh1)).transpose(1, 2)
        h = self._axial_attn(h, self.col_qkv, self.col_out).transpose(1, 2)
        pair = pair + self.drop_path(expand(g1) * h)

        # --- FFN sub-block, modulated by time via AdaLN ---
        h = self.ffn_norm(pair) * (1 + expand(sc2)) + expand(sh2)
        h = self.ffn(h)
        pair = pair + self.drop_path(expand(g2) * h)
        return pair


# ============================================================
# Main Model: Discrete Flow Matching + FlowDiT
# ============================================================

class RNAFlowDiT(nn.Module):
    """Generative RNA Secondary Structure Prediction via Discrete Flow Matching.

    - Condition:  MARS pair features (1D outer + 2D attn + seq_pair)
    - Input:      noisy binary contact map x_t at flow time t (additive injection)
    - Backbone:   FlowDiT (axial attn + 2D RoPE + Flash Attention + AdaLN-Zero on t)
    - Output:     contact logit  p(x_1 = 1 | x_t, t)
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
                 use_seq_oh: bool = True,
                 max_len: int = 512,
                 use_gradient_checkpoint: bool = False,
                 # Discrete flow
                 rho_0: float = 0.005,
                 loss_config: dict | None = None,
                 ):
        super().__init__()
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.hidden_dim = hidden_dim
        self.use_seq_oh = use_seq_oh
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.rho_0 = rho_0

        # MARS extractor
        self.extractor = extractor
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        # --- Pair feature construction (对齐 v9) ---
        mars_attn_ch = mars_n_attn_layers * mars_n_heads
        self.mars_1d_proj = nn.Sequential(
            nn.Linear(mars_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.mars_2d_proj = nn.Sequential(
            nn.Conv2d(mars_attn_ch, 48, 1),
            nn.GELU(),
            nn.Conv2d(48, hidden_dim // 4, 1),
        )
        input_dim = hidden_dim + hidden_dim // 4  # pair_1d (concat i,j = hidden) + pair_2d
        if use_seq_oh:
            self.seq_proj = nn.Linear(16, hidden_dim // 8)
            input_dim += hidden_dim // 8
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noisy x_t additive injection (生成式输入)
        self.xt_proj = nn.Linear(1, hidden_dim)
        nn.init.zeros_(self.xt_proj.weight)
        nn.init.zeros_(self.xt_proj.bias)

        # Time conditioning
        cond_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmb(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FlowDiT blocks (linear DropPath schedule)
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            FlowDiTBlock(dim=hidden_dim, num_heads=num_heads, dim_head=dim_head,
                         cond_dim=cond_dim, ff_mult=ff_mult,
                         dropout=dropout, drop_path=dpr[i],
                         use_rope=use_rope, max_len=max_len)
            for i in range(num_layers)
        ])

        # Output head (time-conditioned via final AdaLN)
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

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

    def _build_mars_pair(self, mars_hidden, mars_attn, seq_oh):
        """Build static pair conditioning (不含 x_t). RoPE 提供位置信息, 无需 pos_bias.

        Returns: (B, L, L, hidden_dim)
        """
        B, L, _ = mars_hidden.shape
        proj_1d = self.mars_1d_proj(mars_hidden)  # (B, L, hidden//2)
        pair_1d = torch.cat([
            proj_1d.unsqueeze(2).expand(-1, -1, L, -1),
            proj_1d.unsqueeze(1).expand(-1, L, -1, -1),
        ], dim=-1)  # (B, L, L, hidden)

        attn_flat = mars_attn.reshape(B, -1, L, L)
        pair_2d = self.mars_2d_proj(attn_flat).permute(0, 2, 3, 1)  # (B, L, L, hidden//4)

        parts = [pair_1d, pair_2d]
        if self.use_seq_oh and seq_oh is not None:
            seq_i = seq_oh.unsqueeze(2).expand(-1, -1, L, -1)
            seq_j = seq_oh.unsqueeze(1).expand(-1, L, -1, -1)
            seq_pair = (seq_i.unsqueeze(-1) * seq_j.unsqueeze(-2)).reshape(B, L, L, 16)
            parts.append(self.seq_proj(seq_pair))  # (B, L, L, hidden//8)

        pair = torch.cat(parts, dim=-1)
        return self.input_proj(pair)  # (B, L, L, hidden)

    # ------------------------------------------------------------------
    # FlowDiT forward: predict logit p(x_1=1 | x_t, t)
    # ------------------------------------------------------------------

    def _dit_forward(self, base_pair: torch.Tensor, x_t: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        """
        base_pair: (B, L, L, D) — static MARS conditioning
        x_t:       (B, 1, L, L) — binary noisy contact map at time t
        t:         (B,)         — flow time
        Returns:   (B, 1, L, L) — symmetrized contact logit
        """
        B, _, L, _ = x_t.shape

        # Additive injection of noisy state x_t
        pair = base_pair + self.xt_proj(x_t.permute(0, 2, 3, 1))  # (B, L, L, D)

        cond = self.time_embed(t)  # (B, cond_dim)

        for block in self.blocks:
            if self.use_gradient_checkpoint and self.training:
                pair = torch.utils.checkpoint.checkpoint(
                    block, pair, cond, use_reentrant=False)
            else:
                pair = block(pair, cond)

        # Time-conditioned output head
        sh, sc = self.final_adaln(cond).chunk(2, dim=-1)
        pair = self.final_norm(pair) * (1 + sc.view(B, 1, 1, -1)) + sh.view(B, 1, 1, -1)
        logit = self.contact_head(pair).permute(0, 3, 1, 2)  # (B, 1, L, L)
        logit = 0.5 * (logit + logit.transpose(-2, -1))       # symmetrize
        return logit

    # ------------------------------------------------------------------
    # Training: Discrete (Bernoulli) Flow Matching
    # ------------------------------------------------------------------

    def forward(self, batch: dict):
        """Training forward pass.

        x_1 = symmetrize(GT contact)            (binary)
        x_t ~ Bernoulli( t·x_1 + (1-t)·ρ_0 )    (noising)
        logit = FlowDiT(x_t, t, cond)           → p(x_1=1|x_t,t)
        loss = ModularFlowLoss(logit, x_1, t)
        """
        contact = batch['contact']            # (B, 1, L, L)
        contact_mask = batch['contact_mask']  # (B, 1, L, L)
        seq_oh = batch.get('seq_oh')
        x_1 = symmetrize_binary(contact) * contact_mask

        B = x_1.shape[0]
        L = x_1.shape[-1]
        device = x_1.device

        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], L)
        base_pair = self._build_mars_pair(mars_hidden, mars_attn, seq_oh)

        # Sample time + Bernoulli noising
        t = torch.rand(B, device=device)
        x_t = sample_x_t_given_x_1(x_1, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_mask

        # Prediction (logit)
        logit = self._dit_forward(base_pair, x_t, t)  # (B, 1, L, L)

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
        seq_oh = batch.get('seq_oh')
        device = contact_mask.device
        B, _, L, _ = contact_mask.shape

        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], L)
        base_pair = self._build_mars_pair(mars_hidden, mars_attn, seq_oh)  # 仅算一次

        x_t = (torch.rand(B, 1, L, L, device=device) < self.rho_0).float()
        x_t = symmetrize_binary(x_t) * contact_mask

        raw = [math.sin(math.pi * (k + 0.5) / (2 * num_steps)) for k in range(num_steps)]
        total_raw = sum(raw)
        dt_list = [r / total_raw for r in raw]

        p_last = torch.zeros_like(x_t)
        t_cum = 0.0
        for dt in dt_list:
            t_tensor = torch.full((B,), t_cum, device=device)
            logit = symmetrize_logit(self._dit_forward(base_pair, x_t, t_tensor))
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

        x_final = project_to_valid_contact_map(x_t, p_last, contact_mask)
        return x_final, p_last

    @torch.no_grad()
    def predict(self, batch, **kwargs):
        """Alias compatible with evaluation interface."""
        return self.sample(batch, **kwargs)
