# -*- coding: utf-8 -*-
"""DA-SE-DiT-MARS v6 backbone.

Same architecture as v5, re-exported for version independence.
v6 changes are primarily in the loss (modular) and training loop (ablation).
The backbone architecture is identical to v5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from symfold.v3.da_se_dit import (
    SinusoidalTimeEmbedding,
    AxialRoPE,
    RMSNorm,
    TriangleMultiplicativeUpdate,
    GatedFFN,
    PatchEmbed2D,
    UnPatchify2D,
    OutputRefineConv,
    DensityHead,
    MultiLayerMarsFusion,
    MarsAttentionProj,
)

from symfold.v4.da_se_dit import (
    CondAttentionBias,
    ControlInjectMLP,
    DilatedAxialAttentionV4,
)


class DASEDiTBlockV6(nn.Module):
    """AdaLN-Zero × biased axial attention + triangle + SwiGLU.

    Identical to V5 block. Kept as separate class for version independence.
    """

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


class DASEDiT_MARS_v6(nn.Module):
    """v6 backbone — architecturally identical to v5.

    All config is exposed for ablation:
      - hidden_dim, num_layers, dilation_pattern, tri_start_layer, etc.
      - control_every: condition injection frequency (0 to disable)
      - use_direct_head: whether to output direct logit (for ablation)
      - use_density_head: whether to output density prediction
    """

    def __init__(self,
                 mars_dim: int = 1056,
                 n_attn_layers: int = 6,
                 n_heads_mars: int = 12,
                 hidden_dim: int = 320,
                 num_heads: int = 4,
                 dim_head: int = 80,
                 num_layers: int = 12,
                 patch_size: int = 4,
                 mars_emb_proj_dim: int = 32,
                 mars_attn_proj_dim: int = 16,
                 mars_hidden_fusion_dim: int = 64,
                 mars_hidden_layers: int = 4,
                 xt_emb_dim: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 dilation_pattern: list | None = None,
                 tri_start_layer: int = 4,
                 tri_dim: int = 64,
                 refine_mid_ch: int = 16,
                 use_seq_oh: bool = True,
                 cond_bias_zero_init: bool = True,
                 control_every: int = 3,
                 use_direct_head: bool = True,
                 use_density_head: bool = True):
        super().__init__()
        if dilation_pattern is None:
            dilation_pattern = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
        assert len(dilation_pattern) == num_layers, \
            f"dilation_pattern length {len(dilation_pattern)} != num_layers {num_layers}"
        self.patch_size = patch_size
        self.use_seq_oh = use_seq_oh
        self.control_every = int(control_every)
        self.use_direct_head = use_direct_head
        self.use_density_head = use_density_head

        # Input feature construction
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

        # Condition bias and control injection
        cond_channels = mars_attn_proj_dim + 1
        self.cond_bias = CondAttentionBias(cond_channels, num_heads, zero_init=cond_bias_zero_init)
        self.control_inject = ControlInjectMLP(cond_channels, hidden_dim) if control_every > 0 else None

        # Global conditioning
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

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DASEDiTBlockV6(
                hidden_dim, num_heads, dim_head, mlp_ratio, dropout,
                dilation=dilation_pattern[i],
                use_triangle=(i >= tri_start_layer),
                tri_dim=tri_dim)
            for i in range(num_layers)
        ])

        # Output heads
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # Flow head (always present)
        self.unpatch = UnPatchify2D(hidden_dim, 1, patch_size)
        self.refine = OutputRefineConv(mid_ch=refine_mid_ch)

        # Direct head (optional, for ablation)
        if use_direct_head:
            self.direct_unpatch = UnPatchify2D(hidden_dim, 1, patch_size)
            self.direct_refine = OutputRefineConv(mid_ch=refine_mid_ch)

        # Density head (optional, for ablation)
        if use_density_head:
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
            if self.control_every > 0 and (idx + 1) % self.control_every == 0 and self.control_inject is not None:
                tokens = tokens + self.control_inject(cond_patch)

        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)

        def expand(x):
            return x.view(x.shape[0], 1, 1, x.shape[-1])

        final_tokens = self.final_norm(tokens) * (1 + expand(sc)) + expand(sh)
        logit = self._mask_logits(self.refine(self.unpatch(final_tokens)), contact_masks)

        outs = [logit]
        if return_density and self.use_density_head:
            outs.append(self.density_head(cond))
        elif return_density:
            outs.append(None)
        if return_direct and self.use_direct_head:
            direct_logit = self._mask_logits(
                self.direct_refine(self.direct_unpatch(final_tokens)), contact_masks)
            outs.append(direct_logit)
        elif return_direct:
            outs.append(None)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)


if __name__ == '__main__':
    torch.manual_seed(0)
    b, l = 2, 32
    m = DASEDiT_MARS_v6(
        mars_dim=128, n_attn_layers=2, n_heads_mars=4,
        hidden_dim=64, num_heads=2, dim_head=16, num_layers=12,
        patch_size=4, dilation_pattern=[1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8],
        tri_start_layer=4, tri_dim=32).eval()
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
