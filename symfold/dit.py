from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).contiguous()


class UnPatchify2D(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.deproj = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.deproj(x)


class AxialDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.row_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.col_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.row_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * mlp_ratio * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada_row = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.ada_col = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.ada_ffn = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        for mod in (self.ada_row[-1], self.ada_col[-1], self.ada_ffn[-1]):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)

    @staticmethod
    def _modulate(x: torch.Tensor, cond_proj: torch.Tensor) -> torch.Tensor:
        shift, scale = cond_proj.chunk(2, dim=-1)
        return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz, rows, cols, dim = x.shape

        y = self._modulate(self.row_norm(x), self.ada_row(cond))
        y = y.reshape(bsz * rows, cols, dim)
        y, _ = self.row_attn(y, y, y, need_weights=False)
        y = y.reshape(bsz, rows, cols, dim)
        x = x + y

        y = self._modulate(self.col_norm(x), self.ada_col(cond))
        y = y.permute(0, 2, 1, 3).reshape(bsz * cols, rows, dim)
        y, _ = self.col_attn(y, y, y, need_weights=False)
        y = y.reshape(bsz, cols, rows, dim).permute(0, 2, 1, 3)
        x = x + y

        y = self._modulate(self.ffn_norm(x), self.ada_ffn(cond))
        x = x + self.ffn(y)
        return x


class OutputRefineConv(nn.Module):
    def __init__(self, mid_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        return logit + self.net(logit)


class AxialDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        patch_size: int = 4,
        dropout: float = 0.1,
        output_refine: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed2D(in_channels, hidden_dim, patch_size)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            AxialDiTBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.unpatch = UnPatchify2D(hidden_dim, 1, patch_size)
        self.refine = OutputRefineConv() if output_refine else nn.Identity()
        self.density_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, t: torch.Tensor, contact_masks: torch.Tensor | None = None):
        tokens = self.patch_embed(features)
        cond = self.time_mlp(t)
        for block in self.blocks:
            tokens = block(tokens, cond)

        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        density = self.density_head(tokens.mean(dim=(1, 2)))
        logit = self.unpatch(tokens)
        logit = self.refine(logit)
        logit = 0.5 * (logit + logit.transpose(-2, -1))

        length = logit.shape[-1]
        idx = torch.arange(length, device=logit.device)
        short = (idx.view(length, 1) - idx.view(1, length)).abs() < 3
        logit = logit.masked_fill(short.view(1, 1, length, length), -10.0)
        if contact_masks is not None:
            logit = logit.masked_fill(contact_masks < 0.5, -10.0)
        return logit, density
