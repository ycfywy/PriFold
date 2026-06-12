# -*- coding: utf-8 -*-
"""PriFold v7: DensityNet — Density-Aware Discriminative RNA Contact Prediction.

Pure discriminative model. No flow, no sampling, no noise.
MARS-LX → Axial Transformer → Contact Map.

Key innovations:
  1. Density-Aware Loss: per-sample adaptive loss based on predicted density
  2. Multi-Scale Axial Attention: dilated axial attention at multiple resolutions
  3. Pair-Aware Density Head: predicts density to guide projection at inference

Architecture:
  RNA seq → MARS-LX (frozen, 160M) → 1D hidden + 2D attention
          → Pair Feature Construction (outer product + attn proj)
          → Axial Transformer Stack (N layers)
          → Contact Logit + Density Prediction
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from prifold.llama2_with_attn import mars_forward_with_attn


# ============================================================
# Building Blocks
# ============================================================

class AxialAttentionBlock(nn.Module):
    """Axial attention: row-attn + col-attn + FFN, with optional dilation."""

    def __init__(self, dim: int, num_heads: int = 4, dim_head: int = 32,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.row_norm = nn.LayerNorm(dim)
        self.row_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.row_out = nn.Linear(inner_dim, dim)

        self.col_norm = nn.LayerNorm(dim)
        self.col_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.col_out = nn.Linear(inner_dim, dim)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        # Zero-init output projections
        nn.init.zeros_(self.row_out.weight)
        nn.init.zeros_(self.row_out.bias)
        nn.init.zeros_(self.col_out.weight)
        nn.init.zeros_(self.col_out.bias)

    def _axial_attn(self, x, qkv_proj, out_proj):
        """x: (B, N, S, D) → attend along S for each N. Uses PyTorch SDPA for efficiency."""
        B, N, S, D = x.shape
        h = self.num_heads
        qkv = qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n s (h d) -> (b n) h s d', h=h)
        k = rearrange(k, 'b n s (h d) -> (b n) h s d', h=h)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        # Use PyTorch's optimized SDPA (flash attention / memory efficient)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else 0.0)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=N)
        return out_proj(out)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """pair: (B, L, L, D) → (B, L, L, D)"""
        # Row attention
        pair = pair + self._axial_attn(self.row_norm(pair), self.row_qkv, self.row_out)
        # Column attention
        x = self.col_norm(pair).transpose(1, 2)
        pair = pair + self._axial_attn(x, self.col_qkv, self.col_out).transpose(1, 2)
        # FFN
        pair = pair + self.ffn(self.ffn_norm(pair))
        return pair


# ============================================================
# Main Model
# ============================================================

class DensityNet(nn.Module):
    """Pure discriminative contact map predictor with density awareness.

    Trainable params: ~5-8M (depending on config).
    Inference: single forward pass, no sampling.
    """

    def __init__(self,
                 extractor: nn.Module,
                 # MARS config
                 freeze_mars: bool = True,
                 mars_dim: int = 1056,
                 mars_n_attn_layers: int = 6,
                 mars_n_heads: int = 12,
                 mars_hidden_layer_indices: list | None = None,
                 # Model config
                 hidden_dim: int = 128,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dim_head: int = 32,
                 ff_mult: int = 4,
                 dropout: float = 0.1,
                 # Density-aware config
                 density_loss_weight: float = 0.3,
                 dst_low_threshold: float = 0.18,
                 dst_tversky_alpha: float = 0.7,
                 dst_tversky_beta: float = 0.3,
                 dst_weight: float = 0.4,
                 # Loss config
                 focal_gamma: float = 1.0,
                 pos_weight_base: float = 99.0,
                 dice_weight: float = 0.5,
                 pair_count_weight: float = 0.3,
                 ratio_penalty_weight: float = 0.2,
                 ratio_penalty_threshold: float = 1.2,
                 ):
        super().__init__()
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.hidden_dim = hidden_dim

        # Loss hyperparams
        self.density_loss_weight = density_loss_weight
        self.dst_low_threshold = dst_low_threshold
        self.dst_tversky_alpha = dst_tversky_alpha
        self.dst_tversky_beta = dst_tversky_beta
        self.dst_weight = dst_weight
        self.focal_gamma = focal_gamma
        self.pos_weight_base = pos_weight_base
        self.dice_weight = dice_weight
        self.pair_count_weight = pair_count_weight
        self.ratio_penalty_weight = ratio_penalty_weight
        self.ratio_penalty_threshold = ratio_penalty_threshold

        # MARS extractor (frozen)
        self.extractor = extractor
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        # --- Feature Construction ---
        mars_attn_ch = mars_n_attn_layers * mars_n_heads  # 72
        # 1D → pair (outer concat)
        self.mars_1d_proj = nn.Sequential(
            nn.Linear(mars_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        # 2D attention → pair
        self.mars_2d_proj = nn.Sequential(
            nn.Conv2d(mars_attn_ch, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, hidden_dim // 4, 1),
        )
        # Sequence pair (4×4=16 outer)
        self.seq_proj = nn.Linear(16, hidden_dim // 8)
        # Pos bias (1 channel)
        # Total input: hidden_dim + hidden_dim//4 + hidden_dim//8 + 1
        input_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 8 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Axial Transformer Stack ---
        self.layers = nn.ModuleList([
            AxialAttentionBlock(dim=hidden_dim, num_heads=num_heads,
                                dim_head=dim_head, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_layers)
        ])

        # --- Output Heads ---
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Density head: global pool → density prediction
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _extract_mars(self, input_ids, attention_mask, set_len):
        if self.freeze_mars:
            self.extractor.eval()
            with torch.no_grad():
                hidden, attn_stack, hidden_layers = mars_forward_with_attn(
                    self.extractor, input_ids, attention_mask,
                    n_attn_layers=self.mars_n_attn_layers,
                    hidden_layer_indices=self.mars_hidden_layer_indices,
                    return_hidden_layers=True)
        else:
            hidden, attn_stack, hidden_layers = mars_forward_with_attn(
                self.extractor, input_ids, attention_mask,
                n_attn_layers=self.mars_n_attn_layers,
                hidden_layer_indices=self.mars_hidden_layer_indices,
                return_hidden_layers=True)

        base_len = input_ids.shape[1] - 2
        h = hidden[:, 1:1+base_len, :]
        a = attn_stack[:, :, :, 1:1+base_len, 1:1+base_len]

        b, cur, d = h.shape
        if cur < set_len:
            pad = set_len - cur
            h = F.pad(h, (0, 0, 0, pad))
            nl, nh = a.shape[1], a.shape[2]
            a_new = torch.zeros(b, nl, nh, set_len, set_len, device=a.device, dtype=a.dtype)
            a_new[:, :, :, :cur, :cur] = a
            a = a_new
        elif cur > set_len:
            h = h[:, :set_len, :]
            a = a[:, :, :, :set_len, :set_len]
        return h, a

    def _build_pair_features(self, mars_hidden, mars_attn, seq_oh, pos_bias):
        """Construct (B, L, L, D) pair representation from MARS features."""
        B, L, _ = mars_hidden.shape

        # 1D outer concat
        proj_1d = self.mars_1d_proj(mars_hidden)  # (B, L, D//2)
        pair_1d = torch.cat([
            proj_1d.unsqueeze(2).expand(-1, -1, L, -1),
            proj_1d.unsqueeze(1).expand(-1, L, -1, -1),
        ], dim=-1)  # (B, L, L, D)

        # 2D attention
        attn_flat = mars_attn.reshape(B, -1, L, L)  # (B, 72, L, L)
        pair_2d = self.mars_2d_proj(attn_flat)  # (B, D//4, L, L)
        pair_2d = pair_2d.permute(0, 2, 3, 1)  # (B, L, L, D//4)

        # Sequence outer product
        seq_i = seq_oh.unsqueeze(2).expand(-1, -1, L, -1)
        seq_j = seq_oh.unsqueeze(1).expand(-1, L, -1, -1)
        seq_pair = (seq_i.unsqueeze(-1) * seq_j.unsqueeze(-2)).reshape(B, L, L, 16)
        seq_pair = self.seq_proj(seq_pair)  # (B, L, L, D//8)

        # Pos bias
        pos = pos_bias.unsqueeze(-1)  # (B, L, L, 1)

        # Combine
        pair = torch.cat([pair_1d, pair_2d, seq_pair, pos], dim=-1)
        pair = self.input_proj(pair)  # (B, L, L, D)
        return pair

    def forward(self, batch: dict):
        """Training forward: single pass, fast."""
        contact = batch['contact']  # (B, 1, L, L)
        contact_mask = batch['contact_mask']
        seq_oh = batch.get('seq_oh')
        pos_bias = batch['pos_bias']
        set_len = contact.shape[-1]

        # Symmetrize GT
        contact = ((contact + contact.transpose(2, 3)) > 0).float() * contact_mask

        # Extract MARS
        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], set_len)

        # Build pair features
        pair = self._build_pair_features(mars_hidden, mars_attn, seq_oh, pos_bias)

        # Axial Transformer (no checkpointing — GPU memory is abundant)
        for layer in self.layers:
            pair = layer(pair)

        # Output
        pair = self.out_norm(pair)
        logit = self.contact_head(pair).permute(0, 3, 1, 2)  # (B, 1, L, L)
        logit = (logit + logit.transpose(2, 3)) / 2  # symmetrize
        logit = logit * contact_mask

        # Density prediction
        valid = contact_mask.squeeze(1)  # (B, L, L)
        pair_pooled = (pair * valid.unsqueeze(-1)).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).unsqueeze(-1).clamp(min=1)
        density_pred = self.density_head(pair_pooled)  # (B, 1)

        # Compute loss
        loss, loss_dict = self._compute_loss(logit, contact, contact_mask, density_pred)
        return loss, loss_dict

    def _compute_loss(self, logit, contact, contact_mask, density_pred):
        """Density-aware multi-component loss."""
        B = logit.shape[0]
        valid = contact_mask
        pred = torch.sigmoid(logit)
        y = contact

        # GT density
        l_eff = valid.squeeze(1)[:, 0, :].sum(dim=-1)  # (B,)
        gt_pairs = (y.squeeze(1) * valid.squeeze(1)).sum(dim=(-1, -2)) / 2
        gt_density = gt_pairs / l_eff.clamp(min=1)  # (B,)

        # --- BCE with adaptive pos_weight + focal ---
        pos_w = (1.0 / gt_density.clamp(min=0.01)).clamp(max=self.pos_weight_base)
        pos_w = pos_w.view(B, 1, 1, 1)
        bce_raw = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
        # Focal modulation
        pt = torch.where(y > 0.5, pred, 1 - pred)
        focal = (1 - pt) ** self.focal_gamma
        bce = (bce_raw * focal * valid).sum() / valid.sum().clamp(min=1)
        # Apply pos_weight manually to positives only
        pos_bce = (bce_raw * focal * y * valid * pos_w).sum() / (y * valid).sum().clamp(min=1)
        neg_bce = (bce_raw * focal * (1 - y) * valid).sum() / ((1 - y) * valid).sum().clamp(min=1)
        bce_loss = pos_bce + neg_bce

        # --- Dice loss ---
        p_flat = (pred * valid).sum(dim=(-1, -2, -3))
        y_flat = (y * valid).sum(dim=(-1, -2, -3))
        inter = (pred * y * valid).sum(dim=(-1, -2, -3))
        dice = 1.0 - (2 * inter + 1) / (p_flat + y_flat + 1)
        dice_loss = dice.mean() * self.dice_weight

        # --- Density-Stratified Tversky (DST) ---
        is_low = (gt_density < self.dst_low_threshold).float()
        tp = (pred * y * valid).sum(dim=(-1, -2, -3))
        fp = (pred * (1 - y) * valid).sum(dim=(-1, -2, -3))
        fn = ((1 - pred) * y * valid).sum(dim=(-1, -2, -3))
        alpha = is_low * self.dst_tversky_alpha + (1 - is_low) * 0.5
        beta = is_low * self.dst_tversky_beta + (1 - is_low) * 0.5
        tversky = tp / (tp + alpha * fp + beta * fn + 1.0)
        dst_loss = (1.0 - tversky).mean() * self.dst_weight

        # --- Pair count loss ---
        pred_density = (pred * valid).sum(dim=(-1, -2, -3)) / (2 * l_eff.clamp(min=1))
        pair_count_loss = F.smooth_l1_loss(pred_density, gt_density) * self.pair_count_weight

        # --- Ratio penalty ---
        ratio = pred_density / gt_density.clamp(min=0.01)
        penalty = F.relu(ratio - self.ratio_penalty_threshold)
        ratio_loss = penalty.mean() * self.ratio_penalty_weight

        # --- Density head loss ---
        density_loss = F.mse_loss(density_pred.squeeze(-1), gt_density) * self.density_loss_weight

        # Total
        total = bce_loss + dice_loss + dst_loss + pair_count_loss + ratio_loss + density_loss

        loss_dict = {
            'bce': bce_loss.detach(),
            'dice': dice_loss.detach(),
            'dst': dst_loss.detach(),
            'pair_count': pair_count_loss.detach(),
            'ratio': ratio_loss.detach(),
            'density': density_loss.detach(),
            'total': total.detach(),
        }
        return total, loss_dict

    @torch.no_grad()
    def predict(self, batch: dict, budget_fraction: float = 0.30,
                use_density_budget: bool = True, score_threshold: float = 0.4):
        """Inference: single forward pass → contact map."""
        contact_mask = batch['contact_mask']
        seq_oh = batch.get('seq_oh')
        pos_bias = batch['pos_bias']
        set_len = contact_mask.shape[-1]

        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], set_len)

        pair = self._build_pair_features(mars_hidden, mars_attn, seq_oh, pos_bias)
        for layer in self.layers:
            pair = layer(pair)

        pair = self.out_norm(pair)
        logit = self.contact_head(pair).permute(0, 3, 1, 2)
        logit = (logit + logit.transpose(2, 3)) / 2
        logit = logit * contact_mask

        # Density prediction for budget
        valid = contact_mask.squeeze(1)
        pair_pooled = (pair * valid.unsqueeze(-1)).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).unsqueeze(-1).clamp(min=1)
        density_pred = self.density_head(pair_pooled)  # (B, 1)

        # Score
        score = torch.sigmoid(logit)

        # Budget: density-guided or fixed
        l_eff = valid[:, 0, :].sum(dim=-1)
        if use_density_budget:
            max_pairs = torch.round(density_pred.squeeze(-1) * l_eff * 1.1).long()
        else:
            max_pairs = torch.round(l_eff * budget_fraction).long()
        max_pairs = max_pairs.clamp(min=2)

        # Greedy projection: take top-k pairs above threshold
        B, _, L, _ = score.shape
        pred_maps = []
        for i in range(B):
            s = score[i, 0]  # (L, L)
            m = contact_mask[i, 0]  # (L, L)
            # Upper triangle + min distance
            idx = torch.arange(L, device=s.device)
            upper = torch.triu(torch.ones(L, L, device=s.device, dtype=torch.bool), diagonal=3)
            candidates = s * m * upper.float()
            candidates[candidates < score_threshold] = 0

            # Top-k
            k = int(max_pairs[i].item())
            flat = candidates.view(-1)
            topk_vals, topk_idx = flat.topk(min(k, (flat > 0).sum().item()))

            contact_map = torch.zeros(L, L, device=s.device)
            if topk_vals.numel() > 0:
                rows = topk_idx // L
                cols = topk_idx % L
                contact_map[rows, cols] = 1.0
                contact_map[cols, rows] = 1.0  # symmetrize
            pred_maps.append(contact_map)

        pred = torch.stack(pred_maps).unsqueeze(1)  # (B, 1, L, L)
        return pred, score
