# -*- coding: utf-8 -*-
"""PriFold v8: DensityNet-Pro — Precision-Focused RNA Contact Prediction.

Improvements over v7 based on comprehensive analysis (2026-06-15):
  1. OHEM: Only hardest negatives contribute to BCE loss → FP gets punished
  2. FP Penalty: Explicit penalty on false positive positions
  3. Length-aware Budget: budget_fraction decays with length to prevent over-prediction
  4. BP Compatibility Mask: Filter non-canonical pairs (AU/GC/GU only)
  5. Shift-aware Soft Loss: Partial credit for predictions within ±1 of GT
  6. Increased dropout (0.1→0.2) + DropPath for generalization

Architecture (same backbone as v7):
  RNA seq → MARS-LX (frozen, 160M) → 1D hidden + 2D attention
          → Pair Feature Construction (outer product + attn proj)
          → Axial Transformer Stack (N layers, with DropPath)
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

class DropPath(nn.Module):
    """Stochastic depth / drop path."""
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


class AxialAttentionBlock(nn.Module):
    """Axial attention: row-attn + col-attn + FFN, with DropPath."""

    def __init__(self, dim: int, num_heads: int = 4, dim_head: int = 32,
                 ff_mult: int = 4, dropout: float = 0.2, drop_path: float = 0.0):
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

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.attn_dropout = dropout

        # Zero-init output projections for stable training
        nn.init.zeros_(self.row_out.weight)
        nn.init.zeros_(self.row_out.bias)
        nn.init.zeros_(self.col_out.weight)
        nn.init.zeros_(self.col_out.bias)

    def _axial_attn(self, x, qkv_proj, out_proj):
        B, N, S, D = x.shape
        h = self.num_heads
        qkv = qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n s (h d) -> (b n) h s d', h=h)
        k = rearrange(k, 'b n s (h d) -> (b n) h s d', h=h)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        dp = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=N)
        return out_proj(out)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        pair = pair + self.drop_path(self._axial_attn(self.row_norm(pair), self.row_qkv, self.row_out))
        x = self.col_norm(pair).transpose(1, 2)
        pair = pair + self.drop_path(self._axial_attn(x, self.col_qkv, self.col_out).transpose(1, 2))
        pair = pair + self.drop_path(self.ffn(self.ffn_norm(pair)))
        return pair


# ============================================================
# BP Compatibility
# ============================================================

# Canonical pairs: A-U, U-A, G-C, C-G, G-U, U-G
# Encoding: A=0, T/U=1, G=2, C=3
_COMPAT_MATRIX = torch.zeros(4, 4, dtype=torch.bool)
_COMPAT_MATRIX[0, 1] = True  # A-U
_COMPAT_MATRIX[1, 0] = True  # U-A
_COMPAT_MATRIX[2, 3] = True  # G-C
_COMPAT_MATRIX[3, 2] = True  # C-G
_COMPAT_MATRIX[2, 1] = True  # G-U
_COMPAT_MATRIX[1, 2] = True  # U-G


def build_bp_compat_mask(seq_oh: torch.Tensor) -> torch.Tensor:
    """Build base-pair compatibility mask from one-hot sequence.
    
    Args:
        seq_oh: (B, L, 4) one-hot encoded sequence
    Returns:
        compat_mask: (B, L, L) boolean mask, True where pairing is canonical
    """
    B, L, _ = seq_oh.shape
    device = seq_oh.device
    # Get base indices
    base_idx = seq_oh.argmax(dim=-1)  # (B, L)
    compat = _COMPAT_MATRIX.to(device)
    # Build pairwise compatibility
    idx_i = base_idx.unsqueeze(2).expand(-1, -1, L)  # (B, L, L)
    idx_j = base_idx.unsqueeze(1).expand(-1, L, -1)  # (B, L, L)
    mask = compat[idx_i, idx_j]  # (B, L, L)
    return mask


# ============================================================
# Main Model
# ============================================================

class DensityNetPro(nn.Module):
    """v8: Precision-focused discriminative contact predictor.

    Key differences from v7:
      - DropPath in transformer layers
      - Higher dropout (0.2 default)
      - OHEM loss
      - FP penalty loss
      - Shift-aware soft loss
      - BP compatibility in loss and inference
      - Length-aware budget at inference
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
                 hidden_dim: int = 160,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dim_head: int = 40,
                 ff_mult: int = 4,
                 dropout: float = 0.2,
                 drop_path: float = 0.1,
                 # === Loss config (all ablatable via config) ===
                 # 1. Focal BCE
                 focal_gamma: float = 1.0,
                 pos_weight_base: float = 99.0,
                 # 2. Dice
                 dice_weight: float = 0.5,
                 # 3. DST (Density-Stratified Tversky)
                 dst_weight: float = 0.4,
                 dst_low_threshold: float = 0.10,
                 dst_tversky_alpha: float = 0.7,
                 dst_tversky_beta: float = 0.3,
                 # 4. Pair count + ratio penalty
                 pair_count_weight: float = 0.3,
                 ratio_penalty_weight: float = 0.2,
                 ratio_penalty_threshold: float = 1.15,
                 # 5. Density head
                 density_loss_weight: float = 0.3,
                 # 6. OHEM (new in v8)
                 ohem_enabled: bool = True,
                 ohem_neg_ratio: int = 3,
                 # 7. FP penalty (new in v8)
                 fp_penalty_enabled: bool = True,
                 fp_penalty_weight: float = 3.0,
                 # 8. BP compatibility (new in v8)
                 bp_compat_enabled: bool = True,
                 bp_compat_weight: float = 0.5,
                 bp_compat_in_inference: bool = True,
                 # 9. Shift-aware soft loss (new in v8)
                 shift_loss_enabled: bool = True,
                 shift_loss_weight: float = 0.3,
                 shift_radius: int = 1,
                 ):
        super().__init__()
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.hidden_dim = hidden_dim

        # Store all loss config
        self.focal_gamma = focal_gamma
        self.pos_weight_base = pos_weight_base
        self.dice_weight = dice_weight
        self.dst_weight = dst_weight
        self.dst_low_threshold = dst_low_threshold
        self.dst_tversky_alpha = dst_tversky_alpha
        self.dst_tversky_beta = dst_tversky_beta
        self.pair_count_weight = pair_count_weight
        self.ratio_penalty_weight = ratio_penalty_weight
        self.ratio_penalty_threshold = ratio_penalty_threshold
        self.density_loss_weight = density_loss_weight
        self.ohem_enabled = ohem_enabled
        self.ohem_neg_ratio = ohem_neg_ratio
        self.fp_penalty_enabled = fp_penalty_enabled
        self.fp_penalty_weight = fp_penalty_weight
        self.bp_compat_enabled = bp_compat_enabled
        self.bp_compat_weight = bp_compat_weight
        self.bp_compat_in_inference = bp_compat_in_inference
        self.shift_loss_enabled = shift_loss_enabled
        self.shift_loss_weight = shift_loss_weight
        self.shift_radius = shift_radius

        # MARS extractor (frozen)
        self.extractor = extractor
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        # --- Feature Construction (same as v7) ---
        mars_attn_ch = mars_n_attn_layers * mars_n_heads  # 72
        self.mars_1d_proj = nn.Sequential(
            nn.Linear(mars_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.mars_2d_proj = nn.Sequential(
            nn.Conv2d(mars_attn_ch, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, hidden_dim // 4, 1),
        )
        self.seq_proj = nn.Linear(16, hidden_dim // 8)
        input_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 8 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Axial Transformer Stack (with DropPath) ---
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            AxialAttentionBlock(dim=hidden_dim, num_heads=num_heads,
                                dim_head=dim_head, ff_mult=ff_mult,
                                dropout=dropout, drop_path=dpr[i])
            for i in range(num_layers)
        ])

        # --- Output Heads ---
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    # ----------------------------------------------------------
    # MARS extraction (same as v7)
    # ----------------------------------------------------------

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
        B, L, _ = mars_hidden.shape
        proj_1d = self.mars_1d_proj(mars_hidden)
        pair_1d = torch.cat([
            proj_1d.unsqueeze(2).expand(-1, -1, L, -1),
            proj_1d.unsqueeze(1).expand(-1, L, -1, -1),
        ], dim=-1)
        attn_flat = mars_attn.reshape(B, -1, L, L)
        pair_2d = self.mars_2d_proj(attn_flat).permute(0, 2, 3, 1)
        seq_i = seq_oh.unsqueeze(2).expand(-1, -1, L, -1)
        seq_j = seq_oh.unsqueeze(1).expand(-1, L, -1, -1)
        seq_pair = (seq_i.unsqueeze(-1) * seq_j.unsqueeze(-2)).reshape(B, L, L, 16)
        seq_pair = self.seq_proj(seq_pair)
        pos = pos_bias.unsqueeze(-1)
        pair = torch.cat([pair_1d, pair_2d, seq_pair, pos], dim=-1)
        pair = self.input_proj(pair)
        return pair

    # ----------------------------------------------------------
    # Forward (training)
    # ----------------------------------------------------------

    def forward(self, batch: dict):
        contact = batch['contact']
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

        # Axial Transformer
        for layer in self.layers:
            pair = layer(pair)

        # Output
        pair = self.out_norm(pair)
        logit = self.contact_head(pair).permute(0, 3, 1, 2)
        logit = (logit + logit.transpose(2, 3)) / 2
        logit = logit * contact_mask

        # Density prediction
        valid = contact_mask.squeeze(1)
        pair_pooled = (pair * valid.unsqueeze(-1)).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).unsqueeze(-1).clamp(min=1)
        density_pred = self.density_head(pair_pooled)

        # BP compatibility mask for loss
        bp_mask = None
        if self.bp_compat_enabled and seq_oh is not None:
            bp_mask = build_bp_compat_mask(seq_oh)  # (B, L, L)

        # Compute loss
        loss, loss_dict = self._compute_loss(logit, contact, contact_mask, density_pred, bp_mask)
        return loss, loss_dict

    # ----------------------------------------------------------
    # Loss computation
    # ----------------------------------------------------------

    def _compute_loss(self, logit, contact, contact_mask, density_pred, bp_mask=None):
        B = logit.shape[0]
        valid = contact_mask
        pred = torch.sigmoid(logit)
        y = contact

        # GT density
        l_eff = valid.squeeze(1)[:, 0, :].sum(dim=-1)
        gt_pairs = (y.squeeze(1) * valid.squeeze(1)).sum(dim=(-1, -2)) / 2
        gt_density = gt_pairs / l_eff.clamp(min=1)

        loss_dict = {}

        # ====== 1. Focal BCE with OHEM ======
        bce_raw = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
        pt = torch.where(y > 0.5, pred, 1 - pred)
        focal = (1 - pt) ** self.focal_gamma
        bce_focal = bce_raw * focal * valid

        # Positive loss (with adaptive pos_weight)
        pos_w = (1.0 / gt_density.clamp(min=0.01)).clamp(max=self.pos_weight_base)
        pos_w = pos_w.view(B, 1, 1, 1)
        pos_mask = (y > 0.5) & (valid > 0.5)
        pos_loss = (bce_focal * pos_mask.float() * pos_w).sum() / pos_mask.float().sum().clamp(min=1)

        # Negative loss (with OHEM)
        neg_mask = (y < 0.5) & (valid > 0.5)
        if self.ohem_enabled:
            # Only keep top-k hardest negatives per sample
            neg_bce = bce_focal * neg_mask.float()  # (B, 1, L, L)
            num_pos = pos_mask.float().sum(dim=(-1, -2, -3)).clamp(min=1)  # (B,)
            k = (num_pos * self.ohem_neg_ratio).long().clamp(min=10)

            # Per-sample top-k
            neg_flat = neg_bce.view(B, -1)  # (B, L*L)
            total_neg = neg_mask.float().view(B, -1).sum(dim=1)
            neg_loss_sum = 0.0
            for i in range(B):
                ki = min(int(k[i].item()), int(total_neg[i].item()))
                if ki > 0:
                    topk_vals, _ = neg_flat[i].topk(ki)
                    neg_loss_sum = neg_loss_sum + topk_vals.sum()
            neg_loss = neg_loss_sum / k.float().sum().clamp(min=1)
        else:
            neg_loss = (bce_focal * neg_mask.float()).sum() / neg_mask.float().sum().clamp(min=1)

        bce_loss = pos_loss + neg_loss
        loss_dict['bce'] = bce_loss.detach()

        # ====== 2. Dice loss ======
        if self.dice_weight > 0:
            p_flat = (pred * valid).sum(dim=(-1, -2, -3))
            y_flat = (y * valid).sum(dim=(-1, -2, -3))
            inter = (pred * y * valid).sum(dim=(-1, -2, -3))
            dice = 1.0 - (2 * inter + 1) / (p_flat + y_flat + 1)
            dice_loss = dice.mean() * self.dice_weight
        else:
            dice_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['dice'] = dice_loss.detach()

        # ====== 3. DST (Density-Stratified Tversky) ======
        if self.dst_weight > 0:
            is_low = (gt_density < self.dst_low_threshold).float()
            tp = (pred * y * valid).sum(dim=(-1, -2, -3))
            fp = (pred * (1 - y) * valid).sum(dim=(-1, -2, -3))
            fn = ((1 - pred) * y * valid).sum(dim=(-1, -2, -3))
            alpha = is_low * self.dst_tversky_alpha + (1 - is_low) * 0.5
            beta = is_low * self.dst_tversky_beta + (1 - is_low) * 0.5
            tversky = tp / (tp + alpha * fp + beta * fn + 1.0)
            dst_loss = (1.0 - tversky).mean() * self.dst_weight
        else:
            dst_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['dst'] = dst_loss.detach()

        # ====== 4. Pair count + ratio penalty ======
        pred_density = (pred * valid).sum(dim=(-1, -2, -3)) / (2 * l_eff.clamp(min=1))
        pair_count_loss = F.smooth_l1_loss(pred_density, gt_density) * self.pair_count_weight
        ratio = pred_density / gt_density.clamp(min=0.01)
        ratio_loss = F.relu(ratio - self.ratio_penalty_threshold).mean() * self.ratio_penalty_weight
        loss_dict['pair_count'] = pair_count_loss.detach()
        loss_dict['ratio'] = ratio_loss.detach()

        # ====== 5. Density head loss ======
        density_loss = F.mse_loss(density_pred.squeeze(-1), gt_density) * self.density_loss_weight
        loss_dict['density'] = density_loss.detach()

        # ====== 6. FP Penalty (new in v8) ======
        if self.fp_penalty_enabled:
            fp_mask = (pred > 0.5) & (y < 0.5) & (valid > 0.5)
            fp_penalty = (pred * fp_mask.float()).sum() / fp_mask.float().sum().clamp(min=1)
            fp_loss = fp_penalty * self.fp_penalty_weight
        else:
            fp_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['fp_penalty'] = fp_loss.detach()

        # ====== 7. BP Compatibility loss (new in v8) ======
        if self.bp_compat_enabled and bp_mask is not None:
            # Penalize predictions on non-canonical positions
            non_compat = (~bp_mask).float().unsqueeze(1)  # (B, 1, L, L)
            bp_penalty = (pred * non_compat * valid).sum() / (non_compat * valid).sum().clamp(min=1)
            bp_loss = bp_penalty * self.bp_compat_weight
        else:
            bp_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['bp_compat'] = bp_loss.detach()

        # ====== 8. Shift-aware soft loss (new in v8) ======
        if self.shift_loss_enabled:
            # Create dilated GT: GT ± shift_radius considered "near-correct"
            y_bin = (y > 0.5).float()
            # Dilate GT with max_pool
            kernel = 2 * self.shift_radius + 1
            y_dilated = F.max_pool2d(y_bin, kernel_size=kernel, stride=1,
                                      padding=self.shift_radius)
            # Near-miss: pred=1, GT=0, but dilated_GT=1
            near_miss_mask = (pred > 0.5) & (y < 0.5) & (y_dilated > 0.5) & (valid > 0.5)
            # Give partial credit: reduce the penalty for near-miss FP
            # This is implemented as negative loss (reducing total loss for near-miss)
            near_miss_reward = (pred * near_miss_mask.float()).sum() / near_miss_mask.float().sum().clamp(min=1)
            shift_loss = -near_miss_reward * self.shift_loss_weight  # negative = reward
        else:
            shift_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['shift'] = shift_loss.detach()

        # ====== Total ======
        total = (bce_loss + dice_loss + dst_loss + pair_count_loss + ratio_loss +
                 density_loss + fp_loss + bp_loss + shift_loss)
        loss_dict['total'] = total.detach()

        return total, loss_dict

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict, budget_fraction: float = 0.30,
                use_density_budget: bool = True, score_threshold: float = 0.45,
                length_decay: float = 0.3):
        """Inference with length-aware budget and BP compatibility filtering."""
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

        # Density prediction
        valid = contact_mask.squeeze(1)
        pair_pooled = (pair * valid.unsqueeze(-1)).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).unsqueeze(-1).clamp(min=1)
        density_pred = self.density_head(pair_pooled)

        score = torch.sigmoid(logit)

        # BP compatibility mask
        bp_mask = None
        if self.bp_compat_in_inference and seq_oh is not None:
            bp_mask = build_bp_compat_mask(seq_oh)  # (B, L, L)

        # Length-aware budget
        l_eff = valid[:, 0, :].sum(dim=-1)
        if use_density_budget:
            # Density-guided with length decay: longer sequences get tighter budget
            length_factor = (100.0 / l_eff.clamp(min=50)) ** length_decay
            max_pairs = torch.round(density_pred.squeeze(-1) * l_eff * length_factor * 1.05).long()
        else:
            max_pairs = torch.round(l_eff * budget_fraction).long()
        max_pairs = max_pairs.clamp(min=2)

        # Greedy projection
        B, _, L, _ = score.shape
        pred_maps = []
        for i in range(B):
            s = score[i, 0]
            m = contact_mask[i, 0]
            upper = torch.triu(torch.ones(L, L, device=s.device, dtype=torch.bool), diagonal=3)
            candidates = s * m * upper.float()
            candidates[candidates < score_threshold] = 0

            # Apply BP compatibility filter
            if bp_mask is not None:
                candidates = candidates * bp_mask[i].float()

            # Top-k
            k = int(max_pairs[i].item())
            flat = candidates.view(-1)
            n_valid = (flat > 0).sum().item()
            topk_vals, topk_idx = flat.topk(min(k, n_valid))

            contact_map = torch.zeros(L, L, device=s.device)
            if topk_vals.numel() > 0:
                rows = topk_idx // L
                cols = topk_idx % L
                contact_map[rows, cols] = 1.0
                contact_map[cols, rows] = 1.0
            pred_maps.append(contact_map)

        pred = torch.stack(pred_maps).unsqueeze(1)
        return pred, score
