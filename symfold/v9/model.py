# -*- coding: utf-8 -*-
"""PriFold v9: DensityNet-Pro+ — 基于 v8 分析的全面改进。

v9 改进清单（用户指定）:
  P1. 降低 DST threshold — 对更低密度区间更激进惩罚
  P2. Shift-aware Loss (margin) — 偏移1位的 FP 惩罚小于完全错误
  P3. 增强正则化 — 更大 Dropout/DropPath、更强数据增强
  P4. 非标准配对处理 — 允许非标准碱基配对（去掉 BP compat 惩罚）
  P5. 长距离配对建模 — 引入 2D RoPE 相对位置编码

Architecture:
  RNA seq → MARS-LX (frozen, 160M) → 1D hidden + 2D attention
          → Pair Feature Construction (outer prod + attn + seq_pair)
          → 2D RoPE Positional Encoding  ← [P5] NEW
          → Axial Transformer Stack (8 layers, DropPath=0.15, Dropout=0.2)
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


# ============================================================
# [P5] 2D Rotary Position Embedding (RoPE)
# ============================================================

class RotaryPositionEmbedding2D(nn.Module):
    """2D RoPE for axial attention in pairwise contact maps.
    
    对 row attention 和 col attention 分别施加 1D RoPE，
    使模型能学到"位置 i 和位置 j 的相对距离"信息，
    有助于长距离配对建模。
    """
    def __init__(self, dim_head: int, max_len: int = 512):
        super().__init__()
        self.dim_head = dim_head
        # 频率基底：θ_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)
        # 预计算 sin/cos cache
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, dim_head//2)
        # cos, sin: (max_len, dim_head)
        cos_cache = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin_cache = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)
    
    def _rotate_half(self, x):
        """Rotate half: [x1, x2] → [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary(self, q, k, seq_len: int):
        """Apply RoPE to q, k tensors.
        
        q, k: (B*N, H, S, D) where S is the sequence dimension
        """
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class AxialAttentionBlock(nn.Module):
    """Axial attention with 2D RoPE: row-attn + col-attn + FFN.
    
    [P5] RoPE 让 attention 感知相对位置，提升长距离配对能力。
    [P3] 更大的 dropout 和 drop_path 增强正则化。
    """

    def __init__(self, dim: int, num_heads: int = 6, dim_head: int = 32,
                 ff_mult: int = 4, dropout: float = 0.2, drop_path: float = 0.0,
                 use_rope: bool = True, max_len: int = 512):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.use_rope = use_rope

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

        # [P5] RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding2D(dim_head, max_len)
        
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
        
        # [P5] Apply RoPE
        if self.use_rope:
            q, k = self.rope.apply_rotary(q, k, S)
        
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
# Main Model
# ============================================================

class DensityNetProPlus(nn.Module):
    """v9: 基于用户需求全面改进的 RNA contact predictor。

    改进清单:
      P1. DST threshold 降低 (0.10→0.05)，对更多样本施加低密度保护
      P2. Shift-aware margin loss：shift±1 只受 0.3x 惩罚，shift±2 受 0.6x
      P3. 正则化增强：dropout=0.2, drop_path=0.15, 更强数据增强
      P4. 去掉 BP compat 惩罚（允许非标准配对）
      P5. 2D RoPE 位置编码，提升长距离建模
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
                 hidden_dim: int = 192,
                 num_layers: int = 8,
                 num_heads: int = 6,
                 dim_head: int = 32,
                 ff_mult: int = 4,
                 dropout: float = 0.2,         # [P3] 0.15→0.2
                 drop_path: float = 0.15,      # [P3] 0.1→0.15
                 use_rope: bool = True,        # [P5]
                 # Loss config
                 focal_gamma: float = 1.0,
                 pos_weight_base: float = 99.0,
                 dice_weight: float = 0.5,
                 dst_weight: float = 0.5,      # [P1] 0.4→0.5 加强
                 dst_low_threshold: float = 0.05,  # [P1] 0.10→0.05 更激进
                 dst_tversky_alpha: float = 0.7,
                 dst_tversky_beta: float = 0.3,
                 pair_count_weight: float = 0.3,
                 ratio_penalty_weight: float = 0.2,
                 ratio_penalty_threshold: float = 1.20,
                 density_loss_weight: float = 0.3,
                 ohem_enabled: bool = True,
                 ohem_neg_ratio: int = 3,
                 fp_penalty_enabled: bool = True,
                 fp_penalty_weight: float = 2.0,
                 # [P4] BP compat 默认关闭（允许非标准配对）
                 bp_compat_enabled: bool = False,
                 bp_compat_weight: float = 0.0,
                 bp_compat_in_inference: bool = False,
                 # [P2] Shift-aware margin loss
                 shift_loss_enabled: bool = True,
                 shift_loss_weight: float = 0.8,
                 shift_radius: int = 2,
                 ):
        super().__init__()
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.hidden_dim = hidden_dim
        self.use_rope = use_rope

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

        # --- Feature Construction ---
        mars_attn_ch = mars_n_attn_layers * mars_n_heads  # 72
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
        self.seq_proj = nn.Linear(16, hidden_dim // 8)
        input_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 8  # [P5] 去掉 pos_bias 通道，用 RoPE 替代
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Axial Transformer Stack with RoPE [P5] + stronger DropPath [P3] ---
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            AxialAttentionBlock(dim=hidden_dim, num_heads=num_heads,
                                dim_head=dim_head, ff_mult=ff_mult,
                                dropout=dropout, drop_path=dpr[i],
                                use_rope=use_rope, max_len=512)
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
    # MARS extraction
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

    def _build_pair_features(self, mars_hidden, mars_attn, seq_oh):
        """Build pair features. [P5] 不再需要 pos_bias，RoPE 提供位置信息。"""
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
        # [P5] 不拼接 pos_bias，位置信息由 RoPE 在 attention 中注入
        pair = torch.cat([pair_1d, pair_2d, seq_pair], dim=-1)
        pair = self.input_proj(pair)
        return pair

    # ----------------------------------------------------------
    # Forward (training)
    # ----------------------------------------------------------

    def forward(self, batch: dict):
        contact = batch['contact']
        contact_mask = batch['contact_mask']
        seq_oh = batch.get('seq_oh')
        set_len = contact.shape[-1]

        # Symmetrize GT
        contact = ((contact + contact.transpose(2, 3)) > 0).float() * contact_mask

        # Extract MARS
        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], set_len)

        # Build pair features [P5] 不需要 pos_bias
        pair = self._build_pair_features(mars_hidden, mars_attn, seq_oh)

        # Axial Transformer with RoPE
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

        # Compute loss  [P4] 不使用 BP compat
        loss, loss_dict = self._compute_loss(logit, contact, contact_mask, density_pred)
        return loss, loss_dict

    # ----------------------------------------------------------
    # Loss computation
    # ----------------------------------------------------------

    def _compute_loss(self, logit, contact, contact_mask, density_pred):
        B = logit.shape[0]
        valid = contact_mask
        pred = torch.sigmoid(logit)
        y = contact

        # GT density
        l_eff = valid.squeeze(1)[:, 0, :].sum(dim=-1)
        gt_pairs = (y.squeeze(1) * valid.squeeze(1)).sum(dim=(-1, -2)) / 2
        gt_density = gt_pairs / l_eff.clamp(min=1)

        loss_dict = {}

        # ====== 1. Focal BCE with Vectorized OHEM ======
        bce_raw = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
        pt = torch.where(y > 0.5, pred, 1 - pred)
        focal = (1 - pt) ** self.focal_gamma
        bce_focal = bce_raw * focal * valid

        # Positive loss
        pos_w = (1.0 / gt_density.clamp(min=0.01)).clamp(max=self.pos_weight_base)
        pos_w = pos_w.view(B, 1, 1, 1)
        pos_mask = (y > 0.5) & (valid > 0.5)
        pos_loss = (bce_focal * pos_mask.float() * pos_w).sum() / pos_mask.float().sum().clamp(min=1)

        # Negative loss with VECTORIZED OHEM
        neg_mask = (y < 0.5) & (valid > 0.5)
        if self.ohem_enabled:
            neg_bce = bce_focal * neg_mask.float()
            num_pos = pos_mask.float().view(B, -1).sum(dim=1).clamp(min=1)
            k = (num_pos * self.ohem_neg_ratio).long().clamp(min=10)
            max_k = int(k.max().item())
            neg_flat = neg_bce.view(B, -1)
            topk_vals, _ = neg_flat.topk(min(max_k, neg_flat.shape[1]), dim=1)
            indices = torch.arange(topk_vals.shape[1], device=topk_vals.device).unsqueeze(0)
            valid_topk = indices < k.unsqueeze(1)
            neg_loss = (topk_vals * valid_topk.float()).sum() / k.float().sum()
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

        # ====== 3. [P1] DST — 更低 threshold (0.05), 更强权重 (0.5) ======
        if self.dst_weight > 0:
            is_low = (gt_density < self.dst_low_threshold).float()  # [P1] threshold=0.05
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

        # ====== 6. FP Penalty ======
        if self.fp_penalty_enabled:
            fp_mask = (pred > 0.5) & (y < 0.5) & (valid > 0.5)
            fp_penalty = (pred * fp_mask.float()).sum() / fp_mask.float().sum().clamp(min=1)
            fp_loss = fp_penalty * self.fp_penalty_weight
        else:
            fp_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['fp_penalty'] = fp_loss.detach()

        # ====== 7. [P2] Shift-aware MARGIN Loss ======
        # 核心思想：FP 如果在 GT±1 范围内，loss 只算 0.3 倍；±2 范围内算 0.6 倍
        # 相比旧版的"奖励"设计，这里改为"分层惩罚"——更直觉
        if self.shift_loss_enabled:
            y_bin = (y > 0.5).float()
            
            # 构建 distance-to-GT map：每个 FP 位置距最近 GT 的曼哈顿距离
            # 用 max_pool 近似：dilate GT by radius 1 和 radius 2
            y_dilated_1 = F.max_pool2d(y_bin, kernel_size=3, stride=1, padding=1)
            y_dilated_2 = F.max_pool2d(y_bin, kernel_size=5, stride=1, padding=2)
            
            fp_positions = (pred > 0.5) & (y < 0.5) & (valid > 0.5)
            
            # 分层：shift±1 = 在 dilated_1 内但不在 GT 内
            near_1 = fp_positions & (y_dilated_1 > 0.5)   # 距 GT ≤ 1
            near_2 = fp_positions & (y_dilated_2 > 0.5) & (y_dilated_1 < 0.5)  # 距 GT = 2
            far_fp = fp_positions & (y_dilated_2 < 0.5)    # 距 GT > 2
            
            # BCE loss 对不同层级的 FP 施加不同权重
            fp_bce = bce_raw * valid  # 原始 BCE（未加 focal）
            
            # near_1: 只承受 0.3 倍惩罚（大幅减轻）
            # near_2: 承受 0.6 倍惩罚
            # far_fp: 承受 1.0 倍惩罚（正常）
            # 实现方式：计算应该被"减免"的 loss，作为负项
            relief_1 = (fp_bce * near_1.float() * 0.7).sum()  # 减免 70%
            relief_2 = (fp_bce * near_2.float() * 0.4).sum()  # 减免 40%
            total_fp_bce = (fp_bce * fp_positions.float()).sum().clamp(min=1)
            
            shift_loss = -(relief_1 + relief_2) / total_fp_bce * self.shift_loss_weight
        else:
            shift_loss = torch.tensor(0.0, device=logit.device)
        loss_dict['shift'] = shift_loss.detach()

        # ====== Total ======
        total = (bce_loss + dice_loss + dst_loss + pair_count_loss + ratio_loss +
                 density_loss + fp_loss + shift_loss)
        loss_dict['total'] = total.detach()

        return total, loss_dict

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict, budget_fraction: float = 0.30,
                use_density_budget: bool = True, score_threshold: float = 0.43,
                length_decay: float = 0.15, budget_floor: float = 0.6):
        """Inference — [P4] 不过滤非标准配对。"""
        contact_mask = batch['contact_mask']
        seq_oh = batch.get('seq_oh')
        set_len = contact_mask.shape[-1]

        mars_hidden, mars_attn = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], set_len)

        pair = self._build_pair_features(mars_hidden, mars_attn, seq_oh)
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

        # Length-aware budget with floor
        l_eff = valid[:, 0, :].sum(dim=-1)
        if use_density_budget:
            length_factor = (100.0 / l_eff.clamp(min=50)) ** length_decay
            length_factor = length_factor.clamp(min=budget_floor)
            max_pairs = torch.round(density_pred.squeeze(-1) * l_eff * length_factor * 1.05).long()
        else:
            max_pairs = torch.round(l_eff * budget_fraction).long()
        max_pairs = max_pairs.clamp(min=2)

        # Greedy projection — [P4] 不过滤非标准配对
        B, _, L, _ = score.shape
        pred_maps = []
        for i in range(B):
            s = score[i, 0]
            m = contact_mask[i, 0]
            upper = torch.triu(torch.ones(L, L, device=s.device, dtype=torch.bool), diagonal=3)
            candidates = s * m * upper.float()
            candidates[candidates < score_threshold] = 0

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
