# -*- coding: utf-8 -*-
"""PriFold-SymFlow v4 Discrete Flow Matching (self-contained).

Includes:
  - Forward noising (sample_x_t_given_x_1)
  - Symmetrize utilities
  - CTMC rates computation
  - Physics losses (Stacking, NonCrossing)
  - BernoulliFlowLoss_v5 (with direct + pair_count calibration)
  - Projection functions (score-first, sample, hybrid)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Base utilities
# ======================================================================

def sample_x_t_given_x_1(x_1: torch.Tensor, t: torch.Tensor,
                          rho_0: float = 0.005) -> torch.Tensor:
    B = x_1.shape[0]
    t_b = t.view(B, 1, 1, 1)
    p_one = t_b * x_1 + (1.0 - t_b) * rho_0
    return (torch.rand_like(x_1) < p_one).float()


def symmetrize_binary(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, x.transpose(-2, -1))


def symmetrize_logit(logit: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logit + logit.transpose(-2, -1))


# ============================================================
# CTMC rates — same as v3
# ============================================================

def compute_ctmc_rates(x_t: torch.Tensor, p_x1: torch.Tensor,
                       t: torch.Tensor, rho_0: float = 0.005,
                       rate_clip: float = 50.0):
    B = x_t.shape[0]
    t_b = t.view(B, 1, 1, 1)
    eps = 1e-6
    p_xt_1 = (1.0 - t_b) * rho_0 + t_b * p_x1
    p_xt_0 = 1.0 - p_xt_1
    rate_01 = torch.clamp(p_x1 - rho_0, min=0.0) / (p_xt_0 + eps)
    rate_10 = torch.clamp(rho_0 - p_x1, min=0.0) / (p_xt_1 + eps)
    rate_01 = torch.clamp(rate_01, max=rate_clip)
    rate_10 = torch.clamp(rate_10, max=rate_clip)
    return rate_01, rate_10


# ============================================================
# Physics Losses — enhanced from v3
# ============================================================

class StackingLoss(nn.Module):
    """Encourage stacking continuity: (i,j) paired → (i+1,j-1) should be paired."""

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self, logit, contact_masks):
        if self.weight == 0:
            return torch.tensor(0.0, device=logit.device)
        prob = torch.sigmoid(logit)
        prob_shift = F.pad(prob[:, :, 1:, :-1], (1, 0, 0, 1))
        mask = contact_masks[:, :, 1:, :-1]
        mask = F.pad(mask, (1, 0, 0, 1))
        mask = mask * contact_masks
        stack_agreement = prob * prob_shift * mask
        loss = -stack_agreement.sum() / mask.sum().clamp(min=1.0)
        return self.weight * loss


class NonCrossingLoss(nn.Module):
    """Penalize multiple pairs per base (soft constraint for non-crossing)."""

    def __init__(self, weight: float = 0.02):
        super().__init__()
        self.weight = weight

    def forward(self, logit, contact_masks):
        if self.weight == 0:
            return torch.tensor(0.0, device=logit.device)
        prob = torch.sigmoid(logit) * contact_masks
        row_sum = prob.squeeze(1).sum(dim=-1)
        excess = F.relu(row_sum - 1.0)
        loss = excess.mean()
        return self.weight * loss


# ============================================================
# Adaptive Density-Aware Loss (NEW in v4)
# ============================================================



# ======================================================================
# v5 Loss + Projection
# ======================================================================

class BernoulliFlowLoss_v5(nn.Module):
    """Bernoulli flow loss with direct-logit and pair-count calibration."""

    def __init__(self, rho_0: float = 0.005, time_weight: bool = True,
                 pos_weight_base: float = 199.0,
                 pos_weight_min: float = 10.0,
                 focal_gamma: float = 2.0,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.02,
                 density_weight: float = 0.2,
                 direct_weight: float = 0.3,
                 pair_count_weight: float = 0.05):
        super().__init__()
        self.rho_0 = rho_0
        self.time_weight = time_weight
        self.pos_weight_base = pos_weight_base
        self.pos_weight_min = pos_weight_min
        self.focal_gamma = focal_gamma
        self.density_weight = density_weight
        self.direct_weight = direct_weight
        self.pair_count_weight = pair_count_weight
        self.stacking_loss = StackingLoss(weight=stack_weight)
        self.nc_loss = NonCrossingLoss(weight=nc_weight)

    def _valid_mask(self, logit, contact_masks):
        L = logit.shape[-1]
        idx = torch.arange(L, device=logit.device)
        short_range_ok = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3)
        return contact_masks * short_range_ok.view(1, 1, L, L).float()

    def _compute_adaptive_pos_weight(self, x_1, contact_masks):
        B = x_1.shape[0]
        with torch.no_grad():
            valid = contact_masks.squeeze(1)
            L_eff = valid[:, 0, :].sum(dim=-1)
            gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
            pair_per_base = gt_pairs / L_eff.clamp(min=1)
            alpha = (pair_per_base / 0.5).clamp(0, 1)
            pos_weight = self.pos_weight_min + alpha * (self.pos_weight_base - self.pos_weight_min)
        return pos_weight.view(B, 1, 1, 1)

    def _masked_adaptive_bce(self, logit, x_1, t, contact_masks):
        B = logit.shape[0]
        pos_weight = self._compute_adaptive_pos_weight(x_1, contact_masks)
        logsig = F.logsigmoid(logit)
        lognsig = F.logsigmoid(-logit)
        bce = -(pos_weight * x_1 * logsig + (1 - x_1) * lognsig)
        if self.focal_gamma > 0:
            with torch.no_grad():
                p = torch.sigmoid(logit)
                pt = p * x_1 + (1 - p) * (1 - x_1)
                focal_w = (1 - pt) ** self.focal_gamma
            bce = bce * focal_w
        if self.time_weight and t is not None:
            t_b = t.view(B, 1, 1, 1).clamp(0.0, 1.0 - 1e-3)
            bce = bce * (1.0 / (1.0 - t_b * (1.0 - self.rho_0)))
        mask = self._valid_mask(logit, contact_masks)
        return (bce * mask).sum() / mask.sum().clamp(min=1.0)

    def _density_targets(self, x_1, contact_masks):
        valid = contact_masks.squeeze(1)
        L_eff = valid[:, 0, :].sum(dim=-1)
        gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
        gt_density = (gt_pairs / L_eff.clamp(min=1)).unsqueeze(1)
        return gt_pairs, gt_density, L_eff

    def forward(self, logit, x_1, t, contact_masks,
                density_pred=None, direct_logit=None, return_gt_density=False):
        bce_loss = self._masked_adaptive_bce(logit, x_1, t, contact_masks)
        stack_loss = self.stacking_loss(logit, contact_masks)
        nc_loss = self.nc_loss(logit, contact_masks)

        gt_pairs, gt_density, L_eff = self._density_targets(x_1, contact_masks)

        density_loss = torch.tensor(0.0, device=logit.device)
        if density_pred is not None and self.density_weight > 0:
            # Huber is less dominated by extreme density bins than MSE.
            density_loss = self.density_weight * F.smooth_l1_loss(density_pred, gt_density)

        direct_loss = torch.tensor(0.0, device=logit.device)
        pair_count_loss = torch.tensor(0.0, device=logit.device)
        if direct_logit is not None:
            direct_loss = self.direct_weight * self._masked_adaptive_bce(
                direct_logit, x_1, None, contact_masks)
            mask = self._valid_mask(direct_logit, contact_masks)
            pred_pairs = (torch.sigmoid(direct_logit) * mask).sum(dim=(-1, -2, -3)) / 2
            pred_density = pred_pairs / L_eff.clamp(min=1)
            pair_count_loss = self.pair_count_weight * F.smooth_l1_loss(
                pred_density, gt_density.squeeze(1))

        total = bce_loss + stack_loss + nc_loss + density_loss + direct_loss + pair_count_loss
        loss_dict = {
            'bce': bce_loss.detach(),
            'stack': stack_loss.detach(),
            'nc': nc_loss.detach(),
            'density': density_loss.detach(),
            'direct': direct_loss.detach(),
            'pair_count': pair_count_loss.detach(),
        }
        if return_gt_density:
            return total, loss_dict, gt_density
        return total, loss_dict


def _valid_pair_mask(score: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
    B, _, L, _ = score.shape
    idx = torch.arange(L, device=score.device)
    valid = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3).view(1, 1, L, L).float()
    return valid * contact_masks


def _normalize_max_pairs(max_pairs, B: int, L: int, device):
    if max_pairs is None:
        return torch.full((B,), L // 2, device=device, dtype=torch.long)
    if torch.is_tensor(max_pairs):
        out = max_pairs.to(device=device).view(-1).long()
        if out.numel() == 1:
            out = out.expand(B)
        return out.clamp(min=0, max=L // 2)
    return torch.full((B,), int(max_pairs), device=device, dtype=torch.long).clamp(min=0, max=L // 2)


def project_score_to_valid_contact_map(score: torch.Tensor,
                                       contact_masks: torch.Tensor,
                                       max_pairs=None,
                                       min_score: float = 0.5) -> torch.Tensor:
    """Greedy max matching decoded directly from scores over legal edges.

    Without a score threshold, score-only decoding would always select L/2
    pairs because sigmoid scores are positive everywhere. `min_score` is the
    default pair-confidence cutoff; `max_pairs` is an optional additional budget.
    """
    B, _, L, _ = score.shape
    device = score.device
    max_pairs_t = _normalize_max_pairs(max_pairs, B, L, device)
    s = score * _valid_pair_mask(score, contact_masks)
    s = 0.5 * (s + s.transpose(-2, -1))
    out = torch.zeros_like(score)
    s_remain = s.clone()
    selected = torch.zeros(B, device=device, dtype=torch.long)
    threshold = float(min_score)
    for _ in range(L // 2):
        flat = s_remain.view(B, L * L)
        max_val, max_idx = flat.max(dim=-1)
        active = (max_val > threshold) & (selected < max_pairs_t)
        if not active.any():
            break
        i = max_idx // L
        j = max_idx % L
        bb = torch.arange(B, device=device)[active]
        ii = i[active]
        jj = j[active]
        out[bb, 0, ii, jj] = 1.0
        out[bb, 0, jj, ii] = 1.0
        selected[bb] += 1
        s_remain[bb, 0, ii, :] = 0.0
        s_remain[bb, 0, :, ii] = 0.0
        s_remain[bb, 0, jj, :] = 0.0
        s_remain[bb, 0, :, jj] = 0.0
    return out


def project_to_valid_contact_map(x: torch.Tensor, score: torch.Tensor,
                                  contact_masks: torch.Tensor,
                                  max_pairs=None,
                                  min_score: float = 0.5) -> torch.Tensor:
    """v3-compatible candidate projection: only sampled x==1 edges are eligible."""
    return project_score_to_valid_contact_map(
        x * (score + 1e-6), contact_masks, max_pairs=max_pairs, min_score=min_score)


def project_hybrid_contact_map(x: torch.Tensor, score: torch.Tensor,
                               contact_masks: torch.Tensor,
                               candidate_weight: float = 0.35,
                               max_pairs=None,
                               min_score: float = 0.5) -> torch.Tensor:
    """Hybrid decode: score over all edges plus a bonus for sampled candidates."""
    hybrid = score + candidate_weight * x.float()
    return project_score_to_valid_contact_map(
        hybrid, contact_masks, max_pairs=max_pairs, min_score=min_score)


def density_to_max_pairs(density_pred: torch.Tensor, contact_masks: torch.Tensor,
                         scale: float = 1.0, min_pairs: int = 0):
    """Convert predicted pair-per-base density to per-sample pair budget."""
    valid = contact_masks.squeeze(1)
    L_eff = valid[:, 0, :].sum(dim=-1)
    budget = torch.round(density_pred.view(-1) * L_eff * scale).long()
    if min_pairs > 0:
        budget = budget.clamp(min=min_pairs)
    return budget
