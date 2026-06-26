# -*- coding: utf-8 -*-
"""PriFold-SymFlow v5 discrete flow loss.

v5 changes vs v4:
  1. Reduced focal_gamma (2.0 → 1.0) to preserve gradient signal from medium-difficulty positives.
  2. Added Dice loss on direct_logit to directly optimize F1-like objective.
  3. Much stronger pair_count_weight (0.05 → 0.3) with ratio penalty.
  4. Asymmetric pos_weight: lower ceiling to reduce FP encouragement.
  5. Hard ratio penalty when pred/gt > 1.2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from symfold.v3.discrete_flow import (
    sample_x_t_given_x_1,
    symmetrize_binary,
    symmetrize_logit,
    compute_ctmc_rates,
    StackingLoss,
    NonCrossingLoss,
)

# Re-export for convenience
from symfold.v4.discrete_flow import (
    project_score_to_valid_contact_map,
    project_to_valid_contact_map,
    project_hybrid_contact_map,
    density_to_max_pairs,
)


class BernoulliFlowLoss_v6(nn.Module):
    """v5/v6 loss: stronger gradient signal + anti-overprediction.

    Key differences from v5 (BernoulliFlowLoss_v5):
      - focal_gamma default 1.0 (was 2.0): preserves more gradient from medium predictions
      - dice_weight > 0: adds differentiable Dice loss on direct head (F1 proxy)
      - pair_count_weight default 0.3 (was 0.05): much stronger calibration
      - ratio_penalty_weight: explicit penalty when pred_density/gt_density > threshold
      - pos_weight_base default 99 (was 199): reduce FP encouragement
    """

    def __init__(self, rho_0: float = 0.005, time_weight: bool = True,
                 pos_weight_base: float = 99.0,
                 pos_weight_min: float = 10.0,
                 focal_gamma: float = 1.0,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.03,
                 density_weight: float = 0.2,
                 direct_weight: float = 0.4,
                 pair_count_weight: float = 0.3,
                 dice_weight: float = 0.5,
                 ratio_penalty_weight: float = 0.2,
                 ratio_penalty_threshold: float = 1.2):
        super().__init__()
        self.rho_0 = rho_0
        self.time_weight = time_weight
        self.pos_weight_base = pos_weight_base
        self.pos_weight_min = pos_weight_min
        self.focal_gamma = focal_gamma
        self.density_weight = density_weight
        self.direct_weight = direct_weight
        self.pair_count_weight = pair_count_weight
        self.dice_weight = dice_weight
        self.ratio_penalty_weight = ratio_penalty_weight
        self.ratio_penalty_threshold = ratio_penalty_threshold
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

    def _dice_loss(self, logit, x_1, contact_masks):
        """Differentiable Dice loss (F1 proxy) on positive regions."""
        mask = self._valid_mask(logit, contact_masks)
        p = torch.sigmoid(logit) * mask
        gt = x_1 * mask
        # Per-sample dice
        intersection = (p * gt).sum(dim=(-1, -2, -3))
        union = p.sum(dim=(-1, -2, -3)) + gt.sum(dim=(-1, -2, -3))
        dice = (2 * intersection + 1) / (union + 1)  # smooth
        return (1 - dice).mean()

    def _ratio_penalty(self, pred_density, gt_density):
        """Penalize when predicted density exceeds GT by threshold ratio."""
        ratio = pred_density / gt_density.squeeze(1).clamp(min=1e-4)
        excess = F.relu(ratio - self.ratio_penalty_threshold)
        return excess.mean()

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
            density_loss = self.density_weight * F.smooth_l1_loss(density_pred, gt_density)

        direct_loss = torch.tensor(0.0, device=logit.device)
        pair_count_loss = torch.tensor(0.0, device=logit.device)
        dice_loss = torch.tensor(0.0, device=logit.device)
        ratio_penalty = torch.tensor(0.0, device=logit.device)

        if direct_logit is not None:
            # Direct BCE (no time weighting)
            direct_loss = self.direct_weight * self._masked_adaptive_bce(
                direct_logit, x_1, None, contact_masks)

            # Dice loss on direct head (F1 proxy)
            if self.dice_weight > 0:
                dice_loss = self.dice_weight * self._dice_loss(direct_logit, x_1, contact_masks)

            # Pair count calibration (much stronger than v4)
            mask = self._valid_mask(direct_logit, contact_masks)
            pred_pairs = (torch.sigmoid(direct_logit) * mask).sum(dim=(-1, -2, -3)) / 2
            pred_density = pred_pairs / L_eff.clamp(min=1)
            pair_count_loss = self.pair_count_weight * F.smooth_l1_loss(
                pred_density, gt_density.squeeze(1))

            # Ratio penalty: explicitly punish overprediction
            if self.ratio_penalty_weight > 0:
                ratio_penalty = self.ratio_penalty_weight * self._ratio_penalty(
                    pred_density, gt_density)

        total = (bce_loss + stack_loss + nc_loss + density_loss
                 + direct_loss + dice_loss + pair_count_loss + ratio_penalty)
        loss_dict = {
            'bce': bce_loss.detach(),
            'stack': stack_loss.detach(),
            'nc': nc_loss.detach(),
            'density': density_loss.detach(),
            'direct': direct_loss.detach(),
            'dice': dice_loss.detach(),
            'pair_count': pair_count_loss.detach(),
            'ratio_pen': ratio_penalty.detach(),
        }
        if return_gt_density:
            return total, loss_dict, gt_density
        return total, loss_dict
