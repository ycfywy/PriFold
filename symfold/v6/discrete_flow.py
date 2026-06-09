# -*- coding: utf-8 -*-
"""PriFold-SymFlow v6 discrete flow loss — modular & ablation-friendly.

All loss components are individually configurable via config:
  - Each component has an enable flag and a weight.
  - New components (e.g., tversky, label_smoothing) can be added easily.
  - Designed to support scientific ablation studies.

Usage in config:
  "loss": {
    "bce": {"enabled": true, "pos_weight_base": 99, "pos_weight_min": 10, "focal_gamma": 1.0, "time_weight": true},
    "dice": {"enabled": true, "weight": 0.5, "smooth": 1.0},
    "tversky": {"enabled": false, "weight": 0.5, "alpha": 0.3, "beta": 0.7, "smooth": 1.0},
    "pair_count": {"enabled": true, "weight": 0.3},
    "ratio_penalty": {"enabled": true, "weight": 0.2, "threshold": 1.2},
    "density": {"enabled": true, "weight": 0.2},
    "direct": {"enabled": true, "weight": 0.4},
    "stacking": {"enabled": true, "weight": 0.05},
    "non_crossing": {"enabled": true, "weight": 0.03},
    "label_smoothing": {"enabled": false, "epsilon": 0.01}
  }
"""
from __future__ import annotations

from typing import Any

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

# Re-export projection utilities
from symfold.v4.discrete_flow import (
    project_score_to_valid_contact_map,
    project_to_valid_contact_map,
    project_hybrid_contact_map,
    density_to_max_pairs,
)


class ModularFlowLoss(nn.Module):
    """Modular Bernoulli flow loss for ablation studies.

    Every loss component is independently enabled/disabled via config dict.
    This makes it trivial to run ablation experiments by just changing the config JSON.
    """

    def __init__(self, loss_config: dict, rho_0: float = 0.005):
        super().__init__()
        self.rho_0 = rho_0
        self.cfg = loss_config

        # BCE config
        bce_cfg = self.cfg.get('bce', {})
        self.bce_enabled = bce_cfg.get('enabled', True)
        self.pos_weight_base = bce_cfg.get('pos_weight_base', 99.0)
        self.pos_weight_min = bce_cfg.get('pos_weight_min', 10.0)
        self.focal_gamma = bce_cfg.get('focal_gamma', 1.0)
        self.time_weight = bce_cfg.get('time_weight', True)

        # Label smoothing config
        ls_cfg = self.cfg.get('label_smoothing', {})
        self.label_smoothing_enabled = ls_cfg.get('enabled', False)
        self.label_smoothing_eps = ls_cfg.get('epsilon', 0.01)

        # Dice loss config
        dice_cfg = self.cfg.get('dice', {})
        self.dice_enabled = dice_cfg.get('enabled', True)
        self.dice_weight = dice_cfg.get('weight', 0.5)
        self.dice_smooth = dice_cfg.get('smooth', 1.0)

        # Tversky loss config (generalized Dice: alpha controls FP, beta controls FN)
        tv_cfg = self.cfg.get('tversky', {})
        self.tversky_enabled = tv_cfg.get('enabled', False)
        self.tversky_weight = tv_cfg.get('weight', 0.5)
        self.tversky_alpha = tv_cfg.get('alpha', 0.3)  # FP weight (lower = less FP penalty)
        self.tversky_beta = tv_cfg.get('beta', 0.7)    # FN weight (higher = more recall push)
        self.tversky_smooth = tv_cfg.get('smooth', 1.0)

        # Pair count calibration
        pc_cfg = self.cfg.get('pair_count', {})
        self.pair_count_enabled = pc_cfg.get('enabled', True)
        self.pair_count_weight = pc_cfg.get('weight', 0.3)

        # Ratio penalty (asymmetric overprediction penalty)
        rp_cfg = self.cfg.get('ratio_penalty', {})
        self.ratio_penalty_enabled = rp_cfg.get('enabled', True)
        self.ratio_penalty_weight = rp_cfg.get('weight', 0.2)
        self.ratio_penalty_threshold = rp_cfg.get('threshold', 1.2)

        # Density regression
        den_cfg = self.cfg.get('density', {})
        self.density_enabled = den_cfg.get('enabled', True)
        self.density_weight = den_cfg.get('weight', 0.2)

        # Direct head BCE
        dir_cfg = self.cfg.get('direct', {})
        self.direct_enabled = dir_cfg.get('enabled', True)
        self.direct_weight = dir_cfg.get('weight', 0.4)

        # Stacking loss
        stack_cfg = self.cfg.get('stacking', {})
        self.stacking_enabled = stack_cfg.get('enabled', True)
        stack_w = stack_cfg.get('weight', 0.05)
        self.stacking_loss = StackingLoss(weight=stack_w)

        # Non-crossing loss
        nc_cfg = self.cfg.get('non_crossing', {})
        self.nc_enabled = nc_cfg.get('enabled', True)
        nc_w = nc_cfg.get('weight', 0.03)
        self.nc_loss = NonCrossingLoss(weight=nc_w)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _valid_mask(self, logit: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
        L = logit.shape[-1]
        idx = torch.arange(L, device=logit.device)
        short_range_ok = ((idx.view(L, 1) - idx.view(1, L)).abs() >= 3)
        return contact_masks * short_range_ok.view(1, 1, L, L).float()

    def _compute_adaptive_pos_weight(self, x_1: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
        B = x_1.shape[0]
        with torch.no_grad():
            valid = contact_masks.squeeze(1)
            L_eff = valid[:, 0, :].sum(dim=-1)
            gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
            pair_per_base = gt_pairs / L_eff.clamp(min=1)
            alpha = (pair_per_base / 0.5).clamp(0, 1)
            pos_weight = self.pos_weight_min + alpha * (self.pos_weight_base - self.pos_weight_min)
        return pos_weight.view(B, 1, 1, 1)

    def _density_targets(self, x_1: torch.Tensor, contact_masks: torch.Tensor):
        valid = contact_masks.squeeze(1)
        L_eff = valid[:, 0, :].sum(dim=-1)
        gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
        gt_density = (gt_pairs / L_eff.clamp(min=1)).unsqueeze(1)
        return gt_pairs, gt_density, L_eff

    def _apply_label_smoothing(self, x_1: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing: soft targets instead of hard 0/1."""
        if self.label_smoothing_enabled:
            eps = self.label_smoothing_eps
            return x_1 * (1 - eps) + (1 - x_1) * eps
        return x_1

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _bce_loss(self, logit: torch.Tensor, x_1: torch.Tensor,
                  t: torch.Tensor | None, contact_masks: torch.Tensor) -> torch.Tensor:
        """Adaptive focal BCE with optional time weighting."""
        if not self.bce_enabled:
            return torch.tensor(0.0, device=logit.device)

        B = logit.shape[0]
        x_1_smooth = self._apply_label_smoothing(x_1)
        pos_weight = self._compute_adaptive_pos_weight(x_1, contact_masks)

        logsig = F.logsigmoid(logit)
        lognsig = F.logsigmoid(-logit)
        bce = -(pos_weight * x_1_smooth * logsig + (1 - x_1_smooth) * lognsig)

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

    def _dice_loss(self, logit: torch.Tensor, x_1: torch.Tensor,
                   contact_masks: torch.Tensor) -> torch.Tensor:
        """Differentiable Dice loss (F1 proxy)."""
        if not self.dice_enabled:
            return torch.tensor(0.0, device=logit.device)

        mask = self._valid_mask(logit, contact_masks)
        p = torch.sigmoid(logit) * mask
        gt = x_1 * mask
        intersection = (p * gt).sum(dim=(-1, -2, -3))
        union = p.sum(dim=(-1, -2, -3)) + gt.sum(dim=(-1, -2, -3))
        dice = (2 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        return self.dice_weight * (1 - dice).mean()

    def _tversky_loss(self, logit: torch.Tensor, x_1: torch.Tensor,
                      contact_masks: torch.Tensor) -> torch.Tensor:
        """Tversky loss: generalized Dice with asymmetric FP/FN control.

        - alpha < beta: penalize FN more (push recall)
        - alpha > beta: penalize FP more (push precision)
        """
        if not self.tversky_enabled:
            return torch.tensor(0.0, device=logit.device)

        mask = self._valid_mask(logit, contact_masks)
        p = torch.sigmoid(logit) * mask
        gt = x_1 * mask

        tp = (p * gt).sum(dim=(-1, -2, -3))
        fp = (p * (1 - gt)).sum(dim=(-1, -2, -3))
        fn = ((1 - p) * gt).sum(dim=(-1, -2, -3))

        tversky = (tp + self.tversky_smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.tversky_smooth)
        return self.tversky_weight * (1 - tversky).mean()

    def _pair_count_loss(self, logit: torch.Tensor, gt_density: torch.Tensor,
                         L_eff: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
        """Pair count calibration: L1 between predicted and GT density."""
        if not self.pair_count_enabled:
            return torch.tensor(0.0, device=logit.device)

        mask = self._valid_mask(logit, contact_masks)
        pred_pairs = (torch.sigmoid(logit) * mask).sum(dim=(-1, -2, -3)) / 2
        pred_density = pred_pairs / L_eff.clamp(min=1)
        return self.pair_count_weight * F.smooth_l1_loss(pred_density, gt_density.squeeze(1))

    def _ratio_penalty_loss(self, pred_density: torch.Tensor,
                            gt_density: torch.Tensor) -> torch.Tensor:
        """Asymmetric penalty when predicted density exceeds GT by threshold."""
        if not self.ratio_penalty_enabled:
            return torch.tensor(0.0, device=pred_density.device)

        ratio = pred_density / gt_density.squeeze(1).clamp(min=1e-4)
        excess = F.relu(ratio - self.ratio_penalty_threshold)
        return self.ratio_penalty_weight * excess.mean()

    def _density_loss(self, density_pred: torch.Tensor | None,
                      gt_density: torch.Tensor) -> torch.Tensor:
        """Density head regression loss."""
        if not self.density_enabled or density_pred is None:
            return torch.tensor(0.0, device=gt_density.device)
        return self.density_weight * F.smooth_l1_loss(density_pred, gt_density)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logit: torch.Tensor, x_1: torch.Tensor,
                t: torch.Tensor, contact_masks: torch.Tensor,
                density_pred: torch.Tensor | None = None,
                direct_logit: torch.Tensor | None = None,
                return_gt_density: bool = False):
        """Compute total loss with all enabled components.

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of component losses (all detached)
        """
        # Flow head BCE
        bce_loss = self._bce_loss(logit, x_1, t, contact_masks)

        # Structural losses
        stack_loss = self.stacking_loss(logit, contact_masks) if self.stacking_enabled else \
            torch.tensor(0.0, device=logit.device)
        nc_loss = self.nc_loss(logit, contact_masks) if self.nc_enabled else \
            torch.tensor(0.0, device=logit.device)

        # Density targets
        gt_pairs, gt_density, L_eff = self._density_targets(x_1, contact_masks)

        # Density regression
        density_loss = self._density_loss(density_pred, gt_density)

        # Direct head losses
        direct_loss = torch.tensor(0.0, device=logit.device)
        dice_loss = torch.tensor(0.0, device=logit.device)
        tversky_loss = torch.tensor(0.0, device=logit.device)
        pair_count_loss = torch.tensor(0.0, device=logit.device)
        ratio_penalty = torch.tensor(0.0, device=logit.device)

        if direct_logit is not None:
            # Direct head BCE (no time weighting)
            if self.direct_enabled:
                direct_loss = self.direct_weight * self._bce_loss(
                    direct_logit, x_1, None, contact_masks)

            # Dice on direct head
            dice_loss = self._dice_loss(direct_logit, x_1, contact_masks)

            # Tversky on direct head
            tversky_loss = self._tversky_loss(direct_logit, x_1, contact_masks)

            # Pair count calibration
            pair_count_loss = self._pair_count_loss(direct_logit, gt_density, L_eff, contact_masks)

            # Ratio penalty
            if self.ratio_penalty_enabled:
                mask = self._valid_mask(direct_logit, contact_masks)
                pred_pairs = (torch.sigmoid(direct_logit) * mask).sum(dim=(-1, -2, -3)) / 2
                pred_density = pred_pairs / L_eff.clamp(min=1)
                ratio_penalty = self._ratio_penalty_loss(pred_density, gt_density)

        total = (bce_loss + stack_loss + nc_loss + density_loss
                 + direct_loss + dice_loss + tversky_loss
                 + pair_count_loss + ratio_penalty)

        loss_dict = {
            'bce': bce_loss.detach(),
            'stack': stack_loss.detach(),
            'nc': nc_loss.detach(),
            'density': density_loss.detach(),
            'direct': direct_loss.detach(),
            'dice': dice_loss.detach(),
            'tversky': tversky_loss.detach(),
            'pair_count': pair_count_loss.detach(),
            'ratio_pen': ratio_penalty.detach(),
        }

        if return_gt_density:
            return total, loss_dict, gt_density
        return total, loss_dict

    def describe(self) -> str:
        """Return human-readable description of enabled losses (for logging)."""
        lines = ['ModularFlowLoss components:']
        if self.bce_enabled:
            lines.append(f'  [✓] BCE: focal_gamma={self.focal_gamma}, '
                         f'pos_weight={self.pos_weight_min}~{self.pos_weight_base}, '
                         f'time_weight={self.time_weight}')
        else:
            lines.append('  [✗] BCE: disabled')
        if self.label_smoothing_enabled:
            lines.append(f'  [✓] Label Smoothing: eps={self.label_smoothing_eps}')
        if self.dice_enabled:
            lines.append(f'  [✓] Dice: weight={self.dice_weight}, smooth={self.dice_smooth}')
        else:
            lines.append('  [✗] Dice: disabled')
        if self.tversky_enabled:
            lines.append(f'  [✓] Tversky: weight={self.tversky_weight}, '
                         f'alpha={self.tversky_alpha}, beta={self.tversky_beta}')
        else:
            lines.append('  [✗] Tversky: disabled')
        if self.direct_enabled:
            lines.append(f'  [✓] Direct BCE: weight={self.direct_weight}')
        else:
            lines.append('  [✗] Direct BCE: disabled')
        if self.pair_count_enabled:
            lines.append(f'  [✓] Pair Count: weight={self.pair_count_weight}')
        else:
            lines.append('  [✗] Pair Count: disabled')
        if self.ratio_penalty_enabled:
            lines.append(f'  [✓] Ratio Penalty: weight={self.ratio_penalty_weight}, '
                         f'threshold={self.ratio_penalty_threshold}')
        else:
            lines.append('  [✗] Ratio Penalty: disabled')
        if self.density_enabled:
            lines.append(f'  [✓] Density: weight={self.density_weight}')
        else:
            lines.append('  [✗] Density: disabled')
        if self.stacking_enabled:
            lines.append(f'  [✓] Stacking: weight={self.stacking_loss.weight}')
        else:
            lines.append('  [✗] Stacking: disabled')
        if self.nc_enabled:
            lines.append(f'  [✓] Non-Crossing: weight={self.nc_loss.weight}')
        else:
            lines.append('  [✗] Non-Crossing: disabled')
        return '\n'.join(lines)
