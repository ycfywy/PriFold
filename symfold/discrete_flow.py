from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_x_t_given_x_1(x_1: torch.Tensor, t: torch.Tensor, rho_0: float = 0.005) -> torch.Tensor:
    bsz = x_1.shape[0]
    t_b = t.view(bsz, 1, 1, 1)
    p_one = t_b * x_1 + (1.0 - t_b) * rho_0
    return (torch.rand_like(x_1) < p_one).float()


def symmetrize_binary(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, x.transpose(-2, -1))


def symmetrize_logit(logit: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logit + logit.transpose(-2, -1))


def compute_ctmc_rates(
    x_t: torch.Tensor,
    p_x1: torch.Tensor,
    t: torch.Tensor,
    rho_0: float = 0.005,
    rate_clip: float = 50.0,
):
    bsz = x_t.shape[0]
    t_b = t.view(bsz, 1, 1, 1)
    eps = 1e-6
    p_xt_1 = (1.0 - t_b) * rho_0 + t_b * p_x1
    p_xt_0 = 1.0 - p_xt_1
    rate_01 = torch.clamp(p_x1 - rho_0, min=0.0) / (p_xt_0 + eps)
    rate_10 = torch.clamp(rho_0 - p_x1, min=0.0) / (p_xt_1 + eps)
    return torch.clamp(rate_01, max=rate_clip), torch.clamp(rate_10, max=rate_clip)


def valid_pair_mask(contact_masks: torch.Tensor) -> torch.Tensor:
    _, _, length, _ = contact_masks.shape
    idx = torch.arange(length, device=contact_masks.device)
    short_range_ok = (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
    return contact_masks * short_range_ok.view(1, 1, length, length).float()


class StackingLoss(nn.Module):
    def __init__(self, weight: float = 0.0):
        super().__init__()
        self.weight = weight

    def forward(self, logit: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return torch.tensor(0.0, device=logit.device)
        prob = torch.sigmoid(logit)
        prob_shift = F.pad(prob[:, :, 1:, :-1], (1, 0, 0, 1))
        mask = F.pad(contact_masks[:, :, 1:, :-1], (1, 0, 0, 1)) * contact_masks
        loss = -(prob * prob_shift * mask).sum() / mask.sum().clamp(min=1.0)
        return self.weight * loss


class NonCrossingLoss(nn.Module):
    def __init__(self, weight: float = 0.0):
        super().__init__()
        self.weight = weight

    def forward(self, logit: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return torch.tensor(0.0, device=logit.device)
        prob = torch.sigmoid(logit) * contact_masks
        row_sum = prob.squeeze(1).sum(dim=-1)
        return self.weight * F.relu(row_sum - 1.0).mean()


class BernoulliFlowLoss(nn.Module):
    def __init__(
        self,
        rho_0: float = 0.005,
        time_weight: bool = True,
        pos_weight_base: float = 199.0,
        pos_weight_min: float = 20.0,
        focal_gamma: float = 1.5,
        stack_weight: float = 0.0,
        nc_weight: float = 0.0,
        density_weight: float = 0.0,
    ):
        super().__init__()
        self.rho_0 = rho_0
        self.time_weight = time_weight
        self.pos_weight_base = pos_weight_base
        self.pos_weight_min = pos_weight_min
        self.focal_gamma = focal_gamma
        self.density_weight = density_weight
        self.stacking_loss = StackingLoss(stack_weight)
        self.nc_loss = NonCrossingLoss(nc_weight)

    def _adaptive_pos_weight(self, x_1: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
        bsz = x_1.shape[0]
        with torch.no_grad():
            valid = contact_masks.squeeze(1)
            length_eff = valid[:, 0, :].sum(dim=-1)
            gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
            pair_per_base = gt_pairs / length_eff.clamp(min=1)
            alpha = (pair_per_base / 0.5).clamp(0, 1)
            pos_weight = self.pos_weight_min + alpha * (self.pos_weight_base - self.pos_weight_min)
        return pos_weight.view(bsz, 1, 1, 1)

    def forward(
        self,
        logit: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        contact_masks: torch.Tensor,
        density_pred: torch.Tensor | None = None,
    ):
        bsz = logit.shape[0]
        mask = valid_pair_mask(contact_masks)
        pos_weight = self._adaptive_pos_weight(x_1, contact_masks)
        logsig = F.logsigmoid(logit)
        lognsig = F.logsigmoid(-logit)
        bce = -(pos_weight * x_1 * logsig + (1.0 - x_1) * lognsig)

        if self.focal_gamma > 0:
            with torch.no_grad():
                p = torch.sigmoid(logit)
                pt = p * x_1 + (1.0 - p) * (1.0 - x_1)
                focal_w = (1.0 - pt) ** self.focal_gamma
            bce = bce * focal_w

        if self.time_weight:
            t_b = t.view(bsz, 1, 1, 1).clamp(0.0, 1.0 - 1e-3)
            bce = bce * (1.0 / (1.0 - t_b * (1.0 - self.rho_0)))

        bce_loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
        stack_loss = self.stacking_loss(logit, contact_masks)
        nc_loss = self.nc_loss(logit, contact_masks)

        density_loss = torch.tensor(0.0, device=logit.device)
        if density_pred is not None and self.density_weight > 0:
            with torch.no_grad():
                valid = contact_masks.squeeze(1)
                length_eff = valid[:, 0, :].sum(dim=-1)
                gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
                gt_density = (gt_pairs / length_eff.clamp(min=1)).unsqueeze(1)
            density_loss = self.density_weight * F.mse_loss(density_pred, gt_density)

        total = bce_loss + stack_loss + nc_loss + density_loss
        return total, {
            "bce": bce_loss.detach(),
            "stack": stack_loss.detach(),
            "nc": nc_loss.detach(),
            "density": density_loss.detach(),
        }


@torch.no_grad()
def project_to_valid_contact_map(x: torch.Tensor, score: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
    bsz, _, length, _ = x.shape
    valid = valid_pair_mask(contact_masks)
    s = x * (score + 1e-6) * valid
    s = 0.5 * (s + s.transpose(-2, -1))
    out = torch.zeros_like(x)
    s_remain = s.clone()
    for _ in range(length // 2):
        flat = s_remain.view(bsz, length * length)
        max_val, max_idx = flat.max(dim=-1)
        if (max_val <= 0).all():
            break
        i = max_idx // length
        j = max_idx % length
        active = max_val > 0
        if active.any():
            bb = torch.arange(bsz, device=x.device)[active]
            ii = i[active]
            jj = j[active]
            out[bb, 0, ii, jj] = 1.0
            out[bb, 0, jj, ii] = 1.0
            s_remain[bb, 0, ii, :] = 0.0
            s_remain[bb, 0, :, ii] = 0.0
            s_remain[bb, 0, jj, :] = 0.0
            s_remain[bb, 0, :, jj] = 0.0
    return out * contact_masks
