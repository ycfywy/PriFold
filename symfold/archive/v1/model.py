from __future__ import annotations

import torch
import torch.nn as nn

from .dit import AxialDiT
from .discrete_flow import (
    BernoulliFlowLoss,
    compute_ctmc_rates,
    project_to_valid_contact_map,
    sample_x_t_given_x_1,
    symmetrize_binary,
    symmetrize_logit,
)


class PriFoldSymFlowModel(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        freeze_mars: bool = True,
        mars_dim: int = 1056,
        d_pair: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        patch_size: int = 4,
        dropout: float = 0.1,
        rho_0: float = 0.005,
        use_pos_bias: bool = True,
        output_refine: bool = True,
        pos_weight_base: float = 199.0,
        pos_weight_min: float = 20.0,
        focal_gamma: float = 1.5,
        stack_weight: float = 0.0,
        nc_weight: float = 0.0,
        density_weight: float = 0.2,
    ):
        super().__init__()
        self.extractor = extractor
        self.freeze_mars = freeze_mars
        self.rho_0 = rho_0
        self.use_pos_bias = use_pos_bias
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        self.mars_proj = nn.Sequential(
            nn.Linear(mars_dim, d_pair * 2),
            nn.GELU(),
            nn.Linear(d_pair * 2, d_pair),
        )
        self.x_t_embedding = nn.Embedding(2, 8)
        cond_channels = 2 * d_pair + 8 + (1 if use_pos_bias else 0)
        in_channels = 8 + cond_channels
        self.backbone = AxialDiT(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            dropout=dropout,
            output_refine=output_refine,
        )
        self.flow_loss = BernoulliFlowLoss(
            rho_0=rho_0,
            pos_weight_base=pos_weight_base,
            pos_weight_min=pos_weight_min,
            focal_gamma=focal_gamma,
            stack_weight=stack_weight,
            nc_weight=nc_weight,
            density_weight=density_weight,
        )

    @staticmethod
    def _outer_concat(x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, 2C, L, L)
        _, _, length = x.shape
        xi = x.unsqueeze(-1).expand(-1, -1, -1, length)
        xj = x.unsqueeze(-2).expand(-1, -1, length, -1)
        return torch.cat([xi, xj], dim=1)

    def _extract_mars_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_mars:
            with torch.no_grad():
                output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        else:
            output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
        return output[1]

    def build_conditions(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        seq_oh = batch["seq_oh"]
        pos_bias = batch["pos_bias"]
        set_len = batch["contact"].shape[-1]
        base_token_len = input_ids.shape[1] - 2

        hidden = self._extract_mars_hidden(input_ids, attention_mask)
        base_hidden = hidden[:, 1:1 + base_token_len, :]
        if base_hidden.shape[1] < set_len:
            pad = torch.zeros(
                base_hidden.shape[0],
                set_len - base_hidden.shape[1],
                base_hidden.shape[2],
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            )
            base_hidden = torch.cat([base_hidden, pad], dim=1)
        elif base_hidden.shape[1] > set_len:
            base_hidden = base_hidden[:, :set_len, :]

        mars_1d = self.mars_proj(base_hidden).permute(0, 2, 1).contiguous()
        mars_2d = self._outer_concat(mars_1d)
        seq_2d = self._outer_concat(seq_oh.permute(0, 2, 1).contiguous())
        parts = [mars_2d, seq_2d]
        if self.use_pos_bias:
            parts.append(pos_bias.unsqueeze(1))
        cond = torch.cat(parts, dim=1)
        return 0.5 * (cond + cond.transpose(-2, -1))

    def predict_from_conditions(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, contact_mask: torch.Tensor):
        x_long = x_t.long().squeeze(1).clamp(0, 1)
        x_emb = self.x_t_embedding(x_long).permute(0, 3, 1, 2).contiguous()
        features = torch.cat([x_emb, cond], dim=1)
        return self.backbone(features, t, contact_mask)

    def forward(self, batch: dict):
        contact = symmetrize_binary(batch["contact"]) * batch["contact_mask"]
        contact_mask = batch["contact_mask"]
        bsz = contact.shape[0]
        t = torch.rand(bsz, device=contact.device)
        x_t = sample_x_t_given_x_1(contact, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_mask
        cond = self.build_conditions(batch)
        logit, density = self.predict_from_conditions(x_t, t, cond, contact_mask)
        loss, loss_dict = self.flow_loss(logit, contact, t, contact_mask, density_pred=density)
        loss_dict["total"] = loss.detach()
        return loss, loss_dict

    @torch.no_grad()
    def sample(self, batch: dict, num_steps: int = 20, num_samples_per_input: int = 1):
        contact_mask = batch["contact_mask"]
        device = contact_mask.device
        bsz_real, _, length, _ = contact_mask.shape
        cond = self.build_conditions(batch)
        if num_samples_per_input > 1:
            cond = cond.repeat(num_samples_per_input, 1, 1, 1)
            contact_mask = contact_mask.repeat(num_samples_per_input, 1, 1, 1)
        bsz = contact_mask.shape[0]
        x_t = (torch.rand(bsz, 1, length, length, device=device) < self.rho_0).float()
        x_t = symmetrize_binary(x_t) * contact_mask
        p_x1_last = torch.zeros_like(x_t)

        for step in range(num_steps):
            t_val = step / max(num_steps, 1)
            t_tensor = torch.full((bsz,), t_val, device=device)
            logit, _ = self.predict_from_conditions(x_t, t_tensor, cond, contact_mask)
            logit = symmetrize_logit(logit)
            p_x1 = torch.sigmoid(logit)
            p_x1 = 0.5 * (p_x1 + p_x1.transpose(-2, -1))
            p_x1_last = p_x1
            rate_01, rate_10 = compute_ctmc_rates(x_t, p_x1, t_tensor, rho_0=self.rho_0)
            dt = 1.0 / max(num_steps, 1)
            flip01 = (torch.rand_like(rate_01) < torch.clamp(rate_01 * dt, max=1.0)) & (x_t < 0.5)
            flip10 = (torch.rand_like(rate_10) < torch.clamp(rate_10 * dt, max=1.0)) & (x_t > 0.5)
            x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
            x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
            x_t = symmetrize_binary(x_t) * contact_mask

        pred = project_to_valid_contact_map(x_t, p_x1_last, contact_mask)
        if num_samples_per_input > 1:
            pred = pred.view(num_samples_per_input, bsz_real, 1, length, length).mean(dim=0)
            pred = (pred > 0.5).float()
            pred = symmetrize_binary(pred)
            p_x1_last = p_x1_last.view(num_samples_per_input, bsz_real, 1, length, length).mean(dim=0)
        return pred, p_x1_last
