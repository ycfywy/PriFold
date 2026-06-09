# -*- coding: utf-8 -*-
"""PriFold-SymFlow v6 model wrapper — modular loss + full ablation support.

All loss components and architecture choices are configurable through the config dict,
enabling systematic ablation studies for scientific papers.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .da_se_dit import DASEDiT_MARS_v6
from .discrete_flow import (
    ModularFlowLoss,
    sample_x_t_given_x_1,
    symmetrize_binary,
    symmetrize_logit,
    compute_ctmc_rates,
    project_to_valid_contact_map,
    project_score_to_valid_contact_map,
    project_hybrid_contact_map,
    density_to_max_pairs,
)
from prifold.llama2_with_attn import mars_forward_with_attn


class PriFoldSymFlow_v6(nn.Module):
    """MARS-only SymFlow v6: modular loss with ablation support.

    Key difference from v5:
      - Loss is configured via a "loss" dict in config (each component has enabled/weight)
      - Architecture ablation flags: use_direct_head, use_density_head
      - Compatible with v5 checkpoints when using same architecture settings
    """

    def __init__(self,
                 extractor: nn.Module,
                 # MARS config
                 freeze_mars: bool = True,
                 mars_dim: int = 1056,
                 mars_n_attn_layers: int = 6,
                 mars_n_heads: int = 12,
                 mars_hidden_layer_indices: list | None = None,
                 mars_hidden_fusion_dim: int = 64,
                 use_seq_oh: bool = True,
                 # Architecture config
                 hidden_dim: int = 320,
                 num_heads: int = 4,
                 dim_head: int = 80,
                 num_layers: int = 12,
                 patch_size: int = 4,
                 mars_emb_proj_dim: int = 32,
                 mars_attn_proj_dim: int = 16,
                 xt_emb_dim: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 dilation_pattern: list | None = None,
                 tri_start_layer: int = 4,
                 tri_dim: int = 64,
                 refine_mid_ch: int = 16,
                 cond_bias_zero_init: bool = True,
                 control_every: int = 3,
                 # Architecture ablation flags
                 use_direct_head: bool = True,
                 use_density_head: bool = True,
                 # Flow config
                 rho_0: float = 0.005,
                 density_hint_dropout: float = 1.0,
                 direct_score_weight: float = 0.5,
                 # Modular loss config (dict)
                 loss_config: dict | None = None):
        super().__init__()
        self.rho_0 = rho_0
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.density_hint_dropout = float(density_hint_dropout)
        self.direct_score_weight = float(direct_score_weight)
        self.use_direct_head = use_direct_head
        self.use_density_head = use_density_head

        # MARS extractor
        self.extractor = extractor
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        # Backbone
        self.backbone = DASEDiT_MARS_v6(
            mars_dim=mars_dim,
            n_attn_layers=mars_n_attn_layers,
            n_heads_mars=mars_n_heads,
            mars_hidden_fusion_dim=mars_hidden_fusion_dim,
            mars_hidden_layers=len(self.mars_hidden_layer_indices),
            use_seq_oh=use_seq_oh,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            num_layers=num_layers,
            patch_size=patch_size,
            mars_emb_proj_dim=mars_emb_proj_dim,
            mars_attn_proj_dim=mars_attn_proj_dim,
            xt_emb_dim=xt_emb_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            dilation_pattern=dilation_pattern,
            tri_start_layer=tri_start_layer,
            tri_dim=tri_dim,
            refine_mid_ch=refine_mid_ch,
            cond_bias_zero_init=cond_bias_zero_init,
            control_every=control_every,
            use_direct_head=use_direct_head,
            use_density_head=use_density_head,
        )

        # Modular loss
        if loss_config is None:
            loss_config = self._default_loss_config()
        self.flow_loss = ModularFlowLoss(loss_config, rho_0=rho_0)

    @staticmethod
    def _default_loss_config() -> dict:
        """Default loss config matching v5 best settings."""
        return {
            'bce': {'enabled': True, 'pos_weight_base': 99.0, 'pos_weight_min': 10.0,
                    'focal_gamma': 1.0, 'time_weight': True},
            'dice': {'enabled': True, 'weight': 0.5, 'smooth': 1.0},
            'tversky': {'enabled': False, 'weight': 0.5, 'alpha': 0.3, 'beta': 0.7, 'smooth': 1.0},
            'pair_count': {'enabled': True, 'weight': 0.3},
            'ratio_penalty': {'enabled': True, 'weight': 0.2, 'threshold': 1.2},
            'density': {'enabled': True, 'weight': 0.2},
            'direct': {'enabled': True, 'weight': 0.4},
            'stacking': {'enabled': True, 'weight': 0.05},
            'non_crossing': {'enabled': True, 'weight': 0.03},
            'label_smoothing': {'enabled': False, 'epsilon': 0.01},
        }

    def _extract_mars(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      set_len: int):
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

        base_token_len = input_ids.shape[1] - 2
        hidden_base = hidden[:, 1:1 + base_token_len, :]
        hidden_layers_base = [h[:, 1:1 + base_token_len, :] for h in hidden_layers]
        attn_base = attn_stack[:, :, :, 1:1 + base_token_len, 1:1 + base_token_len]

        b, cur_len, d = hidden_base.shape
        if cur_len < set_len:
            pad_len = set_len - cur_len
            hidden_base = torch.cat([
                hidden_base,
                torch.zeros(b, pad_len, d, device=hidden_base.device, dtype=hidden_base.dtype)
            ], dim=1)
            hidden_layers_base = [
                torch.cat([h, torch.zeros(b, pad_len, d, device=h.device, dtype=h.dtype)], dim=1)
                for h in hidden_layers_base
            ]
            nl, nh = attn_base.shape[1], attn_base.shape[2]
            attn_pad = torch.zeros(b, nl, nh, set_len, set_len,
                                   device=attn_base.device, dtype=attn_base.dtype)
            attn_pad[:, :, :, :cur_len, :cur_len] = attn_base
            attn_base = attn_pad
        elif cur_len > set_len:
            hidden_base = hidden_base[:, :set_len, :]
            hidden_layers_base = [h[:, :set_len, :] for h in hidden_layers_base]
            attn_base = attn_base[:, :, :, :set_len, :set_len]
        return hidden_base, attn_base, hidden_layers_base

    def forward(self, batch: dict):
        contact = symmetrize_binary(batch['contact']) * batch['contact_mask']
        contact_mask = batch['contact_mask']
        set_len = contact.shape[-1]
        b = contact.shape[0]

        mars_hidden, mars_attn, mars_hidden_layers = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], set_len)

        t = torch.rand(b, device=contact.device)
        x_t = sample_x_t_given_x_1(contact, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_mask

        # Density hint (with dropout for training robustness)
        density_hint_train = None
        if self.use_density_head:
            with torch.no_grad():
                valid = contact_mask.squeeze(1)
                l_eff = valid[:, 0, :].sum(dim=-1)
                gt_pairs = (contact.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
                gt_density = (gt_pairs / l_eff.clamp(min=1)).unsqueeze(1)

            if self.training and self.density_hint_dropout > 0:
                keep_mask = (torch.rand(b, 1, device=gt_density.device)
                             > self.density_hint_dropout).float()
                density_hint_train = gt_density * keep_mask
                if keep_mask.sum() == 0:
                    density_hint_train = None
            else:
                density_hint_train = gt_density

        logit, density_pred, direct_logit = self.backbone(
            x_t, t,
            mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
            mars_attn=mars_attn, pos_bias=batch['pos_bias'], seq_oh=batch.get('seq_oh'),
            contact_masks=contact_mask,
            density_hint=density_hint_train,
            return_density=True,
            return_direct=True)

        total_loss, loss_dict = self.flow_loss(
            logit, contact, t, contact_mask,
            density_pred=density_pred,
            direct_logit=direct_logit)
        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

    @torch.no_grad()
    def sample(self, batch: dict, num_steps: int = 20,
               num_samples_per_input: int = 1,
               density_guided: bool = False,
               projection_mode: str = 'score',
               use_density_budget: bool = False,
               budget_scale: float = 1.0,
               candidate_weight: float = 0.35,
               direct_score_weight: float | None = None,
               score_threshold: float = 0.5,
               default_budget_fraction: float = 0.30):
        contact_mask = batch['contact_mask']
        pos_bias = batch['pos_bias']
        device = contact_mask.device
        b_real, _, l, _ = contact_mask.shape
        direct_w = self.direct_score_weight if direct_score_weight is None else float(direct_score_weight)

        mars_hidden, mars_attn, mars_hidden_layers = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], l)
        seq_oh = batch.get('seq_oh')

        if num_samples_per_input > 1:
            mars_hidden = mars_hidden.repeat(num_samples_per_input, 1, 1)
            mars_hidden_layers = [h.repeat(num_samples_per_input, 1, 1) for h in mars_hidden_layers]
            mars_attn = mars_attn.repeat(num_samples_per_input, 1, 1, 1, 1)
            pos_bias = pos_bias.repeat(num_samples_per_input, 1, 1)
            contact_mask = contact_mask.repeat(num_samples_per_input, 1, 1, 1)
            if seq_oh is not None:
                seq_oh = seq_oh.repeat(num_samples_per_input, 1, 1)
        b = b_real * num_samples_per_input

        need_density = density_guided or use_density_budget
        density_pred = None
        x_init = (torch.rand(b, 1, l, l, device=device) < self.rho_0).float()
        x_init = symmetrize_binary(x_init) * contact_mask
        if need_density and self.use_density_head:
            t_half = torch.full((b,), 0.5, device=device)
            _, density_pred, _ = self.backbone(
                x_init, t_half,
                mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
                mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=seq_oh,
                contact_masks=contact_mask, density_hint=None,
                return_density=True, return_direct=True)

        # Cosine schedule for tau-leap
        raw = [math.sin(math.pi * (k + 0.5) / (2 * num_steps)) for k in range(num_steps)]
        total_raw = sum(raw)
        dt_list = [r / total_raw for r in raw]
        x_t = x_init
        score_last = torch.zeros_like(x_t)
        t_cum = 0.0

        for dt in dt_list:
            t_tensor = torch.full((b,), t_cum, device=device)
            backbone_out = self.backbone(
                x_t, t_tensor,
                mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
                mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=seq_oh,
                contact_masks=contact_mask,
                density_hint=density_pred if density_guided else None,
                return_direct=True)

            if self.use_direct_head:
                flow_logit, direct_logit = backbone_out
            else:
                flow_logit = backbone_out if not isinstance(backbone_out, tuple) else backbone_out[0]
                direct_logit = flow_logit  # fallback: use flow as direct

            flow_logit = symmetrize_logit(flow_logit)
            direct_logit = symmetrize_logit(direct_logit)
            p_flow = torch.sigmoid(flow_logit)
            p_direct = torch.sigmoid(direct_logit)
            score_last = (1.0 - direct_w) * p_flow + direct_w * p_direct

            rate_01, rate_10 = compute_ctmc_rates(x_t, p_flow, t_tensor, rho_0=self.rho_0)
            if density_guided and density_pred is not None:
                damp = (2.0 * density_pred).clamp(max=1.0).view(b, 1, 1, 1)
                rate_01 = rate_01 * damp
            f01 = torch.clamp(rate_01 * dt, max=1.0)
            f10 = torch.clamp(rate_10 * dt, max=1.0)
            flip01 = (torch.rand_like(f01) < f01) & (x_t < 0.5)
            flip10 = (torch.rand_like(f10) < f10) & (x_t > 0.5)
            x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
            x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
            x_t = symmetrize_binary(x_t) * contact_mask
            t_cum += dt

        # Aggregate multi-sample
        if num_samples_per_input > 1:
            x_vote = x_t.view(num_samples_per_input, b_real, 1, l, l).mean(0)
            score_final = score_last.view(num_samples_per_input, b_real, 1, l, l).mean(0)
            mask_final = contact_mask.view(num_samples_per_input, b_real, 1, l, l)[0]
            if density_pred is not None:
                density_final = density_pred.view(num_samples_per_input, b_real, 1).mean(0)
            else:
                density_final = None
        else:
            x_vote = x_t
            score_final = score_last
            mask_final = contact_mask
            density_final = density_pred

        # Budget computation
        if use_density_budget and density_final is not None:
            max_pairs = density_to_max_pairs(density_final, mask_final, scale=budget_scale)
        else:
            valid = mask_final.squeeze(1)
            length_eff = valid[:, 0, :].sum(dim=-1)
            max_pairs = torch.round(length_eff * float(default_budget_fraction)).long()

        # Projection
        mode = str(projection_mode).lower()
        if mode == 'sample':
            x_final = project_to_valid_contact_map(
                x_vote, score_final, mask_final,
                max_pairs=max_pairs, min_score=score_threshold)
        elif mode == 'hybrid':
            x_final = project_hybrid_contact_map(
                x_vote, score_final, mask_final,
                candidate_weight=candidate_weight,
                max_pairs=max_pairs,
                min_score=score_threshold)
        else:
            x_final = project_score_to_valid_contact_map(
                score_final, mask_final,
                max_pairs=max_pairs,
                min_score=score_threshold)
        return x_final, score_final
