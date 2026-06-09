# -*- coding: utf-8 -*-
"""PriFold-SymFlow v2 主模型：MARS-only + DA-SE-DiT-MARS + Bernoulli DFM。

对照 SF v5 的 SymFoldModel_v5：
  - 去掉 RNA-FM / UFold 两个外部条件器；
  - 改用 MARS-LX 提供 hidden + 后 N 层 attention map；
  - pos_bias 仍由 PriFold 数据流水线提供，作为输入通道；
  - 训练注入 GT density、推理用预测 density 做 guided sampling；
  - cosine τ-leap schedule + density-guided rate damping。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .da_se_dit import DASEDiT_MARS_v2
from .discrete_flow import (
    BernoulliFlowLoss_v4,
    sample_x_t_given_x_1, symmetrize_binary, symmetrize_logit,
    compute_ctmc_rates, project_to_valid_contact_map,
)
from prifold.llama2_with_attn import mars_forward_with_attn


class PriFoldSymFlow_v2(nn.Module):
    """PriFold-SymFlow v2."""

    def __init__(self,
                 extractor: nn.Module,
                 freeze_mars: bool = True,
                 mars_dim: int = 1056,
                 mars_n_attn_layers: int = 6,
                 mars_n_heads: int = 12,
                 mars_hidden_layer_indices: list | None = None,
                 mars_hidden_fusion_dim: int = 64,
                 use_seq_oh: bool = True,
                 # backbone
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dim_head: int = 64,
                 num_layers: int = 9,
                 patch_size: int = 4,
                 mars_emb_proj_dim: int = 32,
                 mars_attn_proj_dim: int = 16,
                 xt_emb_dim: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 dilation_pattern: list | None = None,
                 tri_start_layer: int = 6,
                 tri_dim: int = 64,
                 refine_mid_ch: int = 16,
                 # flow loss
                 rho_0: float = 0.005,
                 pos_weight_base: float = 199.0,
                 pos_weight_min: float = 20.0,
                 focal_gamma: float = 1.5,
                 stack_weight: float = 0.05,
                 nc_weight: float = 0.02,
                 density_weight: float = 0.2,
                 # ---- density-conditioning regularization ----
                 # 训练时按概率把 GT density 替换为 None,
                 # 让 density head 真正学会从 (x_t, cond) 预测 density,
                 # 推理时 density_pred 才有意义 (避免"密度抄作业"信息泄漏)。
                 # 1.0 = 永不注入 GT (退化到 v1, density 仅作 loss);
                 # 0.0 = 永远注入 GT (容易过拟合, sample 时崩);
                 # 0.5 = 一半时间见到 GT, 一半时间没见到 (推荐)。
                 density_hint_dropout: float = 0.5):
        super().__init__()
        self.rho_0 = rho_0
        self.freeze_mars = freeze_mars
        self.mars_n_attn_layers = mars_n_attn_layers
        self.mars_hidden_layer_indices = mars_hidden_layer_indices or [3, 6, 9, 12]
        self.density_hint_dropout = float(density_hint_dropout)

        # 1) MARS frozen encoder
        self.extractor = extractor
        if freeze_mars:
            self.extractor.eval()
            for p in self.extractor.parameters():
                p.requires_grad = False

        # 2) DA-SE-DiT-MARS backbone
        self.backbone = DASEDiT_MARS_v2(
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
        )

        # 3) Flow loss
        self.flow_loss = BernoulliFlowLoss_v4(
            rho_0=rho_0,
            pos_weight_base=pos_weight_base,
            pos_weight_min=pos_weight_min,
            focal_gamma=focal_gamma,
            stack_weight=stack_weight,
            nc_weight=nc_weight,
            density_weight=density_weight,
        )

    # ------------------------------------------------------------
    # MARS extraction
    # ------------------------------------------------------------

    def _extract_mars(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      set_len: int):
        """Get MARS hidden + last-N attention, sliced to base tokens (no <cls>/<eos>)
        and padded/truncated to set_len.

        Returns
        -------
        hidden_base : (B, set_len, mars_dim)
        attn_base   : (B, n_attn_layers, n_heads, set_len, set_len)
        hidden_layers_base : list[(B, set_len, mars_dim)]
        """
        if self.freeze_mars:
            # model.train() 会递归把 extractor 设回 train；这里必须再次 eval，
            # 否则 frozen MARS 的 dropout 会造成 train/eval conditioning 不一致。
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

        # Strip <cls>/<eos>: tokenizer 加了 max_len+2 的 token，base = [1 : 1 + base_len]
        # 这里 input_ids 已包含 cls/eos，所以去 [1:-1] 即可（再 padding/truncate 到 set_len）
        base_token_len = input_ids.shape[1] - 2
        hidden_base = hidden[:, 1:1 + base_token_len, :]
        hidden_layers_base = [h[:, 1:1 + base_token_len, :] for h in hidden_layers]
        attn_base = attn_stack[:, :, :, 1:1 + base_token_len, 1:1 + base_token_len]

        B, _, D = hidden_base.shape
        cur_len = hidden_base.shape[1]
        if cur_len < set_len:
            pad_len = set_len - cur_len
            hidden_base = torch.cat(
                [hidden_base,
                 torch.zeros(B, pad_len, D, device=hidden_base.device, dtype=hidden_base.dtype)],
                dim=1)
            padded_layers = []
            for h in hidden_layers_base:
                padded_layers.append(torch.cat(
                    [h, torch.zeros(B, pad_len, D, device=h.device, dtype=h.dtype)],
                    dim=1))
            hidden_layers_base = padded_layers
            # pad attention with zeros
            n_layers = attn_base.shape[1]
            n_heads = attn_base.shape[2]
            attn_pad = torch.zeros(B, n_layers, n_heads, set_len, set_len,
                                   device=attn_base.device, dtype=attn_base.dtype)
            attn_pad[:, :, :, :cur_len, :cur_len] = attn_base
            attn_base = attn_pad
        elif cur_len > set_len:
            hidden_base = hidden_base[:, :set_len, :]
            hidden_layers_base = [h[:, :set_len, :] for h in hidden_layers_base]
            attn_base = attn_base[:, :, :, :set_len, :set_len]

        return hidden_base, attn_base, hidden_layers_base

    # ------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------

    def forward(self, batch: dict):
        """Training forward.

        batch (PriFold-SymFlow data pipeline):
            input_ids:    (B, L+2)
            attention_mask: (B, L+2)
            contact:      (B, 1, S, S)  GT contact map
            contact_mask: (B, 1, S, S)
            pos_bias:     (B, S, S)
        """
        contact = symmetrize_binary(batch['contact']) * batch['contact_mask']
        contact_mask = batch['contact_mask']
        pos_bias = batch['pos_bias']
        B = contact.shape[0]
        set_len = contact.shape[-1]

        # 1) MARS
        mars_hidden, mars_attn, mars_hidden_layers = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], set_len)

        # 2) Flow forward noising
        t = torch.rand(B, device=contact.device)
        x_1 = contact
        x_t = sample_x_t_given_x_1(x_1, t, rho_0=self.rho_0)
        x_t = symmetrize_binary(x_t) * contact_mask

        # 3) GT density (用作训练时条件 + density loss 监督)
        with torch.no_grad():
            valid = contact_mask.squeeze(1)
            L_eff = valid[:, 0, :].sum(dim=-1)
            gt_pairs = (x_1.squeeze(1) * valid).sum(dim=(-1, -2)) / 2
            gt_density = (gt_pairs / L_eff.clamp(min=1)).unsqueeze(1)

        # ---- Density-hint dropout: 让模型在 hint=None 路径下也能干活 ----
        # 否则训练时永远见 GT density、推理时 None forward 是 OOD,
        # 导致 sample 阶段 density_pred 完全失真 → P/R/F1 严重下降。
        if self.training and self.density_hint_dropout > 0:
            B = gt_density.shape[0]
            # 每个样本独立掷骰子: 概率 = density_hint_dropout 时把这一行的 hint 设为 0
            # 注意: 模型 _global_cond 里 None → zeros, 这里我们等价地在 batch 内 mask
            keep_mask = (torch.rand(B, 1, device=gt_density.device)
                         > self.density_hint_dropout).float()
            density_hint_train = gt_density * keep_mask
            # 全 batch 都被 mask 掉时, 等价于 density_hint=None 路径
            if keep_mask.sum() == 0:
                density_hint_train = None
        else:
            density_hint_train = gt_density

        # 4) Backbone
        logit, density_pred = self.backbone(
            x_t, t,
            mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
            mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=batch.get('seq_oh'),
            contact_masks=contact_mask,
            density_hint=density_hint_train,
            return_density=True)

        # 5) Loss
        total_loss, loss_dict = self.flow_loss(
            logit, x_1, t, contact_mask, density_pred=density_pred)
        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch: dict, num_steps: int = 20,
               num_samples_per_input: int = 1, density_guided: bool = True):
        """v5-style cosine τ-leap sampling with density guidance."""
        contact_mask = batch['contact_mask']
        pos_bias = batch['pos_bias']
        device = contact_mask.device
        B_real, _, L, _ = contact_mask.shape

        # 1) Extract MARS once
        mars_hidden, mars_attn, mars_hidden_layers = self._extract_mars(
            batch['input_ids'], batch['attention_mask'], L)
        seq_oh = batch.get('seq_oh')

        # 2) Replicate for multi-sample (single sample by default)
        if num_samples_per_input > 1:
            mars_hidden = mars_hidden.repeat(num_samples_per_input, 1, 1)
            mars_hidden_layers = [h.repeat(num_samples_per_input, 1, 1)
                                  for h in mars_hidden_layers]
            mars_attn = mars_attn.repeat(num_samples_per_input, 1, 1, 1, 1)
            pos_bias = pos_bias.repeat(num_samples_per_input, 1, 1)
            contact_mask = contact_mask.repeat(num_samples_per_input, 1, 1, 1)
            if seq_oh is not None:
                seq_oh = seq_oh.repeat(num_samples_per_input, 1, 1)
        B = B_real * num_samples_per_input

        # 3) Density hint
        # density_guided=True: 先做一次 t=0.5 的 forward 估计 density,
        #   再把它当条件喂给后续 sampling。这只在训练时启用了 density_hint_dropout
        #   (让模型学会 hint=None 也能预测) 时才靠谱。
        # density_guided=False: 完全不用 density hint，等价于 v1 sample。
        if density_guided:
            x_init = (torch.rand(B, 1, L, L, device=device) < self.rho_0).float()
            x_init = symmetrize_binary(x_init) * contact_mask
            t_half = torch.full((B,), 0.5, device=device)
            _, density_pred = self.backbone(
                x_init, t_half,
                mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
                mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=seq_oh,
                contact_masks=contact_mask, density_hint=None,
                return_density=True)
        else:
            x_init = (torch.rand(B, 1, L, L, device=device) < self.rho_0).float()
            x_init = symmetrize_binary(x_init) * contact_mask
            density_pred = None

        # 4) Cosine τ-leap schedule
        raw = [math.sin(math.pi * (k + 0.5) / (2 * num_steps)) for k in range(num_steps)]
        total_raw = sum(raw)
        dt_list = [r / total_raw for r in raw]

        x_t = x_init
        p_x1_last = torch.zeros_like(x_t)
        t_cum = 0.0

        for k in range(num_steps):
            dt = dt_list[k]
            t_tensor = torch.full((B,), t_cum, device=device)
            logit = self.backbone(
                x_t, t_tensor,
                mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
                mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=seq_oh,
                contact_masks=contact_mask,
                density_hint=density_pred if density_guided else None,
                return_density=False)
            logit = symmetrize_logit(logit)
            p_x1 = torch.sigmoid(logit)
            p_x1 = 0.5 * (p_x1 + p_x1.transpose(-2, -1))
            p_x1_last = p_x1

            rate_01, rate_10 = compute_ctmc_rates(x_t, p_x1, t_tensor, rho_0=self.rho_0)
            # density-guided rate damping
            if density_guided:
                damp = (2.0 * density_pred).clamp(max=1.0).view(B, 1, 1, 1)
                rate_01 = rate_01 * damp

            f01 = torch.clamp(rate_01 * dt, max=1.0)
            f10 = torch.clamp(rate_10 * dt, max=1.0)
            flip01 = (torch.rand_like(f01) < f01) & (x_t < 0.5)
            flip10 = (torch.rand_like(f10) < f10) & (x_t > 0.5)
            x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
            x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
            x_t = symmetrize_binary(x_t) * contact_mask
            t_cum += dt

        x_final = project_to_valid_contact_map(x_t, p_x1_last, contact_mask)

        # Multi-sample voting: project each, then average + threshold + symmetrize
        if num_samples_per_input > 1:
            x_final = x_final.view(num_samples_per_input, B_real, 1, L, L).mean(0)
            x_final = (x_final > 0.5).float()
            x_final = symmetrize_binary(x_final)
            p_x1_last = p_x1_last.view(num_samples_per_input, B_real, 1, L, L).mean(0)

        return x_final, p_x1_last
