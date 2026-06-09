"""MARS forward with multi-layer attention map extraction.

不修改 prifold/llama2.py 原文件，通过 wrapper 函数为已加载的 MARS Transformer 实例
提供"返回最后 N 层 attention 权重"的能力。

核心思路：MARS 的 Attention 模块默认走 torch.nn.functional.scaled_dot_product_attention
(flash-attn 路径) 不返回 attn weights。这里我们手写 manual softmax 路径并对最后 N 层走
该路径，前面的层仍走 flash-attn 保速。
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from prifold.llama2 import apply_rotary_emb, repeat_kv
from prifold.utils.attn_utils import get_extended_attention_mask


def _manual_attention(layer_attn, x, attn_mask, freqs_cos, freqs_sin):
    """Manual softmax self-attention path that also returns attention weights.

    Mirrors ``prifold.llama2.Attention.forward`` (encoder, no kv_state) but
    keeps the post-softmax attention probabilities for export.

    Returns
    -------
    output : (B, L, D)  attention output (same as the layer's normal output)
    attn   : (B, n_heads, L, L)  softmax attention probabilities (after dropout=0)
    """
    bsz, seqlen, _ = x.shape

    xq = layer_attn.wq(x)
    xk = layer_attn.wk(x)
    xv = layer_attn.wv(x)

    xq = xq.view(bsz, seqlen, layer_attn.n_local_heads, layer_attn.head_dim)
    xk = xk.view(bsz, seqlen, layer_attn.n_local_kv_heads, layer_attn.head_dim)
    xv = xv.view(bsz, seqlen, layer_attn.n_local_kv_heads, layer_attn.head_dim)

    # RoPE
    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    # GQA expand
    xk = repeat_kv(xk, layer_attn.n_rep)
    xv = repeat_kv(xv, layer_attn.n_rep)

    # (B, H, L, D)
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(layer_attn.head_dim)
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = F.softmax(scores.float(), dim=-1).type_as(xq)
    # 不在 attention 输出上 dropout（推理用），但 output 上保持
    output = torch.matmul(attn, xv)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    output = layer_attn.wo(output)
    output = layer_attn.resid_dropout(output)
    return output, attn


def mars_forward_with_attn(model, tokens, attn_mask, n_attn_layers: int = 6,
                           hidden_layer_indices: list[int] | None = None,
                           return_hidden_layers: bool = False):
    """Run MARS encoder forward, returning hidden + last-N attention.

    Parameters
    ----------
    hidden_layer_indices:
        1-based transformer layer indices to return. For MARS-LX (12 layers),
        [3, 6, 9, 12] mirrors RNA-FM/SymFold's multi-layer representation usage.
    return_hidden_layers:
        If True, return (hidden, attn_stack, hidden_layers); otherwise keep the
        original two-return API.
    """
    assert not model.is_decoder, "MARS encoder expected (is_decoder=False)"
    bsz, seqlen = tokens.shape

    if hidden_layer_indices is None:
        hidden_layer_indices = []
    hidden_layer_set = set(hidden_layer_indices)

    h = model.tok_embeddings(tokens)
    if attn_mask is not None:
        h = (h * attn_mask.unsqueeze(-1)).to(h.dtype)
    h = model.dropout(h)

    freqs_cos = model.freqs_cos[:seqlen]
    freqs_sin = model.freqs_sin[:seqlen]

    if attn_mask is not None:
        ext_mask = get_extended_attention_mask(
            attn_mask, (bsz, seqlen), h.device, h.dtype, is_decoder=False)
    else:
        ext_mask = None

    n_layers = len(model.layers)
    manual_start = max(0, n_layers - n_attn_layers)

    attn_list = []
    hidden_list = []
    for idx, layer in enumerate(model.layers):
        layer_no = idx + 1  # 1-based, same convention as RNA-FM repr_layers
        if idx < manual_start:
            # Fast path: layer's own forward (flash-attn), no attention export.
            h, _ = layer(h, ext_mask, freqs_cos, freqs_sin, None)
        else:
            # Manual path mirroring TransformerBlock.forward exactly.
            attn_out, attn_weights = _manual_attention(
                layer.attention, layer.attention_norm(h),
                ext_mask, freqs_cos, freqs_sin)
            h = h + attn_out
            h = h + layer.feed_forward(layer.ffn_norm(h))
            attn_list.append(attn_weights)
        if layer_no in hidden_layer_set:
            # Store pre-final-norm layer representation, matching typical LM feature use.
            hidden_list.append(h)

    h = model.norm(h)
    attn_stack = torch.stack(attn_list, dim=1)  # (B, n_attn_layers, n_heads, L, L)
    if return_hidden_layers:
        if not hidden_list:
            hidden_list = [h]
        return h, attn_stack, hidden_list
    return h, attn_stack
