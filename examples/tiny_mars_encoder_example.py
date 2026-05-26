"""
示例 3: Tiny MARS-like Encoder — 语言模型提取序列上下文特征

对应 PriFold 中的:
- prifold/llama2.py: Transformer
- utils/lm.py: get_extractor()

这里不复刻完整 LLaMA2/MARS，只用 PyTorch TransformerEncoder
演示“token ids -> 每个碱基的上下文向量”这个核心接口。
"""

from __future__ import annotations

import torch
import torch.nn as nn


BASE_TO_ID = {"<pad>": 0, "A": 1, "T": 2, "G": 3, "C": 4, "N": 5}


def encode_sequence(seq: str) -> torch.LongTensor:
    seq = seq.upper().replace("U", "T")
    return torch.LongTensor([BASE_TO_ID.get(base, BASE_TO_ID["N"]) for base in seq])


class TinyMARSLikeEncoder(nn.Module):
    """
    极简 MARS-like Encoder。

    输入:
        input_ids: (B, L)
        attention_mask: (B, L), 1 表示有效 token，0 表示 padding

    输出:
        hidden_states: (B, L, hidden_dim)
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    seq = "AUGCGA"
    input_ids = encode_sequence(seq).unsqueeze(0)      # (1, L)
    attention_mask = torch.ones_like(input_ids)        # 无 padding

    encoder = TinyMARSLikeEncoder(vocab_size=len(BASE_TO_ID), hidden_dim=64)
    hidden = encoder(input_ids, attention_mask)

    print(f"RNA序列:       {seq}")
    print(f"内部token ids: {input_ids.tolist()}")
    print(f"hidden shape:  {tuple(hidden.shape)}")
    print("\n解释:")
    print("hidden[0, i] 是第 i 个碱基经过上下文建模后的向量。")
    print("PriFold 后续会把这些 1D 向量送进 PairwiseOnly，构造 L×L 配对特征。")
