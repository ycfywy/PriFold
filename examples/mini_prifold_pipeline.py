"""
示例 5: MiniPriFold — 把主体结构拆成可运行的最小流水线

这个文件把 PriFold 的主体结构压缩成一个小模型，方便理解数据流：

RNA序列
  -> tokenizer
  -> TinyMARSLikeEncoder         # 类比 MARS/LLaMA2 语言模型
  -> PairwiseOnly                # 1D 序列特征转 L×L 配对特征
  -> AxialAttentionStack         # 类比 RNAformerStack
  -> ContactMapHead              # Linear(D -> 1)
  -> sigmoid + threshold         # 得到 contact map

注意：这是教学示例，随机初始化，输出不代表真实 RNA 二级结构预测。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from axial_attention import AxialAttentionStack


BASE_TO_ID = {"<pad>": 0, "A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
PAIR_SCORES = {"AT": 3, "TA": 3, "GC": 6, "CG": 6, "GT": 1, "TG": 1}


def normalize_rna(seq: str) -> str:
    return seq.upper().replace("U", "T")


def batch_encode(seqs: list[str]) -> tuple[torch.LongTensor, torch.LongTensor, list[str]]:
    normalized = [normalize_rna(seq) for seq in seqs]
    max_len = max(len(seq) for seq in normalized)

    input_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)

    for b, seq in enumerate(normalized):
        ids = [BASE_TO_ID.get(base, BASE_TO_ID["N"]) for base in seq]
        input_ids[b, : len(ids)] = torch.LongTensor(ids)
        attention_mask[b, : len(ids)] = 1

    return input_ids, attention_mask, normalized


def get_posbias(seqs: list[str], scale: float = 0.01) -> torch.Tensor:
    normalized = [normalize_rna(seq) for seq in seqs]
    B = len(normalized)
    L = max(len(seq) for seq in normalized)
    bias = torch.ones(B, L, L)

    for b, seq in enumerate(normalized):
        for i, left in enumerate(seq):
            for j, right in enumerate(seq):
                score = PAIR_SCORES.get(left + right, 0)
                bias[b, i, j] += score * scale

        # padding 区域置 0，避免误导注意力
        if len(seq) < L:
            bias[b, len(seq) :, :] = 0
            bias[b, :, len(seq) :] = 0

    return bias.unsqueeze(1)  # (B, 1, L, L)


class TinyMARSLikeEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64, num_heads: int = 4):
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return self.norm(x)


class PairwiseOnly(nn.Module):
    def __init__(self, lm_dim: int, pair_dim: int):
        super().__init__()
        self.proj = nn.Linear(lm_dim, pair_dim)

    def forward(self, seq_hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq_hidden)
        B, L, D = x.shape
        left = x.unsqueeze(2).expand(B, L, L, D)
        right = x.unsqueeze(1).expand(B, L, L, D)
        return torch.cat([left, right], dim=-1)


class ContactMapHead(nn.Module):
    def __init__(self, pair_dim: int):
        super().__init__()
        self.proj = nn.Linear(pair_dim, 1)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.proj(pair_features).squeeze(-1)


class MiniPriFold(nn.Module):
    def __init__(self, lm_dim: int = 64, pair_dim: int = 32, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.encoder = TinyMARSLikeEncoder(vocab_size=len(BASE_TO_ID), hidden_dim=lm_dim, num_heads=num_heads)
        self.pairwise = PairwiseOnly(lm_dim=lm_dim, pair_dim=pair_dim)
        self.rnaformer = AxialAttentionStack(dim=pair_dim * 2, num_heads=num_heads, num_layers=num_layers)
        self.head = ContactMapHead(pair_dim=pair_dim * 2)

    def forward(self, seqs: list[str]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask, normalized = batch_encode(seqs)
        pos_bias = get_posbias(normalized)

        hidden = self.encoder(input_ids, attention_mask)
        pair_features = self.pairwise(hidden)
        refined_pair_features = self.rnaformer(pair_features, bias=pos_bias)
        logits = self.head(refined_pair_features)

        probs = torch.sigmoid(logits)
        contact_map = (probs > 0.45).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden": hidden,
            "pair_features": pair_features,
            "refined_pair_features": refined_pair_features,
            "logits": logits,
            "probs": probs,
            "contact_map": contact_map,
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    seqs = ["AUGCGAU"]
    model = MiniPriFold()
    outputs = model(seqs)

    print(f"输入序列:                  {seqs[0]}")
    print(f"input_ids shape:           {tuple(outputs['input_ids'].shape)}")
    print(f"hidden shape:              {tuple(outputs['hidden'].shape)}")
    print(f"pair_features shape:       {tuple(outputs['pair_features'].shape)}")
    print(f"refined_pair_features shape:{tuple(outputs['refined_pair_features'].shape)}")
    print(f"logits shape:              {tuple(outputs['logits'].shape)}")
    print(f"contact_map shape:         {tuple(outputs['contact_map'].shape)}")
    print("\n随机初始化模型的 contact_map（仅看形状和流程）:")
    print(outputs["contact_map"][0].int())
