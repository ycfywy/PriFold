"""
示例 4: ContactMapHead — 从 2D 配对特征得到二级结构接触图

对应 PriFold 中的:
- RiboFormer 最后的输出层: Linear(256 -> 1)
- inference.py 中的 sigmoid + threshold=0.45

重点理解:
模型最终不是直接输出 dot-bracket，而是输出 L×L 的 logits。
logits 经过 sigmoid 得到配对概率，再通过阈值得到二值 contact map。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContactMapHead(nn.Module):
    """把每个 (i,j) 的 pair feature 映射成一个配对 logit。"""

    def __init__(self, pair_dim: int):
        super().__init__()
        self.proj = nn.Linear(pair_dim, 1)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        # pair_features: (B, L, L, D)
        return self.proj(pair_features).squeeze(-1)  # (B, L, L)


def logits_to_contact_map(
    logits: torch.Tensor,
    threshold: float = 0.45,
    symmetrize: bool = True,
    remove_diagonal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits -> probabilities -> binary contact map。"""
    probs = torch.sigmoid(logits)

    if symmetrize:
        probs = (probs + probs.transpose(-1, -2)) / 2

    if remove_diagonal:
        L = probs.size(-1)
        eye = torch.eye(L, dtype=torch.bool, device=probs.device)
        probs = probs.masked_fill(eye, 0.0)

    contact_map = (probs > threshold).float()
    return probs, contact_map


def contact_map_to_pairs(contact_map: torch.Tensor) -> list[tuple[int, int]]:
    """把 L×L contact map 转成 i<j 的配对列表。"""
    pairs = []
    L = contact_map.size(0)
    for i in range(L):
        for j in range(i + 1, L):
            if contact_map[i, j] > 0:
                pairs.append((i, j))
    return pairs


def pairs_to_dotbracket(length: int, pairs: list[tuple[int, int]]) -> str:
    """仅演示无假结情况下的 dot-bracket 转换。"""
    chars = ["."] * length
    used = set()
    for i, j in pairs:
        if i in used or j in used:
            continue
        chars[i] = "("
        chars[j] = ")"
        used.add(i)
        used.add(j)
    return "".join(chars)


if __name__ == "__main__":
    torch.manual_seed(0)

    seq = "AUGCGA"
    L = len(seq)
    pair_dim = 32

    # 1. 真实模型里 logits 来自 ContactMapHead(pair_features)
    pair_features = torch.randn(1, L, L, pair_dim)
    head = ContactMapHead(pair_dim)
    random_logits = head(pair_features)
    print(f"随机 pair_features shape: {tuple(pair_features.shape)}")
    print(f"head 输出 logits shape:   {tuple(random_logits.shape)}")

    # 2. 为了更直观，手工构造几个高 logit 位置演示后处理
    manual_logits = torch.full((L, L), -4.0)
    manual_logits[0, 5] = manual_logits[5, 0] = 4.0
    manual_logits[1, 4] = manual_logits[4, 1] = 3.0
    manual_logits[2, 3] = manual_logits[3, 2] = 2.0

    probs, contact_map = logits_to_contact_map(manual_logits, threshold=0.45)
    pairs = contact_map_to_pairs(contact_map)
    dotbracket = pairs_to_dotbracket(L, pairs)

    print(f"\n序列:       {seq}")
    print(f"预测配对:   {pairs}")
    print(f"dot-bracket:{dotbracket}")
    print("\n概率矩阵:")
    print(torch.round(probs * 100) / 100)
    print("\n二值 contact map:")
    print(contact_map.int())
