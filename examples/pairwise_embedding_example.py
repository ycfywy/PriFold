"""
示例 2: PairwiseOnly — 从 1D 序列特征构造 2D 配对特征

对应 PriFold 中的:
- utils/RNAformer/model/Riboformer_outfirst.py: PairwiseOnly

重点理解:
MARS 语言模型输出的是每个位置的 1D 表示 (B, L, D_lm)，
但二级结构预测需要判断每一对碱基 (i, j)，所以要构造 L×L 的 2D 表示。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseOnly(nn.Module):
    """
    简化版 PairwiseOnly。

    输入:
        seq_hidden: (B, L, lm_dim)

    输出:
        pair_features: (B, L, L, 2 * pair_dim)

    对每个位置对 (i, j):
        pair_features[:, i, j] = concat(project(h_i), project(h_j))
    """

    def __init__(self, lm_dim: int, pair_dim: int):
        super().__init__()
        self.proj = nn.Linear(lm_dim, pair_dim)

    def forward(self, seq_hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq_hidden)  # (B, L, pair_dim)
        B, L, D = x.shape

        left = x.unsqueeze(2).expand(B, L, L, D)   # 第 i 个碱基的特征
        right = x.unsqueeze(1).expand(B, L, L, D)  # 第 j 个碱基的特征

#         left[0] = [
#   [[1, 2], [1, 2], [1, 2]],   # 第0行，全是第0个碱基的向量
#   [[3, 4], [3, 4], [3, 4]],   # 第1行，全是第1个碱基的向量
#   [[5, 6], [5, 6], [5, 6]],   # 第2行，全是第2个碱基的向量
# ]
#         right[0] = [
#   [[1, 2], [3, 4], [5, 6]],   # 第0列，全是第0个碱基的向量
#   [[1, 2], [3, 4], [5, 6]],   # 第1列，全是第1个碱基的向量
#   [[1, 2], [3, 4], [5, 6]],   # 第2列，全是第2个碱基的向量
# ]
# pair[0] = [
#   [[1, 2, 1, 2], [1, 2, 3, 4], [1, 2, 5, 6]],
#   [[3, 4, 1, 2], [3, 4, 3, 4], [3, 4, 5, 6]],
#   [[5, 6, 1, 2], [5, 6, 3, 4], [5, 6, 5, 6]],
# ]


        return torch.cat([left, right], dim=-1)    # (B, L, L, 2D)


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 5
    lm_dim = 12
    pair_dim = 4

    # 模拟 MARS 的输出: 每个碱基一个 lm_dim 维向量
    seq_hidden = torch.randn(batch_size, seq_len, lm_dim)

    module = PairwiseOnly(lm_dim=lm_dim, pair_dim=pair_dim)
    pair_features = module(seq_hidden)

    print(f"序列特征 shape: {tuple(seq_hidden.shape)}")
    print(f"配对特征 shape: {tuple(pair_features.shape)}")

    # 验证 pair[i,j] 的前半部分来自 i，后半部分来自 j
    projected = module.proj(seq_hidden)
    i, j = 1, 3
    pair_ij = pair_features[0, i, j]

    print(f"\n查看 pair[{i},{j}]：")
    print("前半部分是否等于 projected[i]:", torch.allclose(pair_ij[:pair_dim], projected[0, i]))
    print("后半部分是否等于 projected[j]:", torch.allclose(pair_ij[pair_dim:], projected[0, j]))

    print("\n直观理解:")
    print("pair[i,j] 同时携带第 i 个碱基和第 j 个碱基的信息，")
    print("后续 RNAformer 就可以判断这两个位置是否应该配对。")
