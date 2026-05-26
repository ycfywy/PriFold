"""
示例 1: RNA 序列预处理 + pos_bias 生物先验矩阵

对应 PriFold 中的:
- inference.py: collate_fn()
- utils/tools.py: get_posbias()

重点理解:
1. RNA 序列内部会把 U 替换成 T
2. 根据碱基配对规则生成 L×L 的位置偏置矩阵
3. A-T / G-C / G-T 会获得更高 bias
"""

from __future__ import annotations

import numpy as np
import torch


BASE_TO_ID = {
    "<pad>": 0,
    "<cls>": 1,
    "<eos>": 2,
    "A": 3,
    "T": 4,
    "G": 5,
    "C": 6,
    "N": 7,
}

PAIR_SCORES = {
    "AT": 3,
    "TA": 3,
    "GC": 6,
    "CG": 6,
    "GT": 1,
    "TG": 1,
}


def normalize_rna(seq: str) -> str:
    """PriFold 内部使用 T 表示 RNA 的 U。"""
    return seq.upper().replace("U", "T")


def tokenize(seq: str) -> torch.LongTensor:
    """极简 tokenizer：加入 <cls>/<eos>，并把碱基映射成 id。"""
    seq = normalize_rna(seq)
    ids = [BASE_TO_ID["<cls>"]]
    ids.extend(BASE_TO_ID.get(base, BASE_TO_ID["N"]) for base in seq)
    ids.append(BASE_TO_ID["<eos>"])
    return torch.LongTensor(ids)


def get_posbias(seqs: list[str], max_len: int, scale: float = 0.01) -> torch.Tensor:
    """
    简化复刻 utils/tools.py:get_posbias()

    Args:
        seqs: 原始 RNA 序列列表，例如 ["AUGC"]
        max_len: token 后长度，即 max(len(seq)+2)
        scale: 生物先验缩放因子

    Returns:
        pos_bias: (B, max_len, max_len)
    """
    normalized = [normalize_rna(seq) for seq in seqs]
    posbias = np.ones((len(seqs), max_len - 2, max_len - 2), dtype=np.float32)

    for batch_idx, seq in enumerate(normalized):
        seq_arr = np.array(list(seq))
        for pair, score in PAIR_SCORES.items():
            row_mask = seq_arr == pair[0]
            col_mask = seq_arr == pair[1]
            posbias[batch_idx, : len(seq), : len(seq)] += np.outer(row_mask, col_mask) * (score * scale)

    # 给 <cls>/<eos> 两侧 padding，保持和 tokenizer 输出对齐
    posbias = np.pad(posbias, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    return torch.tensor(posbias)


def print_matrix(seq: str, mat: torch.Tensor) -> None:
    """只打印真实碱基区域，不打印 <cls>/<eos>。"""
    seq = normalize_rna(seq)
    inner = mat[1 : len(seq) + 1, 1 : len(seq) + 1]

    print("      " + "  ".join(seq))
    for base, row in zip(seq, inner):
        values = "  ".join(f"{x:.2f}" for x in row.tolist())
        print(f"{base}  {values}")


if __name__ == "__main__":
    seq = "AUGCGAU"
    token_ids = tokenize(seq)
    pos_bias = get_posbias([seq], max_len=len(token_ids), scale=0.01)[0]

    print(f"原始 RNA:    {seq}")
    print(f"内部序列:    {normalize_rna(seq)}")
    print(f"token ids:   {token_ids.tolist()}")
    print(f"pos_bias形状: {tuple(pos_bias.shape)}")
    print("\n真实碱基区域的 pos_bias:")
    print_matrix(seq, pos_bias)

    print("\n解释:")
    print("1. 普通位置 bias=1.00")
    print("2. A-T/T-A 位置 bias=1.03")
    print("3. G-C/C-G 位置 bias=1.06")
    print("4. G-T/T-G 位置 bias=1.01")
