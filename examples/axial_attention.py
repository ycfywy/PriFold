"""
轴向注意力 (Axial Attention) 示例实现

演示如何将 2D 特征图 (L×L) 的全注意力 O(L^4) 分解为
行注意力 + 列注意力，降低复杂度至 O(L^3)。

这是 PriFold 中 RNAformerBlock 使用的核心注意力机制的简化版本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
    """
    轴向注意力：对 2D 特征图的某一轴方向做标准多头注意力。

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        orientation: 'row' 或 'column'，指定沿哪个轴做注意力
    """

    def __init__(self, dim: int, num_heads: int = 4, orientation: str = 'row'):
        super().__init__()
        assert dim % num_heads == 0
        assert orientation in ('row', 'column')

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.orientation = orientation

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, L, D) — 2D 配对特征图
            bias: (B, L, L) 或 (B, 1, L, L) — 可选的注意力偏置（如生物先验 pos_bias）

        Returns:
            (B, L, L, D) — 注意力输出
        """
        B, L1, L2, D = x.shape

        # 列注意力：转置行列，复用行注意力逻辑
        if self.orientation == 'column':
            x = x.transpose(1, 2)  # (B, L2, L1, D)

        # 此时统一视为：(B, num_seqs, seq_len, D)
        # 对每个 "序列"（行）内部的 seq_len 个位置做注意力
        residual = x
        x = self.norm(x)

        # QKV 投影
        qkv: torch.Tensor = self.qkv(x)  # (B, num_seqs, seq_len, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # 拆分多头: (B, num_seqs, seq_len, num_heads, head_dim)
        q = q.view(B, L1, L2, self.num_heads, self.head_dim)
        k = k.view(B, L1, L2, self.num_heads, self.head_dim)
        v = v.view(B, L1, L2, self.num_heads, self.head_dim)

        # 调整维度: (B, num_seqs, num_heads, seq_len, head_dim)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # 注意力得分: (B, num_seqs, num_heads, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注入偏置（如生物先验 pos_bias）
        # PriFold 的 Attention2d 是 softmax 后做乘法: attn_weights * bias
        # bias 需要广播到 (B, num_seqs, num_heads, seq_len, seq_len)
        if bias is not None:
            # bias: (B, 1, L, L) → (B, 1, 1, L, L) 广播到所有 num_seqs 和 num_heads
            if bias.dim() == 4:
                bias = bias.unsqueeze(2)  # (B, 1, 1, L, L)
            elif bias.dim() == 3:
                bias = bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L, L)
            attn = attn * bias

        # 加权求和
        out = torch.matmul(attn, v)  # (B, num_seqs, num_heads, seq_len, head_dim)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, L1, L2, D)

        out = self.out_proj(out)
        out = residual + out

        # 列注意力：转回原始布局
        if self.orientation == 'column':
            out = out.transpose(1, 2)

        return out


class AxialAttentionBlock(nn.Module):
    """
    完整的轴向注意力块：行注意力 → 列注意力 → FFN

    对应 PriFold 中的 RNAformerBlock。
    """

    def __init__(self, dim: int, num_heads: int = 4, ffn_mult: float = 4.0):
        super().__init__()
        self.row_attn = AxialAttention(dim, num_heads, orientation='row')
        self.col_attn = AxialAttention(dim, num_heads, orientation='column')

        # FFN
        hidden_dim = int(dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, L, D)
            bias: (B, 1, L, L) 可选注意力偏置
        """
        x = self.row_attn(x, bias)
        x = self.col_attn(x, bias)
        x = x + self.ffn(x)
        return x


class AxialAttentionStack(nn.Module):
    """
    多层轴向注意力堆叠。

    对应 PriFold 中的 RNAformerStack，
    其中 pos_bias 仅注入第 1 层。
    """

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            AxialAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, L, D)
            bias: (B, 1, L, L) — 仅注入第 1 层
        """
        for i, layer in enumerate(self.layers):
            layer_bias = bias if i == 0 else None  # 仅第1层使用 bias
            x = layer(x, layer_bias)
        return x


# ==================== 使用示例 ====================

if __name__ == "__main__":
    torch.manual_seed(42)

    # 参数设置
    batch_size = 2
    seq_len = 8       # RNA 序列长度 (示例用短序列)
    dim = 64          # 特征维度
    num_heads = 4
    num_layers = 4

    # 模拟输入: 2D 配对特征图 (B, L, L, D)
    # 在 PriFold 中，这是 PairwiseOnly 模块输出的外积拼接结果
    pair_features = torch.randn(batch_size, seq_len, seq_len, dim)

    # 模拟生物先验 pos_bias (B, 1, L, L)
    # 在 PriFold 中由 get_posbias() 根据碱基配对规则生成
    pos_bias = torch.ones(batch_size, 1, seq_len, seq_len) * 0.01

    # 构建模型
    model = AxialAttentionStack(dim=dim, num_heads=num_heads, num_layers=num_layers)

    # 前向传播
    output = model(pair_features, bias=pos_bias)

    print(f"输入形状:  {pair_features.shape}")  # (2, 8, 8, 64)
    print(f"输出形状:  {output.shape}")          # (2, 8, 8, 64)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 对比: 轴向注意力 vs 全注意力的复杂度 =====
    L = 500  # 典型 RNA 长度
    full_attn_ops = L ** 4           # 全注意力
    axial_attn_ops = 2 * (L ** 3)   # 行 + 列注意力

    print(f"\n===== 复杂度对比 (L={L}) =====")
    print(f"全注意力:   {full_attn_ops:>15,} ops  O(L^4)")
    print(f"轴向注意力: {axial_attn_ops:>15,} ops  O(L^3)")
    print(f"加速比:     {full_attn_ops / axial_attn_ops:.0f}x")
