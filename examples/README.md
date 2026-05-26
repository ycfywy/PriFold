# PriFold Block Examples

这些示例把 PriFold 主体结构拆成多个小块，建议按顺序阅读和运行。

## 阅读顺序

1. `tokenize_posbias_example.py`
   - RNA 序列预处理
   - `U -> T`
   - 根据 A-T / G-C / G-T 规则生成 `pos_bias`

2. `tiny_mars_encoder_example.py`
   - 用一个小型 Transformer Encoder 类比 MARS 语言模型
   - 演示 `token ids -> hidden states`

3. `pairwise_embedding_example.py`
   - 类比 `PairwiseOnly`
   - 演示如何从 `(B, L, D)` 构造 `(B, L, L, 2D)` 配对特征

4. `axial_attention.py`
   - 类比 RNAformer 的轴向注意力
   - 演示行注意力 + 列注意力
   - `pos_bias` 只注入第 1 层

5. `contact_map_head_example.py`
   - 类比最终输出头
   - 演示 `logits -> sigmoid -> threshold -> contact map`

6. `mini_prifold_pipeline.py`
   - 把以上模块串起来，形成一个可运行的 MiniPriFold 流水线

## 运行方式

在项目根目录运行：

```bash
python examples/tokenize_posbias_example.py
python examples/tiny_mars_encoder_example.py
python examples/pairwise_embedding_example.py
python examples/axial_attention.py
python examples/contact_map_head_example.py
python examples/mini_prifold_pipeline.py
```

如果当前 shell 没有 `python`，先激活项目环境：

```bash
export PATH="/root/aigame/dannyyan/miniconda3/envs/prifold/bin:$PATH"
```

## 注意

这些示例用于理解结构和张量形状，模型都是随机初始化的，不代表真实预测效果。
