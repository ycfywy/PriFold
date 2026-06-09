# PriFold-SymFlow v1 (归档)

> 这是 SymFlow 的早期 baseline 实现，保留用于对照。详细实现细节见 `docs/prifold_symflow_implementation_report.md`、状态评审见 `docs/prifold_symflow_status_review.md`。

## 关键特征

- **主干**：6 层标准 Axial DiT（行/列注意力 + GELU FFN + AdaLN(t)）
- **特征**：x_t embedding (8) + MARS-LX 最后一层 hidden 经 outer concat (128) + seq_oh 外积 (8) + pos_bias (1) = 145 通道
- **训练范围**：bpRNA TR0 + RNAStrAlign tr **合并训练**（已知此设置导致 bpRNA-test 严重崩塌）
- **patch_size**：4
- **采样**：均匀 Euler-CTMC，20 步

## 实测结果

`prifold_symflow_v1_full` 任务 best.pt @ epoch 51（共训 53/60 epoch）：

| 数据集 | F1 | P | R | MCC |
|---|---:|---:|---:|---:|
| val (bpRNA-VL0 + RNAStrAlign-vl) | 0.5793 | — | — | — |
| bpRNA-test | 0.2582 | 0.2134 | 0.3612 | 0.2668 |
| rnastralign-test | 0.7532 | 0.6776 | 0.8542 | 0.7585 |
| archiveii-test | 0.5097 | 0.4461 | 0.6020 | 0.5147 |

## 启动

```bash
cd /root/aigame/dannyyan/PriFold
python symfold/v1/train.py symfold/config/v1/prifold_symflow_v1_full.json
```

## v2 改进点（见 `symfold/v2/README.md`）

1. 主干升级为 SF v5 风格 DA-SE-DiT：9 层 Dilated Axial + Triangle Update + SwiGLU + RoPE + AdaLN-Zero
2. 特征升级：MARS 多层 attention map（72 个）替代 outer concat hidden
3. density 闭环：训练注入 GT、推理引导
4. **训练改成两个独立模型**（bpRNA-only 和 RNAStrAlign-only），对齐 PriFold 主线设置
