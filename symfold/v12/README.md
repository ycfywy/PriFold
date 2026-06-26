# v12: Flow Matching + DiT for RNA Secondary Structure Prediction

## 一句话

用 **Conditional Flow Matching** 学习从高斯噪声到 RNA contact map 的连续流，**DiT (Diffusion Transformer)** 作为骨干网络预测流场。

---

## 核心思想

```
噪声 x₀ ~ N(0,1)  ───── Flow Field v_θ ─────→  Contact Map x₁
     (L×L)              (DiT predicts)              (L×L binary)
```

**训练**：给定 GT contact map x₁，随机采样 t∈[0,1]，构造插值 x_t = (1-t)·x₀ + t·x₁，让 DiT 学会预测 x₁（或速度 v = x₁ - x₀）。

**推理**：从 x₀ ~ N(0,1) 出发，用 Euler 积分沿着学到的流走 N 步到达 x₁，sigmoid 后阈值化得到二值 contact map。

---

## 架构

```
RNA Sequence
    │
    ▼
┌────────────────────────────┐
│  MARS-LX (160M, Encoder)   │  ← 预训练 RNA 语言模型
│  → hidden (B, L, 1056)     │
│  → attention (B, 6, 12, L, L) │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  MARSConditioner            │  ← 投影为 pair 表征
│  → pair_cond (B, L, L, C)  │
└────────────┬───────────────┘
             │
    x_t ─────┤   t ──┐
             │       │
             ▼       ▼
┌────────────────────────────┐
│  Input Proj: [x_t; cond]→D │
│                            │
│  ┌──────────────────────┐  │
│  │  DiT Block × N       │  │
│  │  ┌────────────────┐  │  │
│  │  │ AdaLN-Zero(t)  │  │  │  ← time conditioning
│  │  │ Row Attention   │  │  │
│  │  │ Col Attention   │  │  │
│  │  │ FFN             │  │  │
│  │  └────────────────┘  │  │
│  └──────────────────────┘  │
│                            │
│  Final AdaLN + Linear → 1  │
│  Symmetrize                │
└────────────┬───────────────┘
             │
             ▼
        pred (B, L, L)
        = predicted x₁ or velocity
```

---

## 为什么是 Flow Matching + DiT？

### vs 判别式 (v9/v10)

| | 判别式 | Flow Matching (v12) |
|---|---|---|
| 输出 | 单次前向 → probability | 迭代采样 → 结构 |
| 本质 | 学 P(contact\|seq) | 学噪声→结构的映射 |
| 多样性 | 无 | 不同噪声 → 不同结构 |
| 不确定性 | 无 | 多次采样的方差 |
| 全局一致性 | 无保证 | Flow 天然鼓励全局协调 |

### vs Diffusion (DDPM/DDIM)

| | Diffusion | Flow Matching |
|---|---|---|
| 路径 | 固定 noise schedule | **直线**最优传输路径 |
| 采样步数 | 通常 100-1000 步 | **10-50 步**即可 |
| 训练目标 | 预测噪声 ε | 预测速度 v 或目标 x₁ |
| 理论 | Score matching | **ODE flow** |

### 为什么 DiT？

- Transformer 天然处理 2D 结构化数据（row/col attention）
- AdaLN-Zero 是最高效的 conditioning 方式（零初始化保证训练稳定）
- 已被 ImageGen (DiT, SD3, Flux) 验证为最强生成骨干

---

## 与 v6 的区别

| | v6 | v12 |
|---|---|---|
| Flow 类型 | Discrete (CTMC, binary) | **Continuous** (OT interpolation) |
| 采样 | Tau-leap (随机翻转) | **Euler ODE** (确定性) |
| 状态空间 | {0, 1}^{L×L} | **ℝ^{L×L}** (连续) |
| Loss | 9 个组件 | **1 个 MSE** |
| 骨干 | Dilated Axial + Patch + 三头 | **纯 DiT** |
| 代码行数 | ~800 行 (3 文件) | **~250 行 (1 文件)** |

关键转变：**从离散流到连续流**。v6 的 per-position 独立翻转导致位置漂移；v12 在连续空间做 ODE 积分，路径更平滑，结果更精确。

---

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_dim | 256 | DiT token 维度 |
| num_heads | 8 | 注意力头数 |
| num_layers | 8 | DiT block 数量 |
| ff_mult | 4 | FFN 扩展倍数 |
| dropout | 0.1 | Dropout |
| sigma_min | 1e-4 | 最小噪声尺度 |
| prediction_type | 'x1' | 预测目标: 'x1' 或 'velocity' |
| num_steps (推理) | 50 | Euler 积分步数 |
| threshold (推理) | 0.5 | 二值化阈值 |

---

## 使用

```python
from symfold.v12.model import RNAFlowDiT
from utils.lm import get_extractor

extractor, tokenizer = get_extractor(args)

model = RNAFlowDiT(
    extractor=extractor,
    freeze_mars=True,
    hidden_dim=256,
    num_heads=8,
    num_layers=8,
)

# Training
loss, loss_dict = model(batch)

# Inference (50-step Euler sampling)
pred_binary, pred_prob = model.sample(batch, num_steps=50, threshold=0.5)
```

---

## 文件结构

```
symfold/v12/
├── README.md    ← 本文档
├── __init__.py
└── model.py     ← 完整模型 (~250行)
```

---

## 创新点总结

1. **首次将 Continuous Flow Matching 应用于 RNA 二级结构生成式预测**
2. **DiT (AdaLN-Zero Transformer) 作为 2D contact map 的生成骨干**
3. **MARS 语言模型提供序列级条件，通过 pair projection 注入**
4. **连续空间 ODE 积分替代离散 tau-leap，消除位置漂移**
5. **极简训练目标 (单一 MSE loss)，无需复杂多任务平衡**
