# 模型训练对比分析：v6 vs v10 vs v12

## 一、模型范式概述

| 模型 | 训练范式 | Flow 类型 | 核心思路 |
|------|---------|-----------|---------|
| **v6** | 离散 Flow Matching (CTMC) | 二值 {0,1} 状态空间，tau-leap 采样 | Patch 下采样 + Dilated Axial Attention |
| **v10** | 监督判别式 (Focal BCE + Dice) | 无 flow，直接预测接触矩阵 | 全分辨率 Axial Attention + MARS 解冻 |
| **v12** | 连续 Flow Matching (OT) | 连续 ℝ 空间，Euler ODE 采样 | 全分辨率 Axial Attention + 极简 DiT |

---

## 二、模型架构参数对比

| 参数 | v6 | v10 | v12 |
|------|-----|------|------|
| 模型类 | DA-SE-DiT | DensityNetProPlus (v9) | FlowDiT |
| hidden_dim | 320 | 192 | 256 |
| num_layers | 12 | 8 | 8 |
| num_heads | 4 | 6 | 8 |
| dim_head | 80 | 32 | 32 |
| ff_mult | 4 | 4 | 4 |
| dropout | 0.1 | 0.2 | 0.1 |
| **patch_size** | **4** (下采样 4×) | **无** | **无** |
| Dilation pattern | [1,1,1,2,2,2,4,4,4,8,8,8] | 无 | 无 |
| Triangle updates | 有 (layer 4 起) | 无 | 无 |
| 位置编码 | AxialRoPE | RoPE2D | RoPE2D |
| Conditioning | AdaLN-Zero (time+MARS+density) | 无 (非生成式) | AdaLN-Zero (time) |
| 输出头 | 3个 (Flow+Direct+Density) | 2个 (Contact+Density) | 1个 (pred x₁) |
| **总参数** | 186.7M | 165.7M | 172.6M |
| **可训参数** | 26.1M | **165.7M (全部)** | 11.9M |
| MARS 冻结 | ✅ 冻结 | ❌ 解冻 | ✅ 冻结 |

---

## 三、训练超参数对比

| 参数 | v6 | v10 | v12 |
|------|-----|------|------|
| batch_size (配置) | 6 | 8 | 8 |
| **动态 batch** | 无 | ✅ `LengthBucketBatchSampler` | ✅ `LengthBucketBatchSampler` |
| **长序列(L≈490) batch** | 6 (patch后空间小) | **~1** (动态降到1) | **~1** (动态降到1) |
| gradient_accumulation | 1 | 3 | 8 |
| 等效 batch size | 6 | ~24 (短序列更大) | 动态 (短序列更大，长序列~1×8=8) |
| optimizer | AdamW | AdamW | AdamW |
| lr | 1.5e-4 | head: 1e-4 / MARS: 1e-6 | 3e-4 |
| weight_decay | 0.01 | 0.02 | 0.01 |
| grad_clip | 1.0 | 0.5 | 1.0 |
| AMP dtype | bf16 | bf16 | bf16 |
| epochs | 300 | 150+ | 100 |
| warmup_epochs | 8 | 3 | 5 |
| max_len_filter | 490 | 490 | 490 |
| **Gradient Checkpointing** | ❌ | ❌ | ✅ |
| 数据增强 | ✅ (select=0.1, replace=0.3) | ✅ (select=0.2, replace=0.4) | ❌ |

> **关于动态 batch**：v10 和 v12 都使用 `LengthBucketBatchSampler`，token budget 公式为：
> ```
> dynamic_bs = max(1, max_sq_tokens / L²)
> ```
> 其中 `max_sq_tokens = batch_size × median_len²`。当 L≈490 时，动态 batch 降为 1；短序列(L≈100) batch 可达 8+。
> v12 配置 batch_size=8，结合 gradient_accumulation=8，短序列的等效 batch 更大，长序列则为 1×8=8。

---

## 四、Attention 操作空间分析 — 训练速度差异的核心

### 4.1 v6: Patch 下采样后的 Dilated Axial Attention

```
输入: (B, L, L) → Patch Embed → (B, L/4, L/4, D=320)
Attention 序列长度: L/4 (而非 L)
Attention 矩阵: (B × L/4, H=4, L/4, L/4)

当 L=490:
  Patch 后: 122 × 122
  Row attn 矩阵: (B×122, 4, 122, 122) → 每层 ≈ B × 122 × 4 × 122² × 2B
  对 B=6: 6 × 122 × 4 × 122 × 122 × 2 ≈ 88MB/层
  12 层总计: ~1.1GB
```

### 4.2 v10: 全分辨率 Axial Attention (小 hidden_dim)

```
输入: (B, L, L, D=192)，无 patch
Attention 序列长度: L
Attention 矩阵: (B × L, H=6, L, L)

当 L=490, B=1 (动态降到1):
  Row attn: (1×490, 6, 490, 490) × 2B ≈ 2.8GB/层
  Col attn: 同上 ≈ 2.8GB/层
  8 层总计: ~45GB (单层峰值 ~5.6GB)

当 L=150 (中位数), B=8:
  Row attn: (8×150, 6, 150, 150) × 2B ≈ 324MB/层
  8 层总计: ~5.2GB
```

### 4.3 v12: 全分辨率 Axial Attention (大 hidden_dim + Gradient CP)

```
输入: (B=1, L, L, D=256)，无 patch
Attention 序列长度: L
Attention 矩阵: (B × L, H=8, L, L)

当 L=400, B=1:
  Row attn: (1×400, 8, 400, 400) × 2B ≈ 2.0GB/层
  Col attn: 同上 ≈ 2.0GB/层
  8 层总计 (无CP): ~32GB
  开启 Gradient Checkpointing 后: 峰值 ~10-12GB (只保留边界激活)

当 L=490, B=1:
  Row attn: (1×490, 8, 490, 490) × 2B ≈ 3.7GB/层
  8 层总计 (无CP): ~60GB → OOM!
```

### 4.4 核心公式

**单层 Axial Attention 显存公式 (bf16):**

```
V_attn = 2 × B × L × H × L × L × 2 bytes = 4 × B × H × L³ × 2 bytes
       = 8BHL³ bytes (row + col)
```

| 模型 | B | H | L_eff | V_attn/层 | 总 Attention 显存 |
|------|---|---|-------|-----------|------------------|
| v6 | 6 | 4 | 122 | **88MB** | **1.1GB** (12层) |
| v10 (L=490) | 1 | 6 | 490 | **5.6GB** | **45GB** (8层) |
| v10 (L=150) | 8 | 6 | 150 | **324MB** | **5.2GB** (8层) |
| v12 (L=400) | 1 | 8 | 400 | **4.0GB** | **32GB** (8层) |
| v12 (L=490) | 1 | 8 | 490 | **7.5GB** | **60GB** (8层) |

> **关键结论**: v6 通过 patch_size=4 将有效序列长度从 490 降到 122，使得 attention 显存降低了 **(490/122)³ ≈ 64 倍**。

---

## 五、总显存占用推导

### 显存组成

```
总显存 = 模型参数 + 优化器状态 + 前向激活值 + 梯度
```

### 5.1 v6 显存预算

```
模型参数 (bf16):     186.7M × 2B = 373MB
优化器 (fp32 m+v):   26.1M × 8B  = 209MB  (仅可训部分)
梯度 (bf16):         26.1M × 2B  = 52MB
前向激活值:          ~1.5-2GB (patch 后空间小)
────────────────────────────────────────────
总计 (单卡):         约 2.5-3GB
峰值 (含临时 tensor): 约 5-8GB
```

### 5.2 v10 显存预算

```
模型参数 (bf16):     165.7M × 2B = 331MB
优化器 (fp32 m+v):   165.7M × 8B = 1326MB  (全量可训!)
梯度 (bf16):         165.7M × 2B = 331MB
MARS 1D 前向激活:    ~3-4GB
Axial Attn 激活:     动态，取决于当前 batch 的 L
────────────────────────────────────────────
短序列 (L≈150, B=8):  约 8-12GB
长序列 (L≈490, B=1):  约 50-60GB
```

> **v10 的策略**: 通过 `LengthBucketBatchSampler` 动态调整 batch_size，长序列自动降到 bs=1，使得峰值显存可控在 ~60GB 以内（适配 80GB A100）。

### 5.3 v12 显存预算

```
模型参数 (bf16):     172.6M × 2B = 345MB
优化器 (fp32 m+v):   11.9M × 8B  = 95MB   (仅可训部分)
梯度 (bf16):         11.9M × 2B  = 24MB
MARS 1D 前向激活:    ~3-4GB (冻结，但仍需前向)
DiT 前向激活:        见下
────────────────────────────────────────────

无 Gradient Checkpointing:
  L=200: ~12GB | L=300: ~35GB | L=400: ~77GB | L=490: OOM (>95GB)

有 Gradient Checkpointing (实际使用):
  仅保留每层边界激活，中间激活在 backward 时重算
  L=400: ~10-12GB (峰值)
  L=490: ~15-20GB (峰值)
```

### 5.4 显存对比汇总

| 模型 | 配置 | 估算 Peak 显存 | 限制因素 |
|------|------|---------------|---------|
| **v6** | bs=6, L≤490 | **5-8GB** | 几乎不受限 |
| **v10** | bs=1~8(动态), L≤490 | **8-60GB** (随 L 变化) | MARS 全量优化器 + 全分辨率 attn |
| **v12** | bs=1, L≤400, 有CP | **10-12GB** | Gradient CP 后可控 |
| **v12** | bs=1, L=490, 无CP | **>95GB (OOM)** | 无法训练 |

---

## 六、训练速度实测对比

从日志直接计算：

| 指标 | v6 | v10 | v12 |
|------|-----|------|------|
| 每 epoch 步数 | 1802 | 4815 | 2936 |
| **每 epoch 训练时间** | **257s** (4.3min) | **642s** (10.7min) | **1324s** (22min) |
| 每 step 平均耗时 | ~143ms | ~133ms | ~451ms |
| steps/sec | ~7.0 | ~7.5 | ~2.2 |
| **Validation 耗时** | ~2s | ~38s | **869s** (14.5min) |
| 总 epochs | 300 | 150 | 100 |
| **预计总训练时间** | ~21h | ~28h | ~61h |

### 为什么每 step 差异大

| | v6 | v10 | v12 |
|---|---|---|---|
| 每 step 计算的 attention | patch后, (122,122) | 动态 L, 平均~(150,150) | 动态 L, 长序列 bs=1 |
| 前向+反向复杂度 | 低 | 中 (MARS 也反传) | 高 (全分辨率+CP重算) |
| Grad Checkpointing 开销 | 无 | 无 | **+30~40%** |

### 为什么 v12 Validation 特别慢

- v6: 直接用 direct head 或 20 步 tau-leap 采样
- v10: 单次前向推理即可得到预测
- v12: 需要跑 **50 步 Euler ODE 采样**，每步完整前向传播 → 50× 推理时间

---

## 七、速度差异根因排序

| 排序 | 原因 | 影响 | 备注 |
|------|------|------|------|
| **#1** | 无 patch 下采样，全分辨率 attention | 计算量 64× ↑ | v6 有 patch=4 |
| **#2** | Gradient Checkpointing | 训练速度 -30~40% | v12 必须开启否则 OOM |
| **#3** | 长序列时 batch_size 降到 1 | GPU 利用率低 | v10/v12 动态 bs，v6 固定 bs=6 (patch后空间小) |
| **#4** | 50 步 ODE validation | eval 时间 430× ↑ | v10 单次推理，v6 仅 20 步 |
| **#5** | 连续空间所有位置参与梯度 | 梯度张量更大 | v6 离散态大量 mask skip |

---

## 八、设计哲学总结

| | v6 | v10 | v12 |
|---|---|---|---|
| **设计目标** | 高效离散生成 | 极致监督精度 | 极简连续生成 |
| **效率策略** | Patch↓ + Dilated + 多头并行 | 动态 batch + 小 dim | 动态 batch + Gradient CP |
| **精度策略** | 多 loss 约束 + density guidance | MARS 全量微调 | 连续空间建模 |
| **取舍** | 牺牲分辨率换速度 | 牺牲显存换精度 | 牺牲速度换简洁 |

---

## 九、优化 v12 的可能方向

1. **引入 Patch Embedding** (patch_size=2~4)：最直接有效，可将显存降低 8~64×
2. **Window/Local Axial Attention**：限制 attention 范围，如 window_size=64
3. **Flash Attention**：不减少计算量但显著减少 attention 矩阵的显存占用
4. **进一步优化动态 batch 策略**：当前已有 LengthBucketBatchSampler，可尝试更激进的 token budget
5. **减少 ODE 采样步数**：50→20 或使用自适应步长
6. **Progressive Training**：先小 L 训练，再逐步增大到 490
