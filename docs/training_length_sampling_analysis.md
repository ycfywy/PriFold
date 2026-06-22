# 训练样本长度对模型的影响：采样、Padding 与动态 Batch Size

> 2026-06-22 | 对照代码：`symfold/data.py`、`symfold/train/train_v10.py`、`symfold/config/v10/v10_ddp.json`

---

## 1. 训练集长度分布

bpRNA-train（max_len_filter=490）：

| 区间 | 样本数 | 占比 | 说明 |
|---|---:|---:|---|
| 0-50 | 235 | 2.2% | 极短 |
| 50-100 | 4624 | 42.8% | **主体** |
| 100-150 | 3165 | 29.3% | **次主体** |
| 150-200 | 1224 | 11.3% | 中等 |
| 200-300 | 774 | 7.2% | 中长 |
| 300-400 | 624 | 5.8% | 长 |
| 400-500 | 161 | 1.5% | 极长 |

统计：n=10807, min=33, max=487, **mean=133.5**, **median=105**

---

## 2. 长度分桶采样（LengthBucketBatchSampler）

### 2.1 代码位置

`symfold/data.py` 第 173-236 行。

### 2.2 工作流程

```text
1. 拿到全部样本的长度列表 [len(r.seq) for r in records]
2. 每个 epoch 先 shuffle，再按长度排序
3. 从短到长遍历，对每个位置根据当前长度动态计算该 batch 放多少条
4. 按顺序切 batch
5. 最后把所有 batch 再 shuffle（打乱长度区间的出场顺序）
```

### 2.3 相似长度在一起吗？

**是的。** 排序后连续切 batch，每个 batch 内的样本长度相近。

好处：
- 减少 padding 浪费（batch 内最大长度 ≈ 所有样本长度）
- 显存占用可预测

坏处：
- 一个 epoch 内不同 batch 的梯度信号不均匀（短序列 batch 有更多样本，长序列 batch 可能只有 1 条）

### 2.4 动态 batch size 计算

```python
# 初始化时：
if max_sq_tokens is None:
    median_len = sorted(lengths)[len(lengths) // 2]   # = 105
    max_sq_tokens = batch_size * median_len ** 2       # = 8 × 105² = 88200

# 每个 batch：
def _get_dynamic_batch_size(self, length):
    bs = max(1, max_sq_tokens // (length * length))
    return min(bs, batch_size * 4)   # cap at 32
```

**核心思想**：显存 ∝ B × L²（pair matrix），所以保持 `B × L² ≈ 常数`。

### 2.5 v10 当前配置（batch_size=8）下的实际 batch size

```
max_sq_tokens = 8 × 105² = 88,200
```

| 序列长度 L | L² | 88200 // L² | cap(32) | 实际 bs |
|---:|---:|---:|---:|---:|
| 50 | 2,500 | 35 | 32 | **32** |
| 80 | 6,400 | 13 | 32 | **13** |
| 105 | 11,025 | 8 | 32 | **8** |
| 150 | 22,500 | 3 | 32 | **3** |
| 200 | 40,000 | 2 | 32 | **2** |
| 300 | 90,000 | 0→1 | 32 | **1** |
| 400 | 160,000 | 0→1 | 32 | **1** |
| 487 | 237,169 | 0→1 | 32 | **1** |

---

## 3. Padding 机制

### 3.1 代码位置

`symfold/data.py` 第 243-282 行的 `make_collate_fn`。

### 3.2 Padding 方式

```python
max_l = int(lengths.max())                      # batch 内最大长度
set_len = ceil(max_l / patch_size) * patch_size  # 对齐到 patch_size=4 的倍数

# contact map: pad 到 (set_len, set_len)，填 0
ct = np.zeros((set_len, set_len))
ct[:length, :length] = raw_ct

# mask: 有效区域为 1，padding 为 0
mask = np.zeros((set_len, set_len))
mask[:length, :length] = 1.0

# seq one-hot: pad 到 (set_len, 4)，填 0
# tokenizer: padding='max_length', max_length=max_l+2
```

### 3.3 Mask 的作用

模型中所有 loss 和 inference 都乘以 `contact_mask`：

```python
logit = logit * contact_mask       # padding 位置 logit 强制为 0
bce_focal = bce_raw * focal * valid  # loss 只算有效区域
```

所以 padding 不会影响梯度计算——但它**占显存**。

### 3.4 为什么长度分桶能减少 padding

如果 batch=[长度 50, 长度 400]，set_len=400，短样本浪费了 400²-50²=157,500 个位置的显存。
如果 batch=[长度 395, 长度 400]，set_len=400，浪费极少。

分桶保证 batch 内长度接近 → padding 浪费最小。

---

## 4. 相关训练配置参数

| 配置项 | 值 | 作用 |
|---|---|---|
| `training.batch_size` | 8 | 基础 batch size，用于计算 max_sq_tokens |
| `training.max_len_filter` | 490 | 训练时过滤掉 >490 的样本 |
| `training.gradient_accumulation_steps` | 3 | 每 3 步才做一次 optimizer step |
| `training.grad_clip` | 0.5 | 梯度裁剪（对 bs=1 的长样本尤其重要） |

有效 batch size = `batch_size × grad_accum = 8 × 3 = 24`（对 median 长度样本）

但对长样本(L≥300)，实际单步 bs=1，有效 bs = 1×3 = 3（仅靠 grad_accum 积累）。

---

## 5. 长样本 batch_size=1 的问题

### 5.1 梯度噪声大

当 bs=1 时，一个样本的梯度代表整个 step 的信号。如果该样本恰好是 bad case 或特殊结构，梯度方向可能完全偏。

虽然有 `gradient_accumulation_steps=3`（累积 3 步才 update），但对长样本每步都是 bs=1，所以有效只看了 3 条长序列就更新一次参数。

### 5.2 与短样本训练不对称

| 长度 | 实际 bs | × grad_accum | 有效样本数/update |
|---:|---:|---:|---:|
| 80 | 13 | ×3 | **39** |
| 105 | 8 | ×3 | **24** |
| 200 | 2 | ×3 | **6** |
| 400 | 1 | ×3 | **3** |

短序列每次 update 看了 39 个样本，长序列只看了 3 个。长序列的梯度信号方差更大，训练更不稳定。

### 5.3 对 MARS unfreeze 的影响更大

v10 解冻了 160M 参数。当 bs=1 时：
- 模型从单条长 RNA 计算整个 MARS 梯度
- 如果这条 RNA 恰好是稀有 RFAM，梯度可能把 MARS 参数拉偏
- grad_clip=0.5 能限制幅度，但方向仍然可能有噪声

### 5.4 长样本在训练集中占比

300-490 长度：624 + 161 = 785 条（7.3%），但它们每条占一个完整 batch 位→ 它们贡献了更多 step 数。

粗算：
- 短序列(80): 4624 条 / bs=13 = ~356 batches
- 长序列(300-490): 785 条 / bs=1 = **785 batches**

长序列虽然只占 7.3% 样本，但贡献了更多 step（因为 bs=1）。这意味着训练中**很大比例的 step 都在处理长序列**，但每个 step 只有 1 条样本的梯度信号。

---

## 6. 可能的改进方向

### 6.1 提高 max_sq_tokens

当前 `max_sq_tokens = 88,200`（自动计算）。GPU 有 98GB，实际峰值约 40GB。可以显式设置更大的 budget：

```json
"max_sq_tokens": 300000
```

这会让长序列(L=300)的 bs 从 1 变成 3，L=200 从 2 变成 7。

### 6.2 设 batch_size 下限

修改 `_get_dynamic_batch_size`，加入 `min_batch_size=2`：

```python
def _get_dynamic_batch_size(self, length):
    bs = max(min_batch_size, max_sq_tokens // (length * length))
    return min(bs, batch_size * 4)
```

保证即使最长序列也至少有 2 条/batch，减少单样本噪声。

### 6.3 增大 gradient_accumulation

对长序列区间，可以用更大的 grad_accum 补偿：

```
当前: grad_accum=3, 长序列有效 bs=3
如果: grad_accum=8, 长序列有效 bs=8
```

但这会让短序列的有效 bs 过大（39→104），可能浪费。

### 6.4 混合长度 batch（放弃严格分桶）

不严格按长度排序，而是在每个 batch 中混合不同长度（用 padding 补齐）。好处是梯度更多样，坏处是显存浪费。

权衡方案：**部分混合** — 分桶但桶宽放大（比如 50-200 一桶、200-500 一桶），桶内 shuffle。

### 6.5 长序列 loss 加权

如果长序列训练不稳定，可以对长序列样本的 loss 乘一个 < 1 的权重，减少单条长序列对参数的影响：

```python
length_weight = min(1.0, 150.0 / length)  # 长度>150 权重递减
loss = loss * length_weight
```

---

## 7. 当前配置总结

```json
{
  "batch_size": 8,
  "gradient_accumulation_steps": 3,
  "max_len_filter": 490,
  "grad_clip": 0.5
}
```

实际效果：
- median 长度(105): bs=8, 有效=24 ✅ 稳定
- 中等长度(200): bs=2, 有效=6 ⚠️ 偏小
- 长序列(300-490): bs=1, 有效=3 ❌ 噪声大

**建议**：在 config 中显式加 `"max_sq_tokens": 300000`，可以让长序列 bs 提升到 2-3，同时不会让短序列 OOM。
