# SymFold 迁移到 PriFold 的实验汇报与原因分析

> 撰写时间：2026-06-01
> 对照参考实现：`/root/aigame/dannyyan/RNADiffFold/symfold/`（以下简称 **RDF**，论文 RNADiffFold/SymFold v5）
> 当前实现：`/root/aigame/dannyyan/PriFold/symfold/`（以下简称 **当前**）

---

## 一、做了什么

### 1.1 任务

把 SymFold 的 RNA 二级结构生成式模型迁移到 PriFold 项目中，目标是：
- 在 PriFold 现成的数据/MARS 语言模型基础上，引入 **Bernoulli Discrete Flow Matching** 这种生成式范式；
- 与原 PriFold 的判别式 RNAformer 形成对照，探索"生成 vs 判别"在 RNA contact 任务上的差异。

### 1.2 已完成

- 在 `symfold/` 下实现了完整 pipeline：数据加载、MARS 编码、Axial DiT 主干、Bernoulli flow loss、CTMC 多步采样、greedy projection、训练/评估脚本；
- 跑通了 **v0 smoke**（512 样本/5 epoch）→ **v1 full**（全量 31041 训练样本/53 epoch）两个版本；
- 修复了一个**验证集与 RNAStrAlign 测试集泄漏**的关键 bug（`val` 阶段误用 `ts` 划分，已改为 `vl`）；
- 训练曲线、checkpoint、history.json、四联图自动绘制，pipeline 完整。

### 1.3 v1 训练与测试结果（best.pt @ epoch 51, num_steps=20）

| 数据集 | F1 | P | R | MCC | 主线 PriFold F1 | gap |
|---|---:|---:|---:|---:|---:|---:|
| 验证集（bpRNA-VL0 + RNAStrAlign-vl） | 0.5793 | — | — | — | — | — |
| **bpRNA-test (TS0)** | **0.2582** | 0.2134 | 0.3612 | 0.2668 | 0.7700 | **-0.51** |
| **rnastralign-test** | **0.7532** | 0.6776 | 0.8542 | 0.7585 | 0.9738 | -0.22 |
| **archiveii-test** | **0.5097** | 0.4461 | 0.6020 | 0.5147 | 0.9043 | **-0.39** |

---

## 二、表现"差"的两层定义

需要分清两类对比，否则会得出错误结论：

### 2.1 与 PriFold 主线判别式的差距 → **预期内**
PriFold 主线是 SOTA 判别式监督模型，bpRNA F1=0.77。生成式 flow matching 模型在收敛速度和最终精度上与判别式有天然差距是合理的。**这部分差距不是问题，是设计选择的代价**。

### 2.2 与同样是生成式的 RNADiffFold/SymFold 的差距 → **问题所在**
RDF 在 8 个 benchmark 平均 F1 = 0.735，最强 v3 = 0.752。**我们的 v1 在 bpRNA 上 0.26，archiveII 上 0.51**，这是一个数量级的差距。RDF 已经把"生成式 flow matching"这条路证明能做到接近判别式 SOTA 的水平，所以差距完全可以归因为**实现层面的缺失**。

**本汇报聚焦后者：对照 RDF 找出我们到底缺了什么。**

---

## 三、核心问题诊断（按影响排序）

### 🔥 问题 1：语言模型选错了——MARS T5-encoder 没有 attention map

**这是最致命的单点差距。**

| 维度 | RDF | 当前 |
|---|---|---|
| 语言模型 | **RNA-FM**（BERT 风格，12 层 × 20 head, 640 dim） | **MARS-LX**（LLaMA2 风格 encoder，12 层 × 12 head, 1056 dim） |
| 输出形式 | hidden + **attention map (默认暴露)** | hidden（**attention map 客观存在但 forward 没暴露**） |
| pair feature 来源 | 12 层 × 20 head = **240 个 (L,L) attention map** 直接喂入 | hidden 经 outer concat 强行升 2D |

**关键澄清**：MARS 本身是标准 multi-head self-attention，attention weights `(B, 12, L, L)` 客观存在，只是当前 `Attention.forward` 走 `torch.nn.functional.scaled_dot_product_attention` (flash-attn 路径) 不返回 attention 矩阵。要拿到 144 个 attention map，**只需派生 `llama2_with_attn.py` 让最后 N 层走 manual softmax 路径**，不一定要换 RNA-FM。

RDF 的关键代码（`RDF/symfold/src/v5/model.py L114-141`）：
```python
out = self.fm_conditioner(tokens, repr_layers=[3,6,9,12], return_contacts=True)
attn = out['attentions'][:, :, :, 1:-1, 1:-1]   # (B, 12, 20, L, L)
attn = attn.reshape(B, 12*20, L, L)              # 240 通道天然 pair feature
```

我们的代码（`P/prifold/llama2.py:Attention.forward` L296-302）：
```python
# flash implementation
if flash_attn_available:
    output = torch.nn.functional.scaled_dot_product_attention(
        xq, xk, xv, attn_mask=attn_mask, ...)   # ⚠️ flash 路径不返回 attention weights
```

`Transformer.forward` 也只返回 `(logits, hidden_states)`，attention 信息被吞掉。

**为什么这点最关键**：RNA contact 是个 pair-level 任务。RNA-FM 的 attention map **本身就编码了"哪两个位置该交互"的信息**（这是预训练学出来的），且形状天然是 `(L,L)`，与 contact map 同构。我们用 outer concat 强行把 1D hidden 升 2D，是让 DiT 自己从零学这个映射——浪费了 RNA-FM/RNA-LM 的最有价值产物。

**预期影响**：F1 +5~10 分，是单点最大改进项。

---

### 🔥 问题 2：DiT block 太弱，缺了 RDF v3-v5 堆叠的全部增强

我们的 `AxialDiTBlock`（`P/symfold/dit.py L48-91`）：
- 标准 `nn.MultiheadAttention` 行/列注意力
- 标准 SiLU FFN
- 一个 AdaLN(t)
- 6 层

RDF v5 的 block：
- **Dilated Axial Attention**（dilation 模式 [1,1,1, 2,2,2, 4,4,4]）→ 多尺度长程感受野
- **SwiGLU FFN**（比 ReLU/SiLU 强）
- **Triangle Update**（在 L6-8 层）→ 编码三角约束（i-j 配对、j-k 配对 ⇒ i-k 受限）
- **FiLM 调制**（除 AdaLN 外的额外条件路径）
- **9 层**（vs 我们 6 层）
- **QK-Norm + RoPE**（数值稳定 + 2D 位置编码）

差距等价于：
- 没有多尺度感受野 → 学不到长程配对关系（archiveII 上长 RNA 表现差就是这个原因）；
- 没有 Triangle Update → 学不到嵌套结构约束（这是 AlphaFold 的核心 building block）；
- FFN/Norm 落后于现代标配。

**预期影响**：F1 +3~5 分。

---

### ⚠️ 问题 3：pos_bias 用法错误——当通道吃掉，不是 attention bias

主线 PriFold 的 `RNAformer` 把 `pos_bias` 作为**第 0 层 attention 的 additive bias**（`utils/RNAformer/model/Riboformer_outfirst.py L497-505`）：
```python
bias = bias.unsqueeze(1).unsqueeze(2)        # (B,1,1,L,L)
for idx, layer in enumerate(self.layers):
    if idx == 0:
        pair_act = layer(pair_act, mask, bias)              # 第一层带 bias
    else:
        pair_act = layer(pair_act, mask, zeros_like(bias))  # 其余层 bias=0
```

我们的实现（`P/symfold/model.py: build_conditions`）：
```python
parts = [mars_2d, seq_2d]
if self.use_pos_bias:
    parts.append(pos_bias.unsqueeze(1))     # ⚠️ 直接拼到 channel
cond = torch.cat(parts, dim=1)
```

**问题**：pos_bias 的本意是"提示哪两个位置基于碱基互补规则应该配对"，作为 attention bias 直接抑制/增强 attention logits 才是它最匹配的形态。当通道吃掉相当于让模型自己学这个先验，浪费了显式信号。

**预期影响**：F1 +1~2 分。

---

### ⚠️ 问题 4：seq_oh one-hot 通道冗余

我们额外拼了 8 个 seq_oh 通道（A/T/G/C 的 outer concat）。但 MARS 的第一层就是 `nn.Embedding(vocab_size=20, dim=1056)`，**hidden state 必然包含碱基身份信息**——seq_oh 是 MARS 表征的真子集冗余。

RDF 的实现也没有显式 seq_oh，依赖 RNA-FM 自身的表征。

**预期影响**：F1 ±0.5（相对小，但属于"多花特征空间换不了精度"的复杂化）。

---

### ⚠️ 问题 5：训练 loss 与 RDF 的物理约束差异

RDF v3+ 引入了 **stacking loss + non-crossing loss** 作为辅助物理约束（强制 contact map 满足 RNA 二级结构的拓扑性质）。我们代码里**写了这两个 loss class**（`discrete_flow.py: StackingLoss / NonCrossingLoss`），但配置里**权重设为 0**：

`P/symfold/config/prifold_symflow_v1_full.json`:
```json
"stack_weight": 0.0,
"nc_weight": 0.0,
```

**预期影响**：F1 +0.5~1。

---

### 💡 问题 6：推理采样过于朴素

我们的 sampling 流程：
- 20 步均匀 Euler-CTMC + 单一 trajectory + greedy projection；
- `num_samples_per_input` 默认 1，多样本投票路径甚至有 bug（多样本平均后没重新投影）。

RDF 的推理：
- **Cosine τ-leap schedule**（早期粗、晚期细）；
- **多样本投票 + 重投影**（典型 5 个 trajectory 取交集再投影）；
- **density-guided rate damping**（用 density head 预测的对数动态调速率）。

**预期影响**：F1 +0.5~1。

---

### 💡 问题 7：训练数据采样策略

RDF 使用 **bucket batch sampler** 把同长度样本聚到一起，并对 RNA 家族做加权采样（避免 RNAStrAlign 的 5S/16S rRNA 主导）。

我们的实现（`P/symfold/data.py: LengthBucketBatchSampler`）只做了**长度分桶**，没有做家族加权。

**结果直接体现在数据上**：
- 训练集 RNAStrAlign tr (20234) ≫ bpRNA TR0 (10807)，且 RNAStrAlign 是同家族重复，多样性低；
- 模型严重偏向 RNAStrAlign，**这就是为什么 bpRNA-test 0.26 而 RNAStrAlign-test 0.75** 的核心原因；
- 同时 P (0.21~0.68) ≪ R (0.36~0.85)，说明模型整体过度预测——pos_weight_base=199 + RNAStrAlign 主导导致 density 估计偏高。

**预期影响**：F1 +1~3，且能显著缓解 bpRNA 上的崩塌。

---

## 四、问题影响度小结

| # | 问题 | 类别 | 预期 ΔF1 | 修复难度 |
|---|---|---|---:|---|
| 1 | 没用 RNA-FM 的多层 attention map（用了 MARS hidden） | 🔥 致命 | +5~10 | 中（需改 MARS forward 或换 LM） |
| 2 | DiT block 弱（无 dilation / triangle / SwiGLU / FiLM） | 🔥 致命 | +3~5 | 中-大 |
| 3 | pos_bias 当 channel 而非 attention bias | ⚠️ 重要 | +1~2 | 小 |
| 4 | seq_oh 冗余 | ⚠️ 重要 | ±0.5 | 极小 |
| 5 | stacking / non-crossing loss 权重为 0 | ⚠️ 重要 | +0.5~1 | 极小 |
| 6 | 推理采样朴素（无 cosine schedule / 投票 / density-guided） | 💡 可选 | +0.5~1 | 小-中 |
| 7 | 数据采样无家族加权（bpRNA 严重低估） | ⚠️ 重要 | +1~3 | 小 |

**理论 v2 上限**（保守估计）：bpRNA-test F1 从 0.26 → 0.45+，rnastralign-test 0.75 → 0.85+，archiveII 0.51 → 0.70+。

---

## 五、为什么会做成这样：根因复盘

1. **迁移时只参考了 SymFold v1 baseline (`SEDiT`)，没看 v3-v5**：RDF 的设计经过 5 次迭代积累，我们直接抄了最早最弱的版本，丢掉了所有关键升级。
2. **语言模型替换没有评估其形态适配性**：用 MARS 替换 RNA-FM 时只考虑了"PriFold 已有 MARS 现成"的便利性，**没意识到当前 MARS forward 走 flash-attn 路径直接吞掉了 attention map**——而 attention map 恰恰是 SymFold/RDF 整个 pair feature 的灵魂。**注意：MARS 本身是 LLaMA2 风格的标准多头自注意力 encoder，attention weights 客观存在，只是没暴露**——这意味着不一定要换 LM，派生一份返回 attention 的 forward 即可。
3. **照抄主线 PriFold 数据流但不加思辨**：pos_bias 拼通道这个习惯是从主线 RNAformer 的 SSDataset 流水线沿用过来的，但**主线 RNAformer 内部其实把 pos_bias 当 attention bias 用**，我们只继承了输入侧的形态，没继承内部使用方式。
4. **缺少消融意识**：写了 stacking/non-crossing loss 但权重设为 0，相当于"代码就绪但功能未启用"，没有对照实验来验证这些约束的价值。
5. **训练数据采样未审视**：直接合并 TR0 + tr 当训练集，没有考虑 RNAStrAlign 同家族重复问题，导致 bpRNA-test 严重崩塌。

---

## 六、改进路径

详细方案见 `docs/prifold_symflow_improvement_plan.md`。

按 ROI 排序：

### 优先级 1 — v2 核心升级（一次性整合）

1. **替换或改造语言模型，提供 attention map**：
   - **方案 A（推荐，工作量小）**：派生 `prifold/llama2_with_attn.py`，让 MARS-LX 的最后 N 层（建议 6 层）走 manual softmax 路径返回 attention weights，得到 12 head × N 层 = ~72 个 attention map。MARS 是 LLaMA2 风格 encoder（双向 self-attention，RoPE + RMSNorm + SwiGLU + MLM 预训练），attention 数学上和 BERT/RNA-FM 没本质区别，是否能达到 RNA-FM 同等质量取决于其在 RNA 数据上的预训练程度——经验性问题，但参数量比 RNA-FM 大（160M vs ~100M），值得一试。
   - **方案 B（保守对齐 RDF）**：直接换成 RNA-FM（与 RDF 完全对齐），得到 12 层 × 20 head = 240 个 attention map。代价是放弃 PriFold 主线的 MARS-LX 一致性。
   - **建议**：先做方案 A，如果 v2 在 bpRNA 上 F1 仍 <0.40，再切换方案 B。
2. **DiT block 升级**：dilation pattern + Triangle Update（用仓库现成的 `utils/RNAformer/module/axial_attention.py`）+ SwiGLU + AdaLN-Zero × 4 段；9 层而非 6 层。
3. **pos_bias 改为 attention bias**：在第 0 层 row/col attention 加 additive bias，对齐主线 RNAformer 的用法。
4. **删 seq_oh**：直接清掉，不做消融。
5. **打开 stacking/non-crossing loss**：`stack_weight=0.05, nc_weight=0.05` 起步。

### 优先级 2 — 训练数据改进

6. **家族加权采样**：参考 RDF 的 bucket batch sampler 做 family-weighted；或简单的：把 RNAStrAlign 同家族样本下采样到与 bpRNA 同量级。

### 优先级 3 — 推理优化（v2 见效后再做）

7. Cosine τ-leap schedule + 多样本投票 + density-guided rate damping。

---

## 七、给老板/PM 看的一句话总结

> **当前 SymFlow 实现是 SymFold 系列的早期 v1 弱化变种，且把灵魂特征——语言模型 attention map——丢失了。v1 测试结果（bpRNA F1=0.26）反映的不是"flow matching 范式不行"，而是"我们抄错了版本 + 选错了语言模型"。RDF 的 v3-v5 已证明同范式可达 F1=0.75。下一步 v2 通过 ① 引入多层 attention map ② DiT block 增强 ③ pos_bias 改 attention bias ④ 数据采样修正，**保守预期 bpRNA 测试 F1 翻倍至 0.45+**，与 RDF 同量级才有意义对比。**

---

## 附录：关键文件索引

| 模块 | 当前路径 | RDF 对照路径 |
|---|---|---|
| 主模型 | `P/symfold/model.py` | `RDF/symfold/src/v5/model.py` |
| DiT block | `P/symfold/dit.py` | `RDF/symfold/src/v5/dit.py`（含 dilation/triangle/SwiGLU） |
| Flow loss | `P/symfold/discrete_flow.py` | `RDF/symfold/src/v5/discrete_flow.py` |
| 数据 | `P/symfold/data.py` | `RDF/symfold/src/v5/data.py` |
| LM 加载 | `P/utils/lm.py`（MARS） | `RDF/symfold/src/models/condition/fm_conditioner/`（RNA-FM） |
| 配置 | `P/symfold/config/prifold_symflow_v1_full.json` | `RDF/symfold/train/configs/v5_*.json` |
| 训练曲线 | `P/symfold/outputs/prifold_symflow_v1_full/training_curves.png` | `RDF/symfold/output/v5/*.png` |
| 评估结果 | `P/symfold/outputs/prifold_symflow_v1_full/eval_best.json` | `RDF/symfold/output/v3/eval/*.json` |
