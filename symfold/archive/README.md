# symfold/archive — 历史版本代码归档

本目录存放**非当前训练/评估闭包**所需的历史版本代码，按原结构归档保存，仅作存档参考。

> 归档于 2026-06-26。当前主力训练为 v11a / v11b，backbone 为 v9（v10/v11a 复用）。

## 归档内容

| 子目录 | 内容 |
|--------|------|
| `v1/ v2/ v5/ v7/ v8/ handson/ v11c/` | 早期/中间版本模型代码（v11c 为空目录） |
| `train/` | 旧训练脚本 `train_v2 ~ train_v9`（含共享基类 `train_v3.py`） |
| `eval/` | 旧评估脚本 `eval_v2/v3/v4`、`eval_v6_improved`、`decoder_ablation_v9` |
| `analysis/` | 旧版本的 case 分析与可视化脚本（v5~v8） |

## 重要说明

- 归档脚本之间的 `import`（如 `train_v6 → train_v3`、`eval_v2 → train_v2`）因路径变更**已失效**，如需重跑请手动调整 import 路径（`symfold.xxx` → `symfold.archive.xxx`）或临时移回原位。
- **当前训练/评估代码不依赖本目录任何内容**，可安全保留归档状态。

## 为什么 v3 / v4 / v6 没有被归档

当前正在训练的 **v11b** 复用了它们的 flow 代码，依赖链如下：

```
v11b/model.py
  ├── v3/discrete_flow.py
  └── v6/discrete_flow.py
        ├── v3/discrete_flow.py
        └── v4/discrete_flow.py
              └── v3/discrete_flow.py
```

因此 `v3/ v4/ v6/` 整组保留在 `symfold/` 原位。
