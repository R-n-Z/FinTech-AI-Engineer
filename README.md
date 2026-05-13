# 长序列训练 MoE Block 显存优化 — 赛题项目

招商银行金融科技竞赛 — MoE（Mixture of Experts）训练显存优化。

## 项目结构

```
bank/
├── README.md                    # 本文件 — 项目总览
├── 新建 文本文档.txt             # 赛题题面（完整描述）
└── 附件/
    ├── readme.md                # 快速运行说明
    ├── baseline.py              # 基线实现 MoEBlockBaseline
    ├── solution.py              # 选手提交模板（需实现 MoEBlockOptimized）
    ├── correctness_check.py     # 正确性自测脚本
    └── benchmark.py             # 显存与速度评测脚本
```

## 赛题概要

在单卡环境下对给定的 MoE Block 基础实现进行训练时显存优化。不改变模型数学定义的前提下，降低峰值显存占用并控制计算耗时。

### 模型配置

| 参数 | 值 |
|---|---|
| hidden_size | 2048 |
| moe_intermediate_size | 768 |
| intermediate_size | 6144 |
| num_experts | 128 |
| num_experts_per_tok | 8 |
| norm_topk_prob | true |
| torch_dtype | bfloat16 |

### 评测环境

- 单机单卡 H20-96GB
- Python 3.12 + PyTorch 2.8.0 + CUDA 12.8

### 评分规则

**正确性门槛**（三项全部通过后才进入显存和速度评分）：

| 检查项 | 阈值 |
|---|---|
| 前向输出 | rtol=2e-2, atol=1e-3 |
| 输入梯度 | cosine_sim ≥ 0.995 且 relative_l2 ≤ 1e-2 |
| 参数梯度 | cosine_sim ≥ 0.995 且 relative_l2 ≤ 1e-2 |

**评分权重**：显存优化 : 速度 = 6 : 4

## 快速开始

### 正确性自测

```bash
cd 附件
python correctness_check.py --solution solution.py
```

快速 smoke test（小规模）：
```bash
python correctness_check.py --seq-len 128 --hidden-size 256 --intermediate-size 768 --moe-intermediate-size 96 --num-experts 16 --top-k 4
```

### 性能评测

```bash
python benchmark.py --solution solution.py
```

快速 smoke run：
```bash
python benchmark.py --solution solution.py --seq-lens 2048 --warmup 1 --measure 1
```

## 基线架构

`MoEBlockBaseline` 由以下组件构成：

- **MoERouter** — Top-K 路由，计算 token 到 expert 的分配
- **MoEExperts** — 128 个 expert，每个包含 gate/up/down 投影 + SiLU 激活
- **SharedExpert** — 共享 expert（gate/up/down 线性层 + SiLU）
- **RMSNorm** — 最终归一化层

前向流程：`hidden_states → gate(路由) → experts(稀疏计算) + shared_expert(共享计算) → post_norm → output`

## 优化方向（不限定）

- 中间激活张量的显存生命周期管理
- 计算流程重组（如算子融合、重计算/checkpointing）
- 自定义 CUDA kernel 或 Triton kernel
- torch.compile 优化

## 提交要求

1. `姓名+code.zip` — 包含 `solution.py`（含 `MoEBlockOptimized` 类）
2. `姓名+report.pdf` — 方案报告（优化思路、原理、正确性验证、权衡分析、实验结果）
