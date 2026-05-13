# MoE Memory Optimization Contest

## 文件说明

- `baseline.py`：参考实现 `MoEBlockBaseline`
- `solution.py`：选手提交模板，需实现 `MoEBlockOptimized`
- `correctness_check.py`：正确性验证脚本
- `benchmark.py`：显存和速度评测脚本
- `题面.md`：题面
- `readme.md`：运行说明

## 正确性检查

```bash
python correctness_check.py --solution solution.py
```

检查项：

- 前向输出
- 输入梯度
- router、routed experts、shared expert、post_norm 的全部参数梯度

默认阈值：

- 前向：`rtol=2e-2, atol=1e-3`
- 梯度：`cosine_sim >= 0.995` 且 `relative_l2 <= 1e-2`

快速验证脚本自身是否可运行：

```bash
python correctness_check.py --seq-len 128 --hidden-size 256 --intermediate-size 768 --moe-intermediate-size 96 --num-experts 16 --top-k 4
```

## Benchmark

```bash
python benchmark.py --solution solution.py
```

默认序列长度：

```text
2048, 8192, 32768, 65536, 131072, 262144
```

快速 smoke run：

```bash
python benchmark.py --solution solution.py --seq-lens 2048 --warmup 1 --measure 1
```
