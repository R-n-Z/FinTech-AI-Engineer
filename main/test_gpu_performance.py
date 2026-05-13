"""
GPU 性能完整 Benchmark — Baseline vs Optimized 对比
在 H20 96GB CUDA 12.8 环境中运行本脚本，获取实际性能数据
"""

import argparse
import torch
import time
from types import SimpleNamespace
from baseline import MoEBlockBaseline
from solution import MoEBlockOptimized


def parse_dtype(name: str):
    """解析数据类型"""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[name]


def run_step(model, x):
    """执行一步训练（前向 + 反向）"""
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss.item()


def benchmark_one(model, x, device, warmup_steps=5, measure_steps=20):
    """对单个模型进行 benchmark"""
    # Warmup（丢弃缓存等不稳定因素）
    for _ in range(warmup_steps):
        run_step(model, x)
    torch.cuda.synchronize()

    # 重置内存统计
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # 测量性能
    step_times = []
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_step(model, x)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    avg_ms = sum(step_times) / len(step_times) * 1000
    min_ms = min(step_times) * 1000
    max_ms = max(step_times) * 1000
    std_ms = (sum((t - sum(step_times) / len(step_times)) ** 2
                  for t in step_times) / len(step_times)) ** 0.5 * 1000

    return {
        "peak_memory_mb": peak_mb,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
    }


def build_config():
    """构建竞赛配置"""
    return SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="MoE Baseline vs Optimized 性能完整对比"
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="2048,4096,8192,16384,32768,65536,131072",
        help="测试的序列长度列表（逗号分隔）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="批大小"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="预热步数"
    )
    parser.add_argument(
        "--measure", type=int, default=20, help="测量步数"
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="数据类型",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="gpu_benchmark_results.txt",
        help="保存结果文件路径",
    )
    args = parser.parse_args()

    # 环境检查
    if not torch.cuda.is_available():
        print("❌ 错误：CUDA 不可用，需要在 GPU 环境中运行")
        return

    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)
    seq_lengths = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    config = build_config()

    print("=" * 100)
    print("MoE Block 性能对比 Benchmark")
    print("=" * 100)
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"数据类型: {args.dtype}")
    print(f"批大小: {args.batch_size}")
    print(f"预热步数: {args.warmup}, 测量步数: {args.measure}")
    print()
    print("模型配置:")
    print(
        f"  hidden_size={config.hidden_size}, "
        f"shared_intermediate={config.intermediate_size}, "
        f"moe_intermediate={config.moe_intermediate_size}"
    )
    print(
        f"  num_experts={config.num_experts}, "
        f"top_k={config.num_experts_per_tok}"
    )
    print()

    # 表格头
    print(f"{'SeqLen':>8s} | {'Baseline':^45s} | {'Optimized':^45s} | {'节省比例':>10s}")
    print(
        f"{'':>8s} | {'峰值 (MB)':>12s} {'耗时 (ms)':>15s} {'std':>10s} | "
        f"{'峰值 (MB)':>12s} {'耗时 (ms)':>15s} {'std':>10s} | {'显存':>10s}"
    )
    print("-" * 120)

    results = []

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()

        baseline_result = None
        optimized_result = None
        baseline_error = None
        optimized_error = None

        # 测试 Baseline
        try:
            torch.cuda.reset_peak_memory_stats(device)
            model = MoEBlockBaseline(config).to(device=device, dtype=dtype)
            x = torch.randn(
                args.batch_size,
                seq_len,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
            baseline_result = benchmark_one(
                model, x, device, args.warmup, args.measure
            )
            del model, x
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            baseline_error = "OOM"
        except Exception as e:
            baseline_error = str(e)[:20]

        # 测试 Optimized
        try:
            torch.cuda.reset_peak_memory_stats(device)
            model = MoEBlockOptimized(config).to(device=device, dtype=dtype)
            x = torch.randn(
                args.batch_size,
                seq_len,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
            optimized_result = benchmark_one(
                model, x, device, args.warmup, args.measure
            )
            del model, x
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            optimized_error = "OOM"
        except Exception as e:
            optimized_error = str(e)[:20]

        # 输出结果
        if baseline_error or optimized_error:
            baseline_str = (
                f"{baseline_result['peak_memory_mb']:>12.1f} "
                f"{baseline_result['avg_ms']:>15.2f} "
                f"{baseline_result['std_ms']:>10.2f}"
                if baseline_result
                else f"{'':>12s} {'':>15s} {baseline_error:>10s}"
            )
            optimized_str = (
                f"{optimized_result['peak_memory_mb']:>12.1f} "
                f"{optimized_result['avg_ms']:>15.2f} "
                f"{optimized_result['std_ms']:>10.2f}"
                if optimized_result
                else f"{'':>12s} {'':>15s} {optimized_error:>10s}"
            )
            saved_str = "--"
        else:
            baseline_str = (
                f"{baseline_result['peak_memory_mb']:>12.1f} "
                f"{baseline_result['avg_ms']:>15.2f} "
                f"{baseline_result['std_ms']:>10.2f}"
            )
            optimized_str = (
                f"{optimized_result['peak_memory_mb']:>12.1f} "
                f"{optimized_result['avg_ms']:>15.2f} "
                f"{optimized_result['std_ms']:>10.2f}"
            )
            saved = (
                (baseline_result["peak_memory_mb"] - optimized_result["peak_memory_mb"])
                / baseline_result["peak_memory_mb"]
                * 100
            )
            saved_str = f"{saved:>9.1f}%"

        print(f"{seq_len:>8d} | {baseline_str} | {optimized_str} | {saved_str}")

        results.append({
            "seq_len": seq_len,
            "baseline": baseline_result,
            "baseline_error": baseline_error,
            "optimized": optimized_result,
            "optimized_error": optimized_error,
        })

    # 保存结果
    with open(args.save_results, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("MoE Block 性能对比 Benchmark 结果\n")
        f.write("=" * 120 + "\n")
        f.write(f"设备: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA 版本: {torch.version.cuda}\n")
        f.write(f"PyTorch 版本: {torch.__version__}\n")
        f.write(f"数据类型: {args.dtype}\n")
        f.write(f"批大小: {args.batch_size}\n")
        f.write(f"预热步数: {args.warmup}, 测量步数: {args.measure}\n")
        f.write("\n")

        f.write(f"{'SeqLen':>8s} | {'Baseline Peak (MB)':>16s} | {'Baseline Avg (ms)':>16s} | "
                f"{'Optimized Peak (MB)':>18s} | {'Optimized Avg (ms)':>18s} | {'节省比例':>10s}\n")
        f.write("-" * 120 + "\n")

        for r in results:
            seq_len = r["seq_len"]
            baseline = r["baseline"]
            optimized = r["optimized"]
            baseline_error = r["baseline_error"]
            optimized_error = r["optimized_error"]

            if baseline_error or optimized_error:
                baseline_str = (
                    f"{baseline['peak_memory_mb']:>16.1f} {baseline['avg_ms']:>16.2f}"
                    if baseline
                    else f"{'':>16s} {str(baseline_error):>16s}"
                )
                optimized_str = (
                    f"{optimized['peak_memory_mb']:>18.1f} {optimized['avg_ms']:>18.2f}"
                    if optimized
                    else f"{'':>18s} {str(optimized_error):>18s}"
                )
                saved_str = "--"
            else:
                baseline_str = (
                    f"{baseline['peak_memory_mb']:>16.1f} {baseline['avg_ms']:>16.2f}"
                )
                optimized_str = (
                    f"{optimized['peak_memory_mb']:>18.1f} {optimized['avg_ms']:>18.2f}"
                )
                saved = (
                    (baseline["peak_memory_mb"] - optimized["peak_memory_mb"])
                    / baseline["peak_memory_mb"]
                    * 100
                )
                saved_str = f"{saved:>9.1f}%"

            f.write(
                f"{seq_len:>8d} | {baseline_str} | {optimized_str} | {saved_str}\n"
            )

        f.write("\n" + "=" * 120 + "\n")
        f.write("分析说明：\n")
        f.write("1. 峰值显存 = 整个训练步骤（前向 + 反向）的最大 GPU 内存占用\n")
        f.write("2. 耗时 = forward + backward 的总时间（ms）\n")
        f.write("3. std = 20次测量的时间标准差\n")
        f.write("4. 节省比例 = (baseline - optimized) / baseline * 100%\n")
        f.write("5. OOM = 显存不足，无法运行该序列长度\n")

    print()
    print(f"✅ 结果已保存到 {args.save_results}")


if __name__ == "__main__":
    main()
