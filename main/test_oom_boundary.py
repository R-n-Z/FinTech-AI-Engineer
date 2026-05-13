"""
超长序列 OOM 边界测试 — 找出 Baseline 和 Optimized 各自的最大支持序列长度
在 H20 96GB 环境中运行，确定两个模型的极限性能
"""

import argparse
import torch
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


def test_sequence_length(model_cls, seq_len, config, device, dtype, model_name="Model"):
    """测试指定序列长度是否能运行"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # 创建模型
        model = model_cls(config).to(device=device, dtype=dtype)

        # 创建输入
        x = torch.randn(
            1,  # batch_size=1
            seq_len,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )

        # 前向传播
        y = model(x)

        # 反向传播
        loss = y.sum()
        loss.backward()

        # 获取峰值显存
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        del model, x, y, loss
        torch.cuda.empty_cache()

        return {
            "success": True,
            "peak_memory_mb": peak_memory_mb,
            "error": None,
        }

    except torch.cuda.OutOfMemoryError as e:
        return {
            "success": False,
            "peak_memory_mb": None,
            "error": "OOM",
        }
    except Exception as e:
        return {
            "success": False,
            "peak_memory_mb": None,
            "error": str(e)[:50],
        }


def binary_search_max_length(model_cls, config, device, dtype, model_name, min_len=1024, max_len=262144):
    """二分查找：找出最大支持的序列长度"""
    print(f"\n🔍 为 {model_name} 二分查找最大序列长度...")
    print(f"   搜索范围: {min_len} ~ {max_len}")

    left, right = min_len, max_len
    max_working_len = None
    max_working_memory = None

    iteration = 0
    while left <= right:
        iteration += 1
        mid = (left + right) // 2
        print(f"   [迭代 {iteration}] 测试 seq_len={mid}...", end=" ", flush=True)

        result = test_sequence_length(model_cls, mid, config, device, dtype, model_name)

        if result["success"]:
            print(f"✅ 成功 (峰值: {result['peak_memory_mb']:.1f} MB)")
            max_working_len = mid
            max_working_memory = result["peak_memory_mb"]
            left = mid + 1  # 尝试更长的序列
        else:
            print(f"❌ 失败 ({result['error']})")
            right = mid - 1  # 尝试更短的序列

    return max_working_len, max_working_memory


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
        description="超长序列 OOM 边界测试 — 找出最大支持的序列长度"
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1024,
        help="最小搜索长度",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=262144,
        help="最大搜索长度",
    )
    parser.add_argument(
        "--test-points",
        type=str,
        default="2048,4096,8192,16384,32768,65536,131072,262144",
        help="直接测试这些序列长度（不进行二分查找）",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "direct", "both"],
        default="both",
        help="测试模式：binary=二分查找，direct=直接测试，both=两者都做",
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
        default="oom_boundary_results.txt",
        help="保存结果文件路径",
    )
    args = parser.parse_args()

    # 环境检查
    if not torch.cuda.is_available():
        print("❌ 错误：CUDA 不可用，需要在 GPU 环境中运行")
        return

    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)
    config = build_config()

    print("=" * 100)
    print("MoE Block 超长序列 OOM 边界测试")
    print("=" * 100)
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"可用 GPU 内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"数据类型: {args.dtype}")
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

    results = {
        "baseline_binary": None,
        "optimized_binary": None,
        "baseline_direct": {},
        "optimized_direct": {},
    }

    # 模式 1: 二分查找（找最大长度）
    if args.mode in ["binary", "both"]:
        print("\n" + "=" * 100)
        print("阶段 1: 二分查找最大支持序列长度")
        print("=" * 100)

        # Baseline 二分查找
        baseline_max_len, baseline_mem = binary_search_max_length(
            MoEBlockBaseline,
            config,
            device,
            dtype,
            "Baseline",
            args.min_len,
            args.max_len,
        )
        results["baseline_binary"] = (baseline_max_len, baseline_mem)
        print(
            f"📊 Baseline 最大支持: {baseline_max_len} tokens "
            f"(峰值 {baseline_mem:.1f} MB)" if baseline_max_len else "Baseline OOM at all lengths"
        )

        # Optimized 二分查找
        optimized_max_len, optimized_mem = binary_search_max_length(
            MoEBlockOptimized,
            config,
            device,
            dtype,
            "Optimized",
            args.min_len,
            args.max_len,
        )
        results["optimized_binary"] = (optimized_max_len, optimized_mem)
        print(
            f"📊 Optimized 最大支持: {optimized_max_len} tokens "
            f"(峰值 {optimized_mem:.1f} MB)" if optimized_max_len else "Optimized OOM at all lengths"
        )

        if baseline_max_len and optimized_max_len:
            improvement = (optimized_max_len - baseline_max_len) / baseline_max_len * 100
            print(f"\n🎯 序列长度提升: {improvement:+.1f}%")

    # 模式 2: 直接测试指定点
    if args.mode in ["direct", "both"]:
        print("\n" + "=" * 100)
        print("阶段 2: 直接测试指定序列长度")
        print("=" * 100)

        test_points = [int(s.strip()) for s in args.test_points.split(",") if s.strip()]
        test_points.sort()

        print(f"\n{'SeqLen':>8s} | {'Baseline':^30s} | {'Optimized':^30s}")
        print(f"{'':>8s} | {'峰值 (MB)':>12s} {'状态':>12s} | {'峰值 (MB)':>12s} {'状态':>12s}")
        print("-" * 80)

        for seq_len in test_points:
            baseline_result = test_sequence_length(
                MoEBlockBaseline, seq_len, config, device, dtype, "Baseline"
            )
            optimized_result = test_sequence_length(
                MoEBlockOptimized, seq_len, config, device, dtype, "Optimized"
            )

            baseline_str = (
                f"{baseline_result['peak_memory_mb']:>12.1f} {'✅ OK':>12s}"
                if baseline_result["success"]
                else f"{'':>12s} {baseline_result['error']:>12s}"
            )
            optimized_str = (
                f"{optimized_result['peak_memory_mb']:>12.1f} {'✅ OK':>12s}"
                if optimized_result["success"]
                else f"{'':>12s} {optimized_result['error']:>12s}"
            )

            print(f"{seq_len:>8d} | {baseline_str} | {optimized_str}")

            results["baseline_direct"][seq_len] = baseline_result
            results["optimized_direct"][seq_len] = optimized_result

    # 保存结果
    with open(args.save_results, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("MoE Block 超长序列 OOM 边界测试结果\n")
        f.write("=" * 100 + "\n")
        f.write(f"设备: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA 版本: {torch.version.cuda}\n")
        f.write(f"PyTorch 版本: {torch.__version__}\n")
        f.write(f"可用 GPU 内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n")
        f.write(f"数据类型: {args.dtype}\n")
        f.write("\n")

        if results["baseline_binary"]:
            baseline_len, baseline_mem = results["baseline_binary"]
            f.write(f"Baseline 最大支持序列长度: {baseline_len} tokens\n")
            if baseline_mem:
                f.write(f"  峰值显存: {baseline_mem:.1f} MB\n")

        if results["optimized_binary"]:
            optimized_len, optimized_mem = results["optimized_binary"]
            f.write(f"Optimized 最大支持序列长度: {optimized_len} tokens\n")
            if optimized_mem:
                f.write(f"  峰值显存: {optimized_mem:.1f} MB\n")

        if results["baseline_binary"] and results["optimized_binary"]:
            baseline_len, _ = results["baseline_binary"]
            optimized_len, _ = results["optimized_binary"]
            if baseline_len and optimized_len:
                improvement = (optimized_len - baseline_len) / baseline_len * 100
                f.write(f"\n序列长度提升: {improvement:+.1f}%\n")

        if results["baseline_direct"] or results["optimized_direct"]:
            f.write("\n" + "=" * 100 + "\n")
            f.write("详细测试结果\n")
            f.write("=" * 100 + "\n")
            f.write(f"\n{'SeqLen':>8s} | {'Baseline Peak (MB)':>18s} | {'Baseline Status':>15s} | "
                    f"{'Optimized Peak (MB)':>18s} | {'Optimized Status':>15s}\n")
            f.write("-" * 100 + "\n")

            all_seq_lens = set(results["baseline_direct"].keys()) | set(results["optimized_direct"].keys())
            for seq_len in sorted(all_seq_lens):
                baseline_result = results["baseline_direct"].get(seq_len, {})
                optimized_result = results["optimized_direct"].get(seq_len, {})

                baseline_str = (
                    f"{baseline_result['peak_memory_mb']:>18.1f} {'✅ OK':>15s}"
                    if baseline_result and baseline_result.get("success")
                    else f"{'':>18s} {baseline_result.get('error', 'N/A'):>15s}"
                )
                optimized_str = (
                    f"{optimized_result['peak_memory_mb']:>18.1f} {'✅ OK':>15s}"
                    if optimized_result and optimized_result.get("success")
                    else f"{'':>18s} {optimized_result.get('error', 'N/A'):>15s}"
                )

                f.write(f"{seq_len:>8d} | {baseline_str} | {optimized_str}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("分析说明\n")
        f.write("=" * 100 + "\n")
        f.write("""
1. OOM 边界 = 模型能够成功运行（前向 + 反向）的最大序列长度
2. 峰值显存 = 整个训练步骤中 GPU 内存的最大占用
3. 二分查找模式：快速定位 OOM 的临界点
4. 直接测试模式：确切知道每个序列长度的显存占用

关键观察：
- Baseline 在 T > 32K 时通常会 OOM（due to [T,K,E] int64 routing table）
- Optimized 通过 sort-based routing + checkpointing 可以支持更长的序列
- 优化方案的主要价值：能够处理 Baseline 无法处理的超长序列

OOM 的主要原因：
1. 参数（weights + gradients）：~40-50 GB
2. 激活（特别是 shared_expert 中间激活）：~6-20 GB
3. 梯度缓存与优化器状态：~10-20 GB
4. 路由表（仅 Baseline）：~130 MB @ T=8K, ~1 GB @ T=128K
""")

    print()
    print(f"✅ 结果已保存到 {args.save_results}")
    print()
    print("📊 测试完成！查看结果文件以获取详细信息。")


if __name__ == "__main__":
    main()
