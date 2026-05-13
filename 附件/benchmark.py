import argparse
import importlib.util
import time
from types import SimpleNamespace

import torch


def load_candidate_class(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution_module", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载文件: {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MoEBlockOptimized"):
        raise RuntimeError("solution.py 中未找到类 `MoEBlockOptimized`")
    return getattr(module, "MoEBlockOptimized")


def parse_seq_lengths(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"不支持的 dtype: {name}")


def run_step(model, x):
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss


def benchmark_one(model, x, device, warmup_steps=5, measure_steps=20):
    for _ in range(warmup_steps):
        run_step(model, x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    step_times = []
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_step(model, x)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    avg_ms = sum(step_times) / len(step_times) * 1000
    min_ms = min(step_times) * 1000
    max_ms = max(step_times) * 1000
    return peak_mb, avg_ms, min_ms, max_ms


def build_config():
    return SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
    )


def main():
    parser = argparse.ArgumentParser(description="MoE 显存与速度 benchmark")
    parser.add_argument("--solution", type=str, default="solution.py")
    parser.add_argument("--seq-lens", type=str, default="2048,8192,32768,65536,131072,262144")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 环境"
    device = torch.device("cuda")
    dtype = parse_dtype(args.dtype)
    seq_lengths = parse_seq_lengths(args.seq_lens)
    config = build_config()
    CandidateCls = load_candidate_class(args.solution)

    print(f"Solution: {args.solution}")
    print(
        f"Config: B={args.batch_size}, H={config.hidden_size}, "
        f"moe_intermediate={config.moe_intermediate_size}, "
        f"shared_intermediate={config.intermediate_size}"
    )
    print(f"MoE: num_experts={config.num_experts}, top_k={config.num_experts_per_tok}")
    print(f"Warmup: {args.warmup}, Measure: {args.measure}, dtype={dtype}")
    print()
    print(
        f"{'SeqLen':>8s} | {'Peak Memory (MB)':>16s} | {'Avg (ms)':>10s} | "
        f"{'Min (ms)':>10s} | {'Max (ms)':>10s}"
    )
    print("-" * 76)

    for seq_len in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = None
        x = None
        try:
            model = CandidateCls(config).to(device=device, dtype=dtype)
            x = torch.randn(
                args.batch_size,
                seq_len,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
            peak_mb, avg_ms, min_ms, max_ms = benchmark_one(
                model, x, device, args.warmup, args.measure
            )
            print(
                f"{seq_len:>8d} | {peak_mb:>16.2f} | {avg_ms:>10.2f} | "
                f"{min_ms:>10.2f} | {max_ms:>10.2f}"
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"{seq_len:>8d} | {'OOM':>16s} | {'--':>10s} | "
                f"{'--':>10s} | {'--':>10s}"
            )
        finally:
            del model, x
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
