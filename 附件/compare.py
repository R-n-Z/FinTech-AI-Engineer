import torch
import time
from types import SimpleNamespace
from baseline import MoEBlockBaseline
from solution import MoEBlockOptimized


def bench(cls, name, config, seq_len, device, dtype, warmup=5, measure=10):
    model = cls(config).to(device=device, dtype=dtype)
    x = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        x_in = x.detach().clone().requires_grad_(True)
        y = model(x_in)
        y.sum().backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(measure):
        model.zero_grad(set_to_none=True)
        x_in = x.detach().clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x_in)
        y.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    avg_ms = sum(times) / len(times) * 1000
    del model, x
    torch.cuda.empty_cache()
    return peak_mb, avg_ms


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Optimized 对比评测")
    parser.add_argument("--seq-lens", type=str, default="2048,8192,32768,65536,131072")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 环境"
    device = torch.device("cuda")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    config = SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
    )

    seq_lengths = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]

    print(f"dtype={args.dtype}, warmup={args.warmup}, measure={args.measure}")
    print(f"Config: H={config.hidden_size}, shared_im={config.intermediate_size}, "
          f"moe_im={config.moe_intermediate_size}, E={config.num_experts}, top_k={config.num_experts_per_tok}")
    print()
    print(f"{'SeqLen':>8s} | {'Baseline Peak':>14s} | {'Optimized Peak':>14s} | "
          f"{'Mem Saved':>10s} | {'Baseline ms':>11s} | {'Optimized ms':>11s} | {'Speed Δ':>8s}")
    print("-" * 95)

    for T in seq_lengths:
        try:
            base_mem, base_ms = bench(
                MoEBlockBaseline, "Baseline", config, T, device, dtype,
                args.warmup, args.measure
            )
            opt_mem, opt_ms = bench(
                MoEBlockOptimized, "Optimized", config, T, device, dtype,
                args.warmup, args.measure
            )
            mem_saved = (base_mem - opt_mem) / base_mem * 100 if base_mem > 0 else 0
            speed_delta = (opt_ms - base_ms) / base_ms * 100 if base_ms > 0 else 0
            print(
                f"{T:>8d} | {base_mem:>13.1f} MB | {opt_mem:>13.1f} MB | "
                f"{mem_saved:>9.1f}% | {base_ms:>10.2f} ms | {opt_ms:>10.2f} ms | "
                f"{speed_delta:>+7.1f}%"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{T:>8d} | {'OOM':>13s} | {'OOM':>13s} | {'--':>9s} | {'--':>10s} | {'--':>10s} | {'--':>7s}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{T:>8d} | {'OOM':>13s} | {'OOM':>13s} | {'--':>9s} | {'--':>10s} | {'--':>10s} | {'--':>7s}")
            else:
                raise


if __name__ == "__main__":
    import argparse
    main()
