import argparse
import importlib.util
from types import SimpleNamespace

import torch

from baseline import MoEBlockBaseline


def load_candidate_class(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution_module", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载文件: {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MoEBlockOptimized"):
        raise RuntimeError("solution.py 中未找到类 `MoEBlockOptimized`")
    return getattr(module, "MoEBlockOptimized")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    a_norm = torch.linalg.vector_norm(a_flat).item()
    b_norm = torch.linalg.vector_norm(b_flat).item()
    if a_norm == 0.0 and b_norm == 0.0:
        return 1.0
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()


def relative_l2(ref: torch.Tensor, val: torch.Tensor, eps: float = 1e-12) -> float:
    ref_flat = ref.float().reshape(-1)
    val_flat = val.float().reshape(-1)
    num = torch.linalg.vector_norm(ref_flat - val_flat).item()
    den = torch.linalg.vector_norm(ref_flat).item()
    return num / (den + eps)


def check_tensor(
    name: str,
    ref: torch.Tensor,
    sub: torch.Tensor,
    rtol: float,
    atol: float,
    grad_cos_threshold: float,
    grad_rel_l2_threshold: float,
    is_grad: bool = False,
):
    max_diff = max_abs_diff(ref, sub)
    mean_diff = mean_abs_diff(ref, sub)
    cos = cosine_sim(ref, sub)
    rel_l2 = relative_l2(ref, sub)

    if is_grad:
        ok = rel_l2 <= grad_rel_l2_threshold and cos >= grad_cos_threshold
    else:
        ok = torch.allclose(ref.float(), sub.float(), rtol=rtol, atol=atol)

    return {
        "name": name,
        "ok": ok,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_l2": rel_l2,
        "cos": cos,
    }


def build_config(args):
    return SimpleNamespace(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        norm_topk_prob=True,
    )


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "gate.weight":
                param.normal_(mean=0.0, std=0.02)
            else:
                param.normal_(mean=0.0, std=0.01)


def parse_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"不支持的 dtype: {name}")


def main():
    parser = argparse.ArgumentParser(description="MoE Block 正确性自测脚本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solution", type=str, default="solution.py")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=6144)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    atol = 1e-3
    rtol = 2e-2
    grad_cos_threshold = 0.995
    grad_rel_l2_threshold = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16):
        dtype = torch.float32

    config = build_config(args)
    CandidateCls = load_candidate_class(args.solution)

    ref_model = MoEBlockBaseline(config).to(device=device, dtype=dtype)
    initialize_weights(ref_model)
    sub_model = CandidateCls(config).to(device=device, dtype=dtype)

    try:
        sub_model.load_state_dict(ref_model.state_dict(), strict=True)
    except Exception as exc:
        print("=== 正确性报告 ===")
        print("[失败] 参数结构不一致，无法与 baseline 对齐权重")
        print(f"错误信息: {exc}")
        raise SystemExit(1)

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )
    x_ref = x.detach().clone().requires_grad_(True)
    x_sub = x.detach().clone().requires_grad_(True)

    y_ref = ref_model(x_ref)
    y_sub = sub_model(x_sub)

    grad_out = torch.randn_like(y_ref)
    loss_ref = (y_ref * grad_out).sum()
    loss_sub = (y_sub * grad_out).sum()
    loss_ref.backward()
    loss_sub.backward()

    results = [
        check_tensor(
            "前向输出",
            y_ref,
            y_sub,
            rtol,
            atol,
            grad_cos_threshold,
            grad_rel_l2_threshold,
            is_grad=False,
        ),
        check_tensor(
            "输入梯度",
            x_ref.grad,
            x_sub.grad,
            rtol,
            atol,
            grad_cos_threshold,
            grad_rel_l2_threshold,
            is_grad=True,
        ),
    ]

    ref_params = dict(ref_model.named_parameters())
    sub_params = dict(sub_model.named_parameters())
    param_names = [
        "gate.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "post_norm.weight",
    ]

    for pname in param_names:
        ref_p = ref_params.get(pname)
        sub_p = sub_params.get(pname)
        if ref_p is None or sub_p is None or ref_p.grad is None or sub_p.grad is None:
            results.append(
                {
                    "name": f"参数梯度({pname})",
                    "ok": False,
                    "max_diff": 0.0,
                    "mean_diff": 0.0,
                    "rel_l2": 0.0,
                    "cos": 0.0,
                }
            )
            continue
        results.append(
            check_tensor(
                f"参数梯度({pname})",
                ref_p.grad,
                sub_p.grad,
                rtol,
                atol,
                grad_cos_threshold,
                grad_rel_l2_threshold,
                is_grad=True,
            )
        )

    print("=== 正确性报告 ===")
    print(f"设备: {device.type}, dtype: {str(dtype)}")
    print(
        f"输入维度: B={args.batch_size}, T={args.seq_len}, H={config.hidden_size}"
    )
    print(f"MoE配置: num_experts={config.num_experts}, top_k={config.num_experts_per_tok}")
    print(f"容忍阈值: rtol={rtol}, atol={atol}")
    print(
        f"参数梯度阈值: rel_l2<={grad_rel_l2_threshold}, "
        f"cosine_sim>={grad_cos_threshold}"
    )
    print()

    all_pass = True
    for result in results:
        status = "通过" if result["ok"] else "不通过"
        all_pass = all_pass and result["ok"]
        print(
            f"  {result['name']}: {status} | "
            f"max_abs_diff={result['max_diff']:.6e}, "
            f"mean_abs_diff={result['mean_diff']:.6e}, "
            f"rel_l2={result['rel_l2']:.6e}, cosine_sim={result['cos']:.6f}"
        )

    print()
    if all_pass:
        print("总结: 所有检查项通过")
    else:
        print("总结: 存在未通过的检查项")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
