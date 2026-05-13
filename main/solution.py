import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class MoEBlockOptimized(nn.Module):
    """
    显存优化版 MoE Block。

    与 baseline 的数学定义完全等价，优化点：
    - 使用 sort + bincount 替代 F.one_hot 进行 token→expert 分发，
      消除 [T, K, E] int64 稠密路由表（8K 下节省 ~130 MB，128K 下节省 ~1 GB）。

    参数结构与 baseline 一致，支持 load_state_dict 加载权重。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gate (router) —— 参数结构与 baseline 一致
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size)
        )

        # routed experts —— 参数结构与 baseline 一致
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.moe_intermediate_size,
                self.hidden_size,
            )
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            )
        )

        # shared expert —— 参数结构与 baseline 一致
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.shared_expert.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.shared_expert.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

        # post_norm —— 参数结构与 baseline 一致
        self.post_norm = nn.Module()
        self.post_norm.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        T = hidden_flat.shape[0]

        # ---- Gate / Router ----
        router_logits = F.linear(hidden_flat, self.gate.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(router_logits.dtype)

        # ---- Routed Experts (sort-based) ----
        routed_output = self._routed_experts(hidden_flat, top_k_indices, top_k_weights)

        # ---- Shared Expert ----
        shared_output = self._shared_expert(hidden_flat)

        # ---- Combine + PostNorm ----
        combined = routed_output + shared_output
        output = self._rms_norm(combined)
        return output.reshape(bsz, seq_len, hidden_size)

    # ------------------------------------------------------------------
    #  Sort-based token→expert dispatch (Plan B)
    # ------------------------------------------------------------------
    def _routed_experts(self, hidden_states, top_k_indices, top_k_weights):
        T = hidden_states.shape[0]
        K = self.top_k
        E = self.num_experts
        device = hidden_states.device

        # 展平所有 (token, expert, weight) 对应关系
        token_idx = torch.arange(T, device=device).repeat_interleave(K)   # [T*K]
        expert_flat = top_k_indices.reshape(-1)                           # [T*K]
        weights_flat = top_k_weights.reshape(-1)                          # [T*K]

        # 按 expert 排序 — 同一 expert 的 tokens 聚为连续区间
        sorted_expert, sort_order = expert_flat.sort()
        sorted_token = token_idx[sort_order]
        sorted_weights = weights_flat[sort_order]

        # 每个 expert 的 tokens 数量和偏移量
        expert_counts = torch.bincount(sorted_expert, minlength=E)
        expert_offsets = torch.zeros(E + 1, dtype=torch.int64, device=device)
        expert_offsets[1:] = expert_counts.cumsum(0)

        expert_hit = torch.where(expert_counts > 0)[0]
        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx in expert_hit:
            e_idx = expert_idx.item()
            start = expert_offsets[e_idx].item()
            end = expert_offsets[e_idx + 1].item()
            if start == end:
                continue

            t_idx = sorted_token[start:end]
            current_state = hidden_states[t_idx]

            gate, up = F.linear(
                current_state, self.experts.gate_up_proj[e_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self.experts.down_proj[e_idx]
            )

            current_hidden_states *= sorted_weights[start:end, None]
            final_hidden_states.index_add_(
                0, t_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    # ------------------------------------------------------------------
    #  Shared Expert — chunked + checkpointed (Plan A)
    #  沿序列维度分块计算，每块通过 checkpoint 丢弃中间激活，
    #  显存从 O(T*I) 降至 O(chunk_size*I)。
    #  131K 下中间激活从 ~6 GB → ~160 MB（chunk_size=~3400 时）。
    # ------------------------------------------------------------------
    def _shared_expert(self, x):
        T = x.shape[0]
        # 每块目标 ~40 MB（bf16），4 个中间量合计 ≤ 160 MB
        chunk_size = (40 * 1024 * 1024) // (self.intermediate_size * 2)
        if T <= chunk_size:
            return self._shared_expert_chunk(x)

        outputs = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_out = checkpoint.checkpoint(
                self._shared_expert_chunk, x[start:end], use_reentrant=True
            )
            outputs.append(chunk_out)
        return torch.cat(outputs, dim=0)

    @torch.compile
    def _shared_expert_chunk(self, x_chunk):
        return self.shared_expert.down_proj(
            F.silu(self.shared_expert.gate_proj(x_chunk))
            * self.shared_expert.up_proj(x_chunk)
        )

    # ------------------------------------------------------------------
    #  RMS Norm — 与 baseline 数学等价
    # ------------------------------------------------------------------
    def _rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        return self.post_norm.weight * hidden_states.to(input_dtype)
