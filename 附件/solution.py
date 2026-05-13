import torch
import torch.nn as nn


class MoEBlockOptimized(nn.Module):
    """
    选手在该类中实现自己的显存优化版本。
    约束：保持与 MoEBlockBaseline 数学定义等价。

    允许使用 PyTorch、torch.compile、自定义 CUDA extension、Triton kernel。
    不允许改变参数含义、路由定义、输出定义或梯度数学定义。

    参数结构必须与 baseline 一致，以支持 load_state_dict 加载权重：
    - self.experts.gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
    - self.experts.down_proj: [num_experts, hidden_size, moe_intermediate_size]
    - self.shared_expert.gate_proj.weight: [intermediate_size, hidden_size]
    - self.shared_expert.up_proj.weight: [intermediate_size, hidden_size]
    - self.shared_expert.down_proj.weight: [hidden_size, intermediate_size]
    - self.gate.weight: [num_experts, hidden_size]
    - self.post_norm.weight: [hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size)
        )

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

        self.post_norm = nn.Module()
        self.post_norm.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        # TODO: 在此实现显存优化的 MoE 前向计算
        # 返回: [B, T, H]
        raise NotImplementedError("请实现显存优化的 MoE 前向计算")
