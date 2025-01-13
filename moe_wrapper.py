import torch.nn as nn
import tensorly as tl
import torch
import torch.nn.functional as F
from typing import Dict

tl.set_backend('pytorch')  # used to set the backend for the tensorly library to PyTorch

class MoEsparseRouting(nn.Module):
    def __init__(self, base_module: nn.Module, experts: dict, router_weights: nn.Parameter, common_alpha: int):
        super().__init__()
        self.base_module = base_module
        self.experts: Dict[str, Dict] = experts
        self.num_experts = len(experts)
        self.router_weights = router_weights
        self.alpha = common_alpha
        self.in_features_shape, self.out_features_shape = self.base_module.layer[0].attention.self.value.weight.shape
        print("-"*50, "\nExecuted init of MoE Wrapper")
    
    def select_expert_using_router_for_every_single_input(self, hidden_states: torch.Tensor) -> str:
        print("-"*50, "\nInside select_expert_using_router_for_every_input\n")
        # Pool hidden states to reduce sequence dimension: mean pooling across sequence length
        print("original_hidden_states_shape =", hidden_states.shape)
        pooled_hidden_states = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)
        print(f"Reduced hidden states shape: {pooled_hidden_states.shape}")

        # Compute router logits
        routing_logits = torch.matmul(pooled_hidden_states, self.router_weights)  # Shape: (batch_size, num_experts)
        print(f"Routing logits shape: {routing_logits.shape}")

        # Apply softmax to get probabilities
        routing_probs = torch.softmax(routing_logits, dim=-1)  # Shape: (batch_size, num_experts)
        print(f"Routing probabilities shape: {routing_probs.shape}")

        # Select top-1 expert for each input
        selected_expert_idx = torch.argmax(routing_probs, dim=-1)  # Shape: (batch_size,)
        print(f"Selected expert indices: {len(selected_expert_idx)}")

        # Map indices to expert names
        expert_names = list(self.experts.keys())
        selected_expert_names = [expert_names[idx] for idx in selected_expert_idx.tolist()]
        print(f"Selected experts: {len(selected_expert_names)}")

        return selected_expert_names

    def convert_ttcores_into_weights(self, adapter_module: nn.Module, ttlora_cores: list[torch.Tensor]) -> torch.Tensor:
        # print("-"*50, "\nInside convert_ttcores_into_weights")
        # Ensure all tensors are on the same device
        device = adapter_module.weight.device
        in_features_shape, out_features_shape = adapter_module.weight.shape
        tt_shape_len = len(ttlora_cores)
        for i in range(len(ttlora_cores)):
            ttlora_cores[i] = ttlora_cores[i].to(device)
        if tt_shape_len == 4:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq->jlnp',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3]
            )
        elif tt_shape_len == 5:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq,qrs->jlnpr',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3], ttlora_cores[4]
            )
        elif tt_shape_len == 6:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq,qrs,stu->jlnprt',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3], ttlora_cores[4], ttlora_cores[5]
            )
        elif tt_shape_len == 7:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq,qrs,stu,uvw->jlnprtv',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3], ttlora_cores[4], ttlora_cores[5], ttlora_cores[6]
            )
        elif tt_shape_len == 8:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq,qrs,stu,uvw,wxy->jlnprtvx',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3], ttlora_cores[4], ttlora_cores[5], ttlora_cores[6], ttlora_cores[7]
            )
        elif tt_shape_len == 10:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc->jlnprtvxzb',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3],
                ttlora_cores[4], ttlora_cores[5], ttlora_cores[6], ttlora_cores[7],
                ttlora_cores[8], ttlora_cores[9]
            )
        elif tt_shape_len == 12:
            ttlora_weights = torch.einsum(
                'ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc,cde,efg->jlnprtvxzbdf',
                ttlora_cores[0], ttlora_cores[1], ttlora_cores[2], ttlora_cores[3],
                ttlora_cores[4], ttlora_cores[5], ttlora_cores[6], ttlora_cores[7],
                ttlora_cores[8], ttlora_cores[9], ttlora_cores[10], ttlora_cores[11]
            )

        ttlora_weights = ttlora_weights.to(device)
        self.alpha = torch.tensor(self.alpha, device=device, dtype=ttlora_weights.dtype)

        adapted_weight = adapter_module.weight + self.alpha * ttlora_weights.reshape(in_features_shape, out_features_shape)
        
        return adapted_weight

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        print("-"*50, "\nInside forward pass")
        selected_expert_names = self.select_expert_using_router_for_every_single_input(hidden_states)
        batch_size = hidden_states.shape[0]
        # Iterate over each input in the batch
        for single_input in range(batch_size):
            # print("Expert type selected", selected_expert_names[single_input])
            layer_idx = 0
            for layer in self.base_module.layer:
                # print("At layer", layer_idx)
                # For query
                query_ttcores = self.experts[selected_expert_names[single_input]][f'layer_{layer_idx}']["query"]
                query_ttcores_list = [query_ttcores[key] for key in query_ttcores.keys()]
                # print("-"*50, f"\nExpert's Query TT-Cores: {(type(query_ttcores_list))}")
                for param in query_ttcores_list:
                    pass
                    # print(f"\nParameter requires_grad: {param.requires_grad}")
                    # print(f"Parameter shape: {param.shape}")
                # print("-"*50, f"\nQuery weight before adaptation: {layer.attention.self.query.weight}")
                expert_adapted_query_weight = self.convert_ttcores_into_weights(layer.attention.self.query, query_ttcores_list)
                # print("-"*50, f"\nExpert adapted query weight:\n {expert_adapted_query_weight}")
                layer.attention.self.query.weight = nn.Parameter(expert_adapted_query_weight, requires_grad=False)
                # print("-"*50, f"\nGrad requirement after query weight expert adaptation: {layer.attention.self.query.weight.requires_grad}")

                # For value
                value_ttcores = self.experts[selected_expert_names[single_input]][f'layer_{layer_idx}']["value"]
                value_ttcores_list = [value_ttcores[key] for key in value_ttcores.keys()]
                # print("-"*50, f"\nExpert's Value TT-Cores: {(type(value_ttcores_list))}")
                for param in value_ttcores_list:
                    # print(f"\nParameter requires_grad: {param.requires_grad}")
                    # print(f"Parameter shape: {param.shape}")
                    pass
                # print("-"*50, f"\nValue weight before adaptation: {layer.attention.self.value.weight}")
                expert_adapted_value_weight = self.convert_ttcores_into_weights(layer.attention.self.value, value_ttcores_list)
                # print("-"*50, f"\nExpert adapted value weight: {expert_adapted_value_weight}")
                layer.attention.self.value.weight = nn.Parameter(expert_adapted_value_weight, requires_grad=False)
                # print("-"*50, f"\nGrad requirement after value weight expert adaptation: {layer.attention.self.value.weight.requires_grad}")
                layer_idx += 1

        return self.base_module(hidden_states, *args, **kwargs)