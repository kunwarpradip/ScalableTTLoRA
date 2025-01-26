import torch.nn as nn
import tensorly as tl
import torch
import torch.nn.functional as F
from typing import Dict

tl.set_backend('pytorch')  # used to set the backend for the tensorly library to PyTorch

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, task_labels):
        super(Router, self).__init__()
        self.router_layer = nn.Linear(input_dim, num_experts, bias=False)  # Linear layer as the router
        # print(f"Router weights: {self.router_layer.weight.shape}, requires_grad: {self.router_layer.weight.requires_grad}")
        print("inside constructor of Router")
        self.task_labels = task_labels


    def forward(self, hidden_states):
        # print("Inside Router Forward Method")
        #hidden state original shape: [batch_size, seq_len, hidden_size]
        # Pool hidden states (mean pooling across sequence length)
        pooled_hidden_states = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Compute routing logits using the linear layer
        routing_logits = self.router_layer(pooled_hidden_states)  # Shape: (batch_size, num_experts)
        # routing_logits.requires_grad_(True)
        # routing_logits.retain_grad()

        router_loss_fn = nn.CrossEntropyLoss()
        router_loss = router_loss_fn(routing_logits, self.task_labels) 

        # Apply softmax to get routing probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)  # Shape: (batch_size, num_experts)
        # routing_probs.requires_grad_(True)
        # routing_probs.retain_grad()

        return routing_logits, routing_probs, router_loss

class MoEsparseRouting(nn.Module):
    def __init__(self, base_module: nn.Module, experts: dict, task_labels: torch.Tensor, common_alpha: int):
        super().__init__()
        self.num_experts = len(experts)
        self.base_module = base_module
        self.experts: Dict[str, Dict] = experts
        self.alpha = common_alpha
        self.in_features_shape, self.out_features_shape = self.base_module.layer[0].attention.self.value.weight.shape
        self.task_labels = task_labels
        self.router = Router(input_dim=self.in_features_shape, num_experts=self.num_experts, task_labels= self.task_labels)
        # print("inside constructor of MoEsparseRouting")

    def convert_ttcores_into_weights(self, adapter_module: nn.Module, ttlora_cores: list[torch.Tensor]) -> torch.Tensor:
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
    
    def replace_attention_weights(self, batch_size, selected_expert_names):
        # Iterate over each input in the batch
        for single_input in range(batch_size):
            layer_idx = 0
            for layer in self.base_module.layer:
                # For query
                query_ttcores = self.experts[selected_expert_names[single_input]][f'layer_{layer_idx}']["query"]
                query_ttcores_list = [query_ttcores[key] for key in query_ttcores.keys()]
                expert_adapted_query_weight = self.convert_ttcores_into_weights(layer.attention.self.query, query_ttcores_list)
                layer.attention.self.query.weight.data = expert_adapted_query_weight.clone()
                # layer.attention.self.query.weight = nn.Parameter(expert_adapted_query_weight.clone(), requires_grad=False)

                # For value
                value_ttcores = self.experts[selected_expert_names[single_input]][f'layer_{layer_idx}']["value"]
                value_ttcores_list = [value_ttcores[key] for key in value_ttcores.keys()]
                expert_adapted_value_weight = self.convert_ttcores_into_weights(layer.attention.self.value, value_ttcores_list)
                layer.attention.self.value.weight.data = expert_adapted_value_weight.clone()
                # layer.attention.self.value.weight = nn.Parameter(expert_adapted_value_weight.clone(), requires_grad=False)
            
                layer_idx += 1

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):

        routing_logits, router_probs, routing_loss = self.router(hidden_states)

        # selected_expert_idx = (router_probs == router_probs.max(dim=-1, keepdim=True)[0]).nonzero(as_tuple=True)[1]

        # gumbel = F.gumbel_softmax(routing_logits, tau=1, hard=True)
        # indices = torch.arange(routing_logits.size(0), dtype=torch.float16, requires_grad=False)
        # selected_expert_idx = torch.sum(gumbel * indices)
        # # print("selected_expert_idx before int: ", selected_expert_idx)
        # selected_expert_idx = selected_expert_idx.to(dtype=torch.int32)
        # Task labels as ground truth


        selected_expert_idx = (router_probs == router_probs.max(dim=-1, keepdim=True)[0]).nonzero(as_tuple=True)[1]  # Shape: (batch_size,)
        expert_names = list(self.experts.keys())
        selected_expert_names = [expert_names[idx] for idx in selected_expert_idx.tolist()]
        print("Selected expert indices: ", selected_expert_names)

        # print("routing logits_", routing_logits)
        # print("Selected expert indices: ", selected_expert_idx)
        expert_names = list(self.experts.keys())
        selected_expert_names = [expert_names[selected_expert_idx]]
        # print("Selected expert names: ",expert_names)
        print("Selected expert: Using Gumbel - remember", selected_expert_names)

        batch_size = hidden_states.shape[0]
        # print("batch_size: ", batch_size)
        self.replace_attention_weights(batch_size, selected_expert_names)

        output = self.base_module(hidden_states, *args, **kwargs)
        # print("Output from base module shape: ", output.last_hidden_state.shape)
        # print("Output values (first 5 elements): ", output.last_hidden_state.view(-1)[:5])
        # print("-" * 50)
        # self._compute_router_gradients(router_probs)  
        print("Router loss: ", routing_loss)
        return output
    
    # def _compute_router_gradients(self, router_probs):
    #     """
    #     Manually compute gradients for the router weights.
    #     """
    #     # Use routing_logits since it requires gradients
    #     def custom_backward_hook(grad):
    #         print("\n\nWe are inside custom_backward_hook\n\n")
    #         # Compute loss gradients w.r.t. routing_logits
    #         router_grad = torch.autograd.grad(
    #             outputs=router_probs,  # Router logits
    #             inputs=self.router.router_layer.weight,  # Router weights
    #             grad_outputs=grad,  # Gradients flowing into the routing logits
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0]

    #         # Assign manually computed gradients to the router weights
    #         if router_grad is not None:
    #             self.router.router_layer.weight.grad = router_grad
    #             print("Router gradients: ", router_grad)

    #     # Register the hook directly on routing_logits, which requires gradients
    #     router_probs.register_hook(custom_backward_hook)


