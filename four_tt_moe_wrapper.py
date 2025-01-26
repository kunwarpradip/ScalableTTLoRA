import torch.nn as nn
import tensorly as tl
import torch
import torch.nn.functional as F
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tensor_train

tl.set_backend('pytorch')

def tensorized_multiplication_experts(self, X, tt_cores, m_factors, n_factors, gates):
    """
    - X: [B, m]
    - tt_cores: list of (len(m_factors)+len(n_factors)) cores,
                each core has shape [E, r_i, factor_dim, r_{i+1}],
                where E is the number of experts.
    - gates: [B, E], gating weights for each sample.
      e.g. from a router or gumbel_softmax.
    - returns: [B, prod(n_factors)]
    """
    B = X.shape[0]
    seq_len = X.shape[1]
    # (1) Reshape X => e.g. [B, m_factors[::-1]] + unsqueeze rank
    shape_x = [B] + [seq_len]+  m_factors[::-1]
    # print("\nInput shape reshaped to match the m_factors in reverse", shape_x)
    tt_state = X.view(shape_x).unsqueeze(1)  # => [B, 1, ...]
    # print("\nShape of tt_state after unsqueeze to add a space for r", tt_state.shape)
    tt_state.to(self.device)

    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n

    if len(tt_cores) != total_cores:
        raise ValueError(f"Expected {total_cores} TT-cores, got {len(tt_cores)}")

    # We'll do the same flow: contract out input factors, then add output factors.
    # But each time, we must "mask" the stacked core by gates, sum over E.

    for i in range(num_m):
        # (a) core: [E, r_i, m_i, r_{i+1}]
        core = tt_cores[i].to(self.device)

        # (b) Apply gating => shape [B, r_i, m_i, r_{i+1}]
        #    gates => [B, E], expand => [B, E, 1, 1, 1]
        #    core => [E, r_i, m_i, r_{i+1}] => unsqueeze(0) => [1, E, r_i, m_i, r_{i+1}]
        #    multiply & sum out E => [B, r_i, m_i, r_{i+1}]
        gates_exp = gates.view(B,4,1, 1, 1).to(self.device)
        # if i == 0:
        #     print("\nGates expanded shape", gates_exp.shape)
        #     print("\nCore shape with experts", core.shape)
        core_expanded = core.unsqueeze(0).to(self.device)  # [1, 1, E, r_i, m_i, r_{i+1}]
        # if i == 0:
        #     print("\nCore expanded shape with experts to match gate and select cores for all the inputs", core_expanded.shape)
        masked_core = (core_expanded * gates_exp).sum(dim=1).to(self.device)
        # if i == 0:
        #     print("\nMasked Core shape", masked_core.shape)
        # => [B, r_i, m_i, r_{i+1}]

        # (c) einsum with tt_state
        # tt_state => [B, r_i, ..., m] (the last dimension is the factor dim if i < num_m)
        eq_in = "b r ... m, b r m p -> b p ..."
        tt_state = torch.einsum(eq_in, tt_state, masked_core).to(self.device)
        # if i==0:
        #     print("\nShape of tt_state after einsum with masked core to reduce input_dim", tt_state.shape)
        # if i==len(m_factors)-1:
        #     print("\nFinal Shape of tt_state after einsumth masked core to reduce input_dim", tt_state.shape)
        # => [B, r_{i+1}, leftover...] wi

    # Now do output factors (which add a dimension).
    start_n = num_m
    for i in range(num_n):
        core = tt_cores[start_n + i].to(self.device)  # shape [E, r_i, n_i, r_{i+1}]
        # Mask by gating
        gates_exp = gates.view(B, -1, 1, 1, 1).to(self.device)
        core_expanded = core.unsqueeze(0).to(self.device)  # => [1, E, r_i, n_i, r_{i+1}]
        masked_core = (core_expanded * gates_exp).sum(dim=1).to(self.device)
        # => [B, r_i, n_i, r_{i+1}]

        # eq_out: "b r ..., b r n p -> b p ... n"
        eq_out = "b r ..., b r n p -> b p ... n"
        tt_state = torch.einsum(eq_out, tt_state, masked_core).to(self.device)
        # if i==0:
        #     print("\nShape of tt_state after einsum with masked core to add output_dim", tt_state.shape)
        # if i==len(n_factors)-1:
        #     print("\nFinal Shape of tt_state after einsum with masked core to add output_dim", tt_state.shape)

    # Flatten
    Z = tt_state.view(B, seq_len, -1).to(self.device)
    # print("\nFinal Shape of output after flattening", Z.shape)
    return Z

class MoEsparseRouting(nn.Module):
    def __init__(self, 
                 base_module: nn.Module, 
                 experts: dict,
                 common_alpha: int,
                 m_factors:list, 
                 n_factors:list,
                 train_router_only:bool,
                 device: torch.device):
        super().__init__()
        self.base_module = base_module
        self.m_factors = m_factors
        self.n_factors = n_factors
        self.experts = experts
        self.common_alpha = common_alpha
        self.train_router_only = train_router_only
        self.num_experts = len(experts)
        self.device = device
        print("Inside MoEsparseRouting Initialization")

        # Compute m, n
        self.m = 1
        for f in self.m_factors:
            self.m *= f
        self.n = 1
        for f in self.n_factors:
            self.n *= f
        # print("m, n", self.m, self.n)

        # A trainable router => from [B, m] => [B, E]
        self.router = nn.Linear(self.m, self.num_experts, bias=True)
 
    def custom_query_forward(self, X, gates, stacked_query_cores, *args, **kwargs):
        # print("*"*50)
        # print("Inside custom query forward")
        # (c) Store as buffer
        tt_cores_stacked = stacked_query_cores
        # Count the total TT-cores
        self.num_m = len(self.m_factors)
        self.num_n = len(self.n_factors)
        self.num_cores = self.num_m + self.num_n
        if len(tt_cores_stacked) != self.num_cores:
            raise ValueError(f"Expected {self.num_cores} cores, got {len(tt_cores_stacked)}")

        # Figure out E from the shape of the first core
        example_core = tt_cores_stacked[0]
        self.num_experts = example_core.shape[0]  # first dimension => E

        # (1) Store TT-cores as buffers => freeze them
        if self.train_router_only:
            for i, core in enumerate(tt_cores_stacked):
                # shape => [E, r_i, factor_dim, r_{i+1}]
                self.register_buffer(f"core_{i}", core)
        else:
            # If you wanted them trainable, store as parameters
            # But here the focus is on freezing.
            raise NotImplementedError("We only do freezing in this example")
        
        # (d) Collect the stacked TT-cores from buffers
        # We'll build a python list in the correct order
        stacked_cores_list = []
        for i in range(self.num_cores):
            c_buf = getattr(self, f"core_{i}")  
            # shape => [E, r_i, factor_dim, r_{i+1}]
            stacked_cores_list.append(c_buf)

        # (e) Call the helper function to compute input with tensor cores => [B, s_l, n]
        ttlora_x_computation = tensorized_multiplication_experts(
            self, X, stacked_cores_list, self.m_factors, self.n_factors, gates)
        
        # (f) Override the forward function of the query layer
        alpha = self.common_alpha
        out = ttlora_x_computation*alpha

        # scaling_factor = torch.mean(Q) + torch.mean(X)
        return out

    def custom_value_forward(self, X, gates, stacked_value_cores, *args, **kwargs):
        # print("*"*50)
        # print("Inside custom value forward")
        # (c) Store as buffer
        tt_cores_stacked = stacked_value_cores
        # Count the total TT-cores
        self.num_m = len(self.m_factors)
        self.num_n = len(self.n_factors)
        self.num_cores = self.num_m + self.num_n
        if len(tt_cores_stacked) != self.num_cores:
            raise ValueError(f"Expected {self.num_cores} cores, got {len(tt_cores_stacked)}")

        # Figure out E from the shape of the first core
        example_core = tt_cores_stacked[0]
        self.num_experts = example_core.shape[0]  # first dimension => E

        # (1) Store TT-cores as buffers => freeze them
        if self.train_router_only:
            for i, core in enumerate(tt_cores_stacked):
                # shape => [E, r_i, factor_dim, r_{i+1}]
                self.register_buffer(f"core_{i}", core)
        else:
            # If you wanted them trainable, store as parameters
            # But here the focus is on freezing.
            raise NotImplementedError("We only do freezing in this example")
        
        
        # (d) Collect the stacked TT-cores from buffers
        # We'll build a python list in the correct order
        stacked_cores_list = []
        for i in range(self.num_cores):
            c_buf = getattr(self, f"core_{i}")  
            # shape => [E, r_i, factor_dim, r_{i+1}]
            stacked_cores_list.append(c_buf)

        # (e) Call the helper function to compute input with tensor cores => [B, s_l, n]
        ttlora_x_computation = tensorized_multiplication_experts(
            self, X, stacked_cores_list, self.m_factors, self.n_factors, gates)
        
        # (f) Override the forward function of the query layer
        alpha = self.common_alpha

        # scaling_factor = torch.mean(V) + torch.mean(X)
        out = ttlora_x_computation*alpha
        return out


    def forward(self, X, *args, **kwargs):
        # print("*"*50)
        # print("Inside MoEsparseRouting Forward")
        gumbel=True, 
        temperature=1.0
        B = X.size(0)

        # Router => gating [B, E]
        pooled_hidden_states = X.mean(dim=1)
        logits = self.router(pooled_hidden_states)
        # print("Router Logits shape", logits.shape,"\n")
        if gumbel:
            gates = F.gumbel_softmax(logits, tau=temperature, hard=True)
        else:
            gates = F.softmax(logits, dim=-1)
        # print("Gate shape", gates.shape)
        # print("Gates example:\n", gates[0:10])
        # experts = self.experts
        layer_idx = 0
        for layer in self.base_module.layer:
            # print("layer idx",layer_idx)

            # Start Collecting the TT-cores from here, for this layer, for all experts
            access_first_expert = next(iter(self.experts.values()))

            ##################################################For Value######################################
            # (a) Collect query TT-cores for all experts
            list_query_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["query"]))]
            
            for expert in self.experts.values():
                for i, (core_name, tensor) in enumerate(expert[f"layer_{layer_idx}"]["query"].items()):
                    list_query_cores[i].append(tensor)

            # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
            stacked_query_cores = [torch.stack(core_list) for core_list in list_query_cores]
            # Check by printing the stacked cores
            # for i, core_list in enumerate(stacked_query_cores):
            #     print("*"*20, f"Core {i}:", "*"*20)
            #     print(type(core_list), core_list.shape)

            layer.attention.self.query.forward = lambda X, *args, **kwargs: self.custom_query_forward(X, gates, stacked_query_cores, *args, **kwargs)
            
            ##################################################For Value######################################
            # (a) Collect query TT-cores for all experts
            list_value_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["value"]))]
            for expert in self.experts.values():
                for i, (core_name, tensor) in enumerate(expert[f"layer_{layer_idx}"]["value"].items()):
                    list_value_cores[i].append(tensor)

            # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
            stacked_value_cores = [torch.stack(core_list) for core_list in list_value_cores]
            # Check by printing the stacked cores
            # for i, core_list in enumerate(stacked_query_cores):
            #     print("*"*20, f"Core {i}:", "*"*20)
            #     print(type(core_list), core_list.shape)
            layer.attention.self.value.forward = lambda X, *args, **kwargs: self.custom_value_forward(X, gates, stacked_value_cores, *args, **kwargs)
            #increase the layer index
            layer_idx += 1

        return self.base_module(X, *args, **kwargs)
        
    