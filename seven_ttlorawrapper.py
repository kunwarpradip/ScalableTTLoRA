import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import tensorly as tl
import math
import tensorly as tl

from tensorly.decomposition import tensor_train

tl.set_backend('pytorch') #used to set the backend for the tensorly library to PyTorch

import torch
import tensorly as tl
from tensorly.decomposition import tensor_train

tl.set_backend('pytorch')

# def get_ttlora_rank(r, ttlora_shape):
#             ttlora_rank = [1]
#             for i in range(len(ttlora_shape)-1):
#                 ttlora_rank.append(r)
#             ttlora_rank.append(1)
#             return ttlora_rank

def tensorized_multiplication(X, tt_cores, m_factors, n_factors, device):
    """
    Generalized multiplication of X by a TT-decomposed matrix Y.

    Args:
      X: [B, m], where m = product(m_factors)
      tt_cores: list of length len(m_factors) + len(n_factors),
                each core has shape [r_i, factor_dim, r_{i+1}],
                from something like tensor_train(Y_tensorized, rank=...).
      m_factors: list of input factor dimensions, e.g. [4, 3] => 12
      n_factors: list of output factor dimensions, e.g. [6, 2] => 12

    Returns:
      Z: [B, prod(n_factors)] => the result of X @ Y (in a TT sense).
    """
    
    B = X.size(0)
    S_len = X.size(1)

    # 1) Reshape X => [B, m1, m2, ...]
    shape_x = [B]+[S_len] + m_factors[::-1]
    tt_state = X.view(shape_x)  # e.g. [B, 4, 3] if m_factors=[4,3]

    # 2) Insert an initial rank dimension => [B, 1, m1, m2, ...]
    tt_state = tt_state.unsqueeze(1)  # shape [B, r=1, m1, m2, ...]

    # We'll do:
    #   - first "len(m_factors)" cores:  contract out each input factor
    #   - next "len(n_factors)" cores:   add each output factor dimension
    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n

    if len(tt_cores) != total_cores:
        raise ValueError(f"Expected {total_cores} TT-cores, got {len(tt_cores)}")

    # 3) Process each INPUT factor
    # We want to remove the last dimension from tt_state each time:
    # eq = 'b r ... m, r m p -> b p ...'
    # Explanation:
    #   - 'm' is the last dimension
    #   - '...' lumps any leftover dims in between 'r' and 'm'
    #   - we sum over (r, m), leaving 'b' and leftover '...' plus new rank p
    for i in range(num_m):
       
        core = tt_cores[i]  # shape [r_in, m_i, r_out]
        # base_core = layer_weight_cores[i]
        # core = base_core + alpha * initial_core
        # print(core.shape,tt_state.shape)
        eq_in = 'b r ... m, r m p -> b p ...'
        tt_state = torch.einsum(eq_in, tt_state, core)
        # shape now has same # of dims, except the "m" dimension is contracted out
        # and rank dimension might have changed from r_in to r_out

    # 4) Process each OUTPUT factor
    # Now each output factor is appended at the end:
    # eq = 'b r ..., r n p -> b p ... n'
    # Explanation:
    #   - we sum over 'r'
    #   - leftover dims remain in '...'
    #   - new factor dimension 'n' is appended at the end
    start_n = num_m
    for i in range(num_n):
        core = tt_cores[start_n + i]  # shape [r_in, n_i, r_out]
        # base_core = layer_weight_cores[start_n + i]
        # core = base_core + alpha * initial_core
        eq_out = 'b r ..., r n p -> b p ... n'
        tt_state = torch.einsum(eq_out, tt_state, core)
        # shape now has one more dimension at the end for 'n'

    # 5) Flatten to [B, -1] => [B, prod(n_factors)]
    Z = tt_state.view(B,S_len, -1)
    return Z



class TTLoRALinearWrapper_withcores(nn.Module): #Inherits from nn.Module
        def __init__(self, module: nn.Module, tt_shape, tt_rank, alpha:int):
            super().__init__()
            self.base_module = module
            self.tt_shape = tt_shape
            self.tt_rank = tt_rank
            self.alpha=alpha
            self.in_features_shape, self.out_features_shape = self.base_module.weight.shape

            '''Create a torch tensor dummy Weight_delta of shape (in_feature_shape, out_feature_shape) 
            and initialize all 0s'''
            self.Weight_delta=torch.zeros((self.in_features_shape, self.out_features_shape)).to('cuda')
            # print("Weight_delta Initial shape: ", self.Weight_delta.shape)
            
            '''Then allocate random values using gaussian distribution to dummy Weight_delta'''
            self.reset_parameters()
            # print("Weight_delta shape after reset_parameters: ", self.Weight_delta.shape) 

            '''Decompose the dummy Weight_delta to high dimensional tensor based on the TT shapes'''
            self.Weight_TT_dimension = self.reshape_tensor(torch.tensor(self.Weight_delta)).to('cuda')
            # print("Weight_TT_dimension shape: ", self.Weight_TT_dimension.shape)

            '''We have dummy weight decomposed into multiple tensors based on tt_shape
            Now, we create tensor cores as Parameters which are trainable
            Paramerter wraps the tensors into traninable parameters
            ParameterList holds the list of parameters
            TT Cores are initialized using standard normal distribution based on the ttcores shapes'''
            self.ttlora_cores = nn.ParameterList([nn.Parameter(self.initialize_cores(*shape).to('cuda')) for shape in self.get_ttcores_shapes()])
            # print("Initialized TTLoRA cores as Parameters:")
            # for i, core in enumerate(self.ttlora_cores):
            #     print(f"Core {i}: {core.shape}")
            
            '''Using tensor train, decompose into multiple tensors based on the ranks and shapes provided'''
            self.ttlora_cores_dummy = tensor_train(self.Weight_TT_dimension, self.tt_rank)
            # print("Tensor trained dummy cores based on tt_rank:")
            # for i, core in enumerate(self.ttlora_cores_dummy):
            #     print(f"Core {i}: {core.shape}")

            '''Transfer the values of tensor trained ttlora_cores_dummy to ttlora_cores trainable parameters'''
            for i in range(len(self.ttlora_cores)):
                self.ttlora_cores[i].data = torch.tensor(self.ttlora_cores_dummy[i], dtype=torch.float32).to('cuda')
        
            self.ttlora_cores.requires_grad= True 
            # Make the bias non-trainable
            if self.base_module.bias is not None:
                    self.base_module.bias.requires_grad = False

        def get_ttcores_shapes(self):
            shapes = []
            ranks = self.tt_rank
            for i in range(len(self.tt_shape)):
                shape = (ranks[i], self.tt_shape[i], ranks[i + 1])
                shapes.append(shape)
            return shapes

        def reshape_tensor(self, tensor):
            return tensor.reshape(*self.tt_shape) ## * unpacks the tt_shape list into individual arguments

        def reset_parameters(self):
            '''Initialize the given tensor with random values from a gaussian distribution'''
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.Weight_delta, a=math.sqrt(8))

        def initialize_cores(self, *shape):
            '''Initialize the given tensor with random values from a standard normal distribution (mean = 0 and std = 1)
            and scaled by a calculated standard deviation'''
            std = 1.0 / math.sqrt(shape[1]) #Standard deviation
            return torch.randn(*shape) * std
            
        def forward(self, x: torch.Tensor) -> torch.Tensor: # x is input used in forward pass at every call of model
            if self.alpha > 0:
                # Ensure all tensors are on the same device
                m_factors = [12, 8, 8]
                n_factors = [2,2,2, 8,12]
                device = 'cuda'
                out = tensorized_multiplication(x.to(device), 
                                                self.ttlora_cores, 
                                                m_factors=m_factors, 
                                                n_factors=n_factors, 
                                                device=device) 
                
                # self.ttlora_weights = self.ttlora_weights.to(device)
                # self.alpha = torch.tensor(self.alpha, device=device, dtype=self.ttlora_weights.dtype)
                # base_out = F.linear(input:x.to(device), weight:self.base_module.weight, bias:self.base_module.bias)
                return self.base_module(x.to(device)) + out*self.alpha
                # adapted_weight = self.base_module.weight + self.alpha * self.ttlora_weights.reshape(self.in_features_shape, self.out_features_shape)
            #     return F.linear(x.to(device), adapted_weight, self.base_module.bias)
            # else:
            #     return F.linear(x.to(self.base_module.weight.device), self.base_module.weight, self.base_module.bias)
