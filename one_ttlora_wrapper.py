import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import tensorly as tl
import math
import tensorly as tl

from tensorly.decomposition import tensor_train

tl.set_backend('pytorch') #used to set the backend for the tensorly library to PyTorch

class TTLoRALinearWrapper(nn.Module): #Inherits from nn.Module
        def __init__(self, module: nn.Module, tt_shape, tt_rank, alpha:int):
            super().__init__()
            self.base_module = module
            self.tt_shape = tt_shape
            self.tt_rank = tt_rank
            self.alpha=alpha
            
            self.in_features_shape, self.out_features_shape = self.base_module.weight.shape

            '''Create a torch tensor dummy Weight_delta of shape (in_feature_shape, out_feature_shape) 
            and initialize all 0s'''
            self.Weight_delta=torch.zeros((self.in_features_shape, self.out_features_shape))
            # print("Weight_delta Initial shape: ", self.Weight_delta.shape)
            
            '''Then allocate random values using gaussian distribution to dummy Weight_delta'''
            self.reset_parameters()
            # print("Weight_delta shape after reset_parameters: ", self.Weight_delta.shape) 

            '''Decompose the dummy Weight_delta to high dimensional tensor based on the TT shapes'''
            self.Weight_TT_dimension = self.reshape_tensor(torch.tensor(self.Weight_delta))
            # print("Weight_TT_dimension shape: ", self.Weight_TT_dimension.shape)

            '''We have dummy weight decomposed into multiple tensors based on tt_shape
            Now, we create tensor cores as Parameters which are trainable
            Paramerter wraps the tensors into traninable parameters
            ParameterList holds the list of parameters
            TT Cores are initialized using standard normal distribution based on the ttcores shapes'''
            self.ttlora_cores = nn.ParameterList([nn.Parameter(self.initialize_cores(*shape)) for shape in self.get_ttcores_shapes()])
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
                self.ttlora_cores[i].data = torch.tensor(self.ttlora_cores_dummy[i], dtype=torch.float32)
        
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
                device = self.base_module.weight.device
                for i in range(len(self.ttlora_cores)):
                    self.ttlora_cores[i] = self.ttlora_cores[i].to(device)

                if len(self.tt_shape) == 4:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq->jlnp',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3]
                    )
                elif len(self.tt_shape) == 5:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq,qrs->jlnpr',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3], self.ttlora_cores[4]
                    )
                elif len(self.tt_shape) == 6:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq,qrs,stu->jlnprt',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3], self.ttlora_cores[4], self.ttlora_cores[5]
                    )
                elif len(self.tt_shape) == 7:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq,qrs,stu,uvw->jlnprtv',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3], self.ttlora_cores[4], self.ttlora_cores[5], self.ttlora_cores[6]
                    )
                elif len(self.tt_shape) == 8:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq,qrs,stu,uvw,wxy->jlnprtvx',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3], self.ttlora_cores[4], self.ttlora_cores[5], self.ttlora_cores[6], self.ttlora_cores[7]
                    )
                elif len(self.tt_shape) == 10:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc->jlnprtvxzb',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3],
                        self.ttlora_cores[4], self.ttlora_cores[5], self.ttlora_cores[6], self.ttlora_cores[7],
                        self.ttlora_cores[8], self.ttlora_cores[9]
                    )
                elif len(self.tt_shape) == 12:
                    self.ttlora_weights = torch.einsum(
                        'ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc,cde,efg->jlnprtvxzbdf',
                        self.ttlora_cores[0], self.ttlora_cores[1], self.ttlora_cores[2], self.ttlora_cores[3],
                        self.ttlora_cores[4], self.ttlora_cores[5], self.ttlora_cores[6], self.ttlora_cores[7],
                        self.ttlora_cores[8], self.ttlora_cores[9], self.ttlora_cores[10], self.ttlora_cores[11]
                    )

                self.ttlora_weights = self.ttlora_weights.to(device)
                self.alpha = torch.tensor(self.alpha, device=device, dtype=self.ttlora_weights.dtype)

                adapted_weight = self.base_module.weight + self.alpha * self.ttlora_weights.reshape(self.in_features_shape, self.out_features_shape)
                return F.linear(x.to(device), adapted_weight, self.base_module.bias)
            else:
                return F.linear(x.to(self.base_module.weight.device), self.base_module.weight, self.base_module.bias)
