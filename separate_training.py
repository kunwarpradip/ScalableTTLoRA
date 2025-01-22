import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os

from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset
from create_experts import save_adapter_weights
from torch import nn
import math
import sys
from datasets import Dataset
from transformers import AutoTokenizer

from model import CustomLightningModule
from utils import get_tokenizer, load_local_dataset, get_ttlora_shape, get_ttlora_rank
from ttlora_wrapper import TTLoRALinearWrapper
from moe_wrapper_separate import MoEsparseRouting
from create_experts import parse_expert_files, print_experts_details

tl.set_backend('pytorch')
# Redirect stdout and stderr to a file
sys.stdout = open('output.log', 'w')
sys.stderr = open('output.log', 'w')

dataset_name = "cola" 
model_name = "roberta-base"
model_path = "./roberta-base/roberta-base-model"
tokenizer_path = "./roberta-base/roberta-base-tokenizer"
torch.autograd.set_detect_anomaly(True)

def train_moe_without_ray(config):
    
    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")
    
    #load experts
    saved_adapters_path= config["saved_adapter_path"]
    experts = parse_expert_files(saved_adapters_path)

    '''Load the model and and define the labels'''
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=True)
    model.config.pad_token_id = model.config.eos_token_id

    if model_name == "roberta-base":
        model.roberta.encoder = MoEsparseRouting(model.roberta.encoder, experts, config["common_alpha"])
    

    for name, param in model.named_parameters():
        if "router" in name or "classifier.out_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    data_example = {
        "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
        "label": 1, 
        "id": 0,
        } 
    print("-"*25, "This is how the dataset looks like","-"*25,"\n", data_example, "\n", "-"*100,)
    dataset = Dataset.from_dict({"sentence": [data_example["sentence"]], "label": [data_example["label"]], "id": [data_example["id"]]})
    
    def tokenize_function(examples):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(examples["sentence"], add_special_tokens=True, truncation=True, padding=True)
    tokenized = dataset.map(tokenize_function, batched=True, batch_size=None) 

    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    print("-"*25,"tokenized dataset looks like: ","-"*25,"\n",tokenized["input_ids"],"attention looks like",tokenized["attention_mask"],"label looks like",tokenized["label"])
    
    '''Trying the New way of training the model'''
    input_ids, attention_mask, target = tokenized["input_ids"], tokenized["attention_mask"], tokenized["label"]
    grad_param = []

    print("*"*10,"Before Forward Pass; Look into the grads and grad_fn of parameters whose requires_grad is True","*"*10)
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_param.append(param)
        if param.requires_grad:
            print(name, ", grad:",param.grad, ", grad_fun:", param.grad_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(grad_param, lr=config["learning_rate"])

    def forward_hook(module, input, output):
        if not hasattr(forward_hook, "called"):
            forward_hook.called = True
            print("*"*10,"Inside forward hook; Looking into grad and grad_fn whose grad_fn is not None","*"*10)
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if inp.grad_fn is not None:
                        print(f"{"Module Name", module.__class__.__name__}")
                        print(f"Shape: {inp.shape}, grad_fn: {inp.grad_fn}")
        else:
            if input.grad_fn is not None:
                print(f"{"Module Name", module.__class__.__name__}")
                print(f"Shape: {input.shape}, grad_fn: {input.grad_fn}")
        if isinstance(output, torch.Tensor):
            if output.grad_fn is not None:
                print(f"{"Module Name", module.__class__.__name__}")
                print(f"Shape: {output.shape}, grad_fn: {output.grad_fn}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if out.grad_fn is not None:
                        print(f"{"Module Name", module.__class__.__name__}")
                        print(f"Shape: {out.shape}, grad_fn: {out.grad_fn}")

    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)

    def backward_hook(module, grad_input, grad_output):
        if not hasattr(backward_hook, "called"):
            backward_hook.called = True
            print("*"*10,"Inside Backward hook; Looking into input and output grad at layers with requires_grad as True","*"*10)
        if any(param.requires_grad for param in module.parameters()):
            for i, grad in enumerate(grad_input):
                print(f"Module: {module.__class__.__name__}")
                print(f"Grad input: {grad}")
            for i, grad in enumerate(grad_output):
                print(f"Module: {module.__class__.__name__}")
                print(f"Grad Output: {grad}")
    
    for name, module in model.named_modules():
        module.register_full_backward_hook(backward_hook)

    # print("Router weights gradient before forward:", model.roberta.encoder.router.router_layer.weight.grad)
    scores = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # print("Router weights gradient after forward:", model.roberta.encoder.router.router_layer.weight.grad)

    print("*"*10,"After Forward Pass; Look into the grads and grad_fn of params with require_grad=True","*"*10)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, ", grad:",param.grad, ", grad_fun:", param.grad_fn)


    print("Scores logits",scores.logits)
    _, predicted = torch.max(scores.logits, dim=1)
    print("*"*15,"Predicted label index:", predicted.item(), "*"*15)
    
    loss = criterion(scores.logits, target)
    print("*"*15,"Before Loss backward, Loss is :", loss,"*"*15)
    print("*"*25,"Performing Loss backward","*"*25)
    
    # print("Router weights gradient before loss backward:", model.roberta.encoder.router.router_layer.weight.grad)

    loss.backward()

    # print("Router weights gradient after loss backward:", model.roberta.encoder.router.router_layer.weight.grad)

    print("*"*25,"After Loss backward","*"*25)
    print("Grad values and param values whose grad in not None")
    for i, param in model.named_parameters():
        if param.grad is not None:
            print(i, "param's grad values \n", param.shape, param.grad, param.grad_fn)
            print(param)
    
    # print("Router weights gradient before optimizer step:", model.roberta.encoder.router.router_layer.weight.grad)

    optimizer.step()

    # print("Router weights gradient after optimizer step:", model.roberta.encoder.router.router_layer.weight.grad)

    print("*"*25,"After Optimizer Step","*"*25)
    print("Grad values and param values whose grad in not None")
    for i, param in model.named_parameters():
        if param.grad is not None:
            print(i, "param's grad values \n", param.shape, param.grad, param.grad_fn)
            print(param)

    optimizer.zero_grad()
    print("*"*25,"After Zero Grad, Printing grads values of params whose requires_grad is True","*"*25)

    for i, param in model.named_parameters():
        if param.requires_grad:
            print(i, "param's grad values \n", param.shape, param.grad, param.grad_fn)

def main():
    config = {
        "saved_adapter_path": "./saved_adapters",
        "common_alpha" : 8,
        "learning_rate":  1e-3,}
    
    train_moe_without_ray(config)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main()