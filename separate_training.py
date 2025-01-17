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
from moe_wrapper import MoEsparseRouting
from create_experts import parse_expert_files, print_experts_details

tl.set_backend('pytorch')
# Redirect stdout and stderr to a file
# sys.stdout = open('output.log', 'w')
# sys.stderr = open('output.log', 'w')

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
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    if model_name == "roberta-base":
        model.roberta.encoder = MoEsparseRouting(model.roberta.encoder, experts, config["common_alpha"])
    # print(model)
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(name, "requires_grad", param.requires_grad)

    for name, param in model.named_parameters():
        if name in ["roberta.encoder.router.router_layer.weight", "classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]:
            param.requires_grad = True
        else:
            param.requires_grad = False

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

    print("tokenized data looks like: ",tokenized)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    print("tokenized dataset looks like: ",tokenized["input_ids"],"attention looks like",tokenized["attention_mask"],"label looks like",tokenized["label"])
    
    '''Trying the New way of training the model'''
    input_ids, attention_mask, target = tokenized["input_ids"], tokenized["attention_mask"], tokenized["label"]

    def forward_hook(module, input, output):
        print(f"Forward hook for {module.__class__.__name__}")
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    print(f"Shape: {inp.shape}, grad_fn: {inp.grad_fn}")
        else:
            print(f"Shape: {input.shape}, grad_fn: {input.grad_fn}")
        if isinstance(output, torch.Tensor):
            print(f"Shape: {output.shape}, grad_fn: {output.grad_fn}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"Shape: {out.shape}, grad_fn: {out.grad_fn}")

    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)

        
    criterion = nn.CrossEntropyLoss()
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=config["learning_rate"])
    # target = torch.tensor([data_example["label"]])
    
    scores = model(input_ids, attention_mask=attention_mask)
    print(scores.logits)
    _, predicted = torch.max(scores.logits, dim=1)
    print("Predicted label index:", predicted.item())
    
    loss = criterion(scores.logits, target)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # optimizer.zero_grad()
    print("Before Loss backward")


    # for name, param in model.named_parameters():
    #     if param.grad_fn is not None:
    #         print(f"{name}: grad_fn = {param.grad_fn}")
    #     else:
    #         print(f"{name}: no grad_fn")
    # print(model.roberta.encoder.router.router_layer.weight.requires_grad)

    # loss.backward()
    # print("After Loss backward")
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name}: grad mean = {param.grad.abs().mean()}")
    #     else:
    #         print(f"{name}: no gradient")
    #     print("After Loss backward")
    #     optimizer.step()
   
    # def check_accuracy(loader, model):
    #     num_correct = 0
    #     num_samples = 0
    #     model.eval()
    #     with torch.no_grad():
    #         for x,y in loader:
    #             x=x.to("cuda")
    #             y=y.to("cuda")
    #             scores = model(x)
    #             _, preds = scores.logits.max(1)

    #             num_correct += (preds == y).sum()
    #             num_samples += preds.size(0)
    #     model.train()
    #     return num_correct/num_samples

    # model.to("cuda")
    # print(f"Acuracy on training set: {check_accuracy(tokenized, model)*100:.2f}")
                

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