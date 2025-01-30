import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
import math
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset, Dataset

from torch import nn
from model import CustomLightningModule
from utils import get_tokenizer, load_local_dataset, load_mixed_datasets
from one_ttlora_wrapper import TTLoRALinearWrapper
from eight_ttmoe_multitask_wrapper import MoEsparseRouting
from create_experts import parse_expert_files, print_experts_details, save_adapter_weights
import matplotlib.pyplot as plt

tl.set_backend('pytorch')
# Redirect stdout and stderr to a file
sys.stdout = open('output.log', 'w')
sys.stderr = open('output.log', 'w')

datasets = ["cola", "qnli", "sst2", "mrpc"]  # Define datasets to mix
model_name = "roberta-base"
model_path = "./roberta-base/roberta-base-model"
tokenizer_path = "./roberta-base/roberta-base-tokenizer"
# torch.autograd.set_detect_anomaly(True)

def apply_hooks(model):
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

def train_moe_without_ray(config):
    
    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load experts
    saved_adapters_path= config["saved_adapter_path"]
    experts = parse_expert_files(saved_adapters_path) #dictionary of experts

    '''Load the model and and define the labels'''
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    if model_name == "roberta-base":
        model.roberta.encoder = MoEsparseRouting(model.roberta.encoder, 
                                                experts, 
                                                config["common_alpha"],
                                                config["m_factors"], 
                                                config["n_factors"],
                                                train_router_only=True,
                                                device=device)
        
    # apply_hooks(model)

    for name, param in model.named_parameters():
        if "router" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    '''Dataset loading and check if loaded correctly'''
    train_dataset_dict, val_dataset_dict = load_mixed_datasets(datasets, tokenizer_path)
    
    train_dataset = Dataset.from_dict(train_dataset_dict)
    val_dataset = Dataset.from_dict(val_dataset_dict)

    # print("train dataset feature types", type(train_dataset["input_ids"]), type(train_dataset["attention_mask"]), type(train_dataset["label"]))
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print("Train dataset shapes: ", train_dataset['input_ids'].shape, train_dataset["input_ids"].dtype, train_dataset['attention_mask'].shape, train_dataset['label'].shape)
    print("Validation dataset shapes: ", val_dataset['input_ids'].shape, val_dataset['attention_mask'].shape, val_dataset['label'].shape)

    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128, #max of roberta base is 512 tokens
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=4   #separate subprocesses to load data in parallel
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=128,
        num_workers=4
        #no need to shuffle the validation data as to get the consistent evaluations
    )

    '''For trainig and evaluation'''
    lightning_model = CustomLightningModule(model, 
                                            datasets, 
                                            config["learning_rate"])
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=True,
        mode='min'
        )

    '''Callback provided by PyTorch Lightning that allows to save model checkpoints during training'''
    model_checkpoint_callback=ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc")  

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        # enable_progress_bar=True,
        # enable_model_summary=False, 
        log_every_n_steps=10,
    )
    
    start = time.time()
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                #checkpoint_path is used to load the model from the checkpoint to resume the training
                # ckpt_path="./lightning_logs/version_2/checkpoints/epoch=0-step=819.ckpt"
                )
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    '''Model Testing in test and validation datasets'''
    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc=trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    print("-"*50, "Training Accuracy: ", train_acc, "Validation Accuracy: ", val_acc)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)

    return {"Taining_Accuracy": train_acc[0]['accuracy'], "Validation_Accuray": val_acc[0]['accuracy'], "Trainable_parameters_count": train_params}

def main():
    config = {
        "saved_adapter_path": "./saved_adapters_new",
        "common_alpha" : 8,
        "learning_rate":  1e-2,
        "m_factors": [12,8,8],
        "n_factors": [2,2,2,8,12]
    }
    
    analysis =  train_moe_without_ray(config)
    df = pd.DataFrame(list(analysis.items()), columns=['metric', 'value'])
    print(df)
    filename = f"MoE_multitask_{model_name}_test.csv"
    df.to_csv(filename, index=False)
    # train_moe_without_ray(config)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main()