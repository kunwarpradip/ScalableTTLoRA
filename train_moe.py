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

from model import CustomLightningModule
from utils import get_tokenizer, load_local_dataset, get_ttlora_shape, get_ttlora_rank
from ttlora_wrapper import TTLoRALinearWrapper
from moe_wrapper import MoEsparseRouting
from create_experts import parse_expert_files, print_experts_details

tl.set_backend('pytorch')
# Redirect stdout and stderr to a file
sys.stdout = open('output_with_exp_names.log', 'w')
sys.stderr = open('output_with_exp_names.log', 'w')

dataset_name = "mrpc" 
model_name = "roberta-base"
model_path = "./roberta-base/roberta-base-model"
tokenizer_path = "./roberta-base/roberta-base-tokenizer"
# torch.autograd.set_detect_anomaly(True)

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

    for name, param in model.named_parameters():
        if name in ["roberta.encoder.router.router_layer.weight", "classifier.out_proj.weight"]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # def backward_hook(m, grad_input, grad_output):
    #     print(f"Grad Input: {grad_input[0][:5]}... Grad Output: {grad_output[:5]}...")

    def forward_hook(module, input, output):
        print(f"Forward hook for {module.__class__.__name__}")
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    print(f" Input Shape: {inp.shape}, grad_fn: {inp.grad_fn}")
        else:
            print(f"Input Shape: {input.shape}, grad_fn: {input.grad_fn}")
        if isinstance(output, torch.Tensor):
            print(f"Output Shape: {output.shape}, grad_fn: {output.grad_fn}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"Output Shape: {out.shape}, grad_fn: {out.grad_fn}")

    for name, module in model.named_modules():
        # module.register_forward_hook(forward_hook)
        if name == "roberta.encoder.router.router_layer":
            module.register_forward_hook(forward_hook)
    
    # model.roberta.encoder.router.router_layer.register_forward_hook(forward_hook)
    # model.roberta.encoder.router.router_layer.register_full_backward_hook(backward_hook)
    # model.classifier.out_proj.register_forward_hook(forward_hook)

    def load_local_dataset(data_name):
        path = "../../MoE_OthersWorks/llmCourse_MoE/required_codes/data"
        data_path = os.path.join(path, data_name)
        dataset = load_dataset(data_path)
        return dataset

    '''Dataset loading and check if loaded correctly'''
    # dataset = load_dataset("glue", dataset_name)
    dataset = load_local_dataset(dataset_name)         
    print("-"*25, "This is how the dataset looks like","-"*25,"\n", dataset, "\n", "-"*100,)

    '''Tokenize data and check if correctly tokenized'''
    tokenized = get_tokenizer(tokenizer_path, dataset_name , dataset)
    print(tokenized)
    '''Create train and validation dataset'''
    
    train_dataset = tokenized["train"]
    print("Example of train_dataset:")
    print(train_dataset)
    
    val_dataset = tokenized["validation"]

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
    
    # '''Trying the New way of training the model'''
    # model.to("cuda")
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # for epoch in range(10):
    #     for batch_idx, batch in enumerate(train_loader):
    #         data, target = batch["input_ids"], batch["label"]
    #         data=data.to("cuda")
    #         target=target.to("cuda")

    #         scores = model(data)
    #         loss = criterion(scores.logits, target)
    #         optimizer.zero_grad()
    #         print("Before Loss backward")
    #         loss.backward()
    #         for name, param in model.named_parameters():
    #             if param.grad is not None:
    #                 print(f"{name}: grad mean = {param.grad.abs().mean()}")
    #             else:
    #                 print(f"{name}: no gradient")
    #         print("After Loss backward")
    #         optimizer.step()
   
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
    # print(f"Acuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
                
    '''For trainig and evaluation'''
    lightning_model = CustomLightningModule(model, dataset_name, config["learning_rate"])
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
        accelerator="cuda",
        precision="16-mixed",
        devices=args.gpus,
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
        "saved_adapter_path": "./saved_adapters",
        "common_alpha" : 8,
        "learning_rate":  1e-3,
    }
    
    analysis =  train_moe_without_ray(config)
    df = pd.DataFrame(list(analysis.items()), columns=['metric', 'value'])
    print(df)
    filename = f"{dataset_name}_{model_name}_moe_test.csv"
    df.to_csv(filename, index=False)
    # train_moe_without_ray(config)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main()
    