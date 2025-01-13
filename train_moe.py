import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
import tensorflow.keras.backend as K

from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset
from create_experts import save_adapter_weights
from torch import nn
import math

from model import CustomLightningModule
from utils import get_tokenizer, load_local_dataset, get_ttlora_shape, get_ttlora_rank
from ttlora_wrapper import TTLoRALinearWrapper
from moe_wrapper import MoEsparseRouting
from create_experts import parse_expert_files, print_experts_details
# Set environment variables to suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tl.set_backend('pytorch')

dataset_name = "mrpc"
model_name = "roberta-base"
model_path = "./roberta-base/roberta-base-model"
tokenizer_path = "./roberta-base/roberta-base-tokenizer"

def initialize_router_weights(router_weights):
    '''Initialize the given tensor with random values from a gaussian distribution'''
    torch.manual_seed(10)
    nn.init.kaiming_uniform_(router_weights, a=math.sqrt(2))

def train_moe_without_ray(config):
    
    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")
    
    #load experts
    saved_adapters_path= config["saved_adapter_path"]
    experts = parse_expert_files(saved_adapters_path)
    # print_experts_details(experts)
    

    '''Load the model and and define the labels'''
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    # print("-"*50, "\nThe is how the model looks like :",model)
    # print("-"*50, "\nThis is how the model config looks like:", model.config)

    '''Make model parameters non-trainable and check if it is done correctly'''
    for param in model.parameters():
        param.requires_grad = False
    # print("-"*50, "\nThe model parameters are non-trainable now, For example:")
    # named_parameters = model.named_parameters()
    # for name, param in named_parameters:
    #     print(f"{name}: {param.requires_grad}")
    #     break
    
    in_features_shape, out_features_shape = model.roberta.encoder.layer[0].attention.self.value.weight.shape
    router_weights = nn.Parameter(torch.empty(in_features_shape, len(experts), requires_grad=True))
    initialize_router_weights(router_weights)
    # print("-"*50, f"\nRouter weights shape: {router_weights.shape}")
    # print(f"Router weights requires_grad: {router_weights.requires_grad}")

    if model_name == "roberta-base":
        model.roberta.encoder = MoEsparseRouting(model.roberta.encoder, experts, router_weights, config["common_alpha"]) #layer is sent as module
    # hidden_states = torch.randn(2, 2, in_features_shape)
    # model.roberta.encoder.forward(hidden_states)
    print("-"*50, "New TTLoRA adpated model looks like: \n", model)
    print("-" * 50, "\nModel parameters grad_require and shape:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    '''Dataset loading and check if loaded correctly'''
    dataset = load_dataset("glue", dataset_name)         
    print("-"*50, "This is how the dataset looks like\n", dataset)

    '''Tokenize data and check if correctly tokenized'''
    tokenized = get_tokenizer(tokenizer_path, dataset_name , dataset)

    '''Create train and validation dataset'''
    train_dataset = tokenized["train"]
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

    '''For trainig and evaluation'''
    lightning_model = CustomLightningModule(model, dataset_name, config["learning_rate"])
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
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
    # return {"Trainable_parameters_count": train_params}   #used when training is off to do some layer checking

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

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main()
    