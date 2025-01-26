import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time

from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset
from create_experts import save_adapter_weights

from model import CustomLightningModule
from utils import get_tokenizer, get_ttlora_shape, get_ttlora_rank
from ttlora_wrapper import TTLoRALinearWrapper

tl.set_backend('pytorch')

dataset_name = "mrpc"
model_name = "roberta-base"
model_path = "./roberta-base/roberta-base-model"
tokenizer_path = "./roberta-base/roberta-base-tokenizer"

def train_without_ray(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")
    
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

    '''Load the model and and define the labels'''
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    # print("-"*50, "The is how the model looks like :\n",model)
    # print("-"*50, "This is how the model config looks like: \n", model.config)

    '''Make model parameters non-trainable and check if it is done correctly'''
    for param in model.parameters():
        param.requires_grad = False
    # print("-"*50, "The model parameters are non-trainable now: \n")
    # named_parameters = model.named_parameters()
    # for name, param in named_parameters:
    #     print(f"{name}: {param.requires_grad}")

    '''Define the shape,rank and other configuration of the tensor train decomposition'''
    ttlora_shape = get_ttlora_shape(config["shape"])
    ttlora_rank = get_ttlora_rank(config["rank"], ttlora_shape)
    ttlora_alpha = config["alpha"]
    
    '''Define where to adapt the ttlora'''
    ttlora_adapter_at_query = True
    ttlora_adapter_at_value = True

    '''Assign TTLoRA adapters to the model where defined'''
    # partial is used to fix the arguments (shape, rank and alpha in this case) of the function
    assign_ttlora = partial(TTLoRALinearWrapper, tt_shape=ttlora_shape, tt_rank=ttlora_rank, alpha=ttlora_alpha)

    if model_name == "roberta-base":
        for layer in model.roberta.encoder.layer:
            if ttlora_adapter_at_query:
                layer.attention.self.query = assign_ttlora(layer.attention.self.query) #layer is sent as module
            if ttlora_adapter_at_value:
                layer.attention.self.value = assign_ttlora(layer.attention.self.value) #layer is sent as module
    
    if model_name == "llama2-7b":
        for layer in model.model.layers:
            if ttlora_adapter_at_query:
                layer.self_attn.q_proj = assign_ttlora(layer.self_attn.q_proj)
            if ttlora_adapter_at_value:
                layer.self_attn.v_proj = assign_ttlora(layer.self_attn.k_proj)
    print("-"*50, "New TTLoRA adpated model looks like: \n", model)

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

    # Example usage after fine-tuning
    save_adapter_weights(model, f"./saved_adapters/task_{dataset_name}.pth")

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
        "shape": [12, 8, 8, 2, 2, 2, 8, 12],  #roberta shape = 768x768 (attention head shape)
        # "shape": [16, 8, 8, 4, 4, 8, 8, 16],  #llama shape = 4096x4096 (attention head shape)
        "rank": 4,
        "alpha": 8,
        "learning_rate":  1e-3,
    }
    
    analysis =  train_without_ray(config)
    df = pd.DataFrame(list(analysis.items()), columns=['metric', 'value'])
    print(df)
    filename = f"{dataset_name}_{model_name}.csv"
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main()
    
