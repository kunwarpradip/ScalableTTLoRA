import os
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F 
import math

custom_cache_dir = "../cache_hf/"

os.environ['HF_DATASETS_CACHE'] = custom_cache_dir

def get_ttlora_shape(ttlora_shape_from_config):
    ttlora_shape = ttlora_shape_from_config
    return ttlora_shape

def get_ttlora_rank(r, ttlora_shape):
    ttlora_rank = [1]
    for i in range(len(ttlora_shape)-1):
        ttlora_rank.append(r)
    ttlora_rank.append(1)
    return ttlora_rank

def load_local_dataset(data_name):
    path = "./data"
    data_path = os.path.join(path, data_name)
    dataset = load_dataset(data_path)
    return dataset

def get_tokenizer(path, data_name, dataset):

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_text(batch):
        # Truncation true = truncate the tokenized text to max_length
        # Padding true = pad the tokenized text to max_length
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "rte":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mnli":
            return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "stsb":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "wsc":
            return tokenizer(batch["text"], batch['span1_text'], batch['span2_text'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "winogrande":
            return tokenizer(batch["sentence"], batch['option1'], batch['option2'], batch['answer'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "ax":
            return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "multirc":
            return tokenizer(batch["paragraph"], batch['question'], batch['answer'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "boolq":
            return tokenizer(batch["question"], batch['passage'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "hellaswag":
            return tokenizer(batch["ind"], batch["activity_label"], batch['ctx_a'],batch['ctx_b'],batch['ctx'],batch['endings'],batch['source_id'],batch['split'],batch['split_type'], add_special_tokens=True, truncation=True, padding=True)

    # Map the words in the dataset to the token values of the loaded tokenizer
    # None batch size = process entire dataset as single batch
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 

    # # Show some examples of the tokenized data of the train set
    # print("---------------------------------------------------------")
    # print("\nThis is how it looks like after tokenization is done \n",tokenized)
    # for i in range(1):
    #     print(f"Example {i+1}:")
    #     print("Question: ", tokenized['train'][i]['question'],"\n")
    #     print("Sentence: ", tokenized['train'][i]['sentence'],"\n")
    #     print("Label (0 (not entailment) or 1 (entailment)): ", tokenized['train'][i]['label'],"\n")
    #     print("idx: ", tokenized['train'][i]['idx'],"\n")
    #     print("Input IDs: Question and Sentence Tokenized Together: ", (tokenized['train'][i]['input_ids']),"\n")
    #     print("Decoded Tokenized: ", tokenizer.decode(tokenized['train'][i]['input_ids']),"\n")
    #     print("Attention mask: ", (tokenized['train'][i]['attention_mask']),"\n")

    ### Which columns to keep?
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    # # Show an example after converting to tensors
    # print("---------------------------------------------------------")
    # print("\nThis is how it looks like after converting to tensors \n")
    # for i in range(1):
    #     print(f"Example {i+1}:")
    #     print("Input IDs: ", tokenized['train'][i]['input_ids'],"\n")
    #     print("Attention Mask: ", tokenized['train'][i]['attention_mask'],"\n")
    #     print("Label: ", tokenized['train'][i]['label'],"\n")

    return tokenized