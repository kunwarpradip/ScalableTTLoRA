import os
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F 
import math

custom_cache_dir = "~/.cache/huggingface/"

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
    path = "./data/"
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

    # Show some examples of the tokenized data of the train set
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

    ### change the format into tensors of the specific columns
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    # # Show an example after converting to tensors
    # print("---------------------------------------------------------")
    # print("\nThis is how it looks like after converting to tensors \n")
    # print("Input IDs: ", tokenized['train']['input_ids'].shape,"\n")
    # print("Attention Mask: ", tokenized['train']['attention_mask'].shape,"\n")
    # print("Label: ", tokenized['train']['label'].shape,"\n")

    return tokenized

def get_mix_tokenizer(path, data_name, dataset):

    tokenizer = AutoTokenizer.from_pretrained(path)
    
    def tokenize_text(batch):
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "rte":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "mnli":
            return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "stsb":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "wsc":
            return tokenizer(batch["text"], batch['span1_text'], batch['span2_text'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "winogrande":
            return tokenizer(batch["sentence"], batch['option1'], batch['option2'], batch['answer'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "ax":
            return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "multirc":
            return tokenizer(batch["paragraph"], batch['question'], batch['answer'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "boolq":
            return tokenizer(batch["question"], batch['passage'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
        if data_name == "hellaswag":
            return tokenizer(batch["ind"], batch["activity_label"], batch['ctx_a'],batch['ctx_b'],batch['ctx'],batch['endings'],batch['source_id'],batch['split'],batch['split_type'], add_special_tokens=True, truncation=True, padding='max_length', max_length=512)

    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def load_mixed_datasets(dataset_names, tokenizer_path):
    '''Dataset loading and check if loaded correctly'''
    mixed_train_dataset_dict = {
        
        "input_ids": torch.empty(0,dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    mixed_validation_dataset_dict = {
        "input_ids": torch.empty(0, dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    for dataset_name in dataset_names:
        print("Loading dataset: ", dataset_name)
        dataset = load_dataset("glue",dataset_name)
        tokenized = get_mix_tokenizer(tokenizer_path, dataset_name , dataset)
        train_tokenized_dataset = tokenized["train"]
        train_tokenized_dataset = train_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Train tokenized dataset: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)
        validation_tokenized_dataset = tokenized["validation"]
        validation_tokenized_dataset = validation_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Validation tokenized dataset: ", validation_tokenized_dataset['input_ids'].shape, validation_tokenized_dataset['attention_mask'].shape, validation_tokenized_dataset['label'].shape)

        #########################################For Train###################################################
        mixed_train_dataset_dict["input_ids"] = torch.cat((mixed_train_dataset_dict["input_ids"], 
                                                           train_tokenized_dataset["input_ids"]), dim=0)
        mixed_train_dataset_dict["attention_mask"] = torch.cat((mixed_train_dataset_dict["attention_mask"], 
                                                                train_tokenized_dataset["attention_mask"]), dim=0)
        mixed_train_dataset_dict["label"] = torch.cat((mixed_train_dataset_dict["label"], 
                                                       train_tokenized_dataset["label"]), dim=0)
        #########################################For Validation###################################################

        mixed_validation_dataset_dict["input_ids"] = torch.cat((mixed_validation_dataset_dict["input_ids"], 
                                                                validation_tokenized_dataset["input_ids"]), dim=0)
        mixed_validation_dataset_dict["attention_mask"] = torch.cat((mixed_validation_dataset_dict["attention_mask"], 
                                                                     validation_tokenized_dataset["attention_mask"]), dim=0)
        mixed_validation_dataset_dict["label"] = torch.cat((mixed_validation_dataset_dict["label"], 
                                                            validation_tokenized_dataset["label"]), dim=0)
    
    print("mixed_train_dataset_dict: ", 
          mixed_train_dataset_dict['input_ids'].shape, 
          mixed_train_dataset_dict['attention_mask'].shape, 
          mixed_train_dataset_dict['label'].shape,
          "Data types: ", 
          mixed_train_dataset_dict['input_ids'].dtype, 
          mixed_train_dataset_dict['attention_mask'].dtype, 
          mixed_train_dataset_dict['label'].dtype
          )
    print("mixed_validation_dataset_dict: ", 
          mixed_validation_dataset_dict['input_ids'].shape, 
          mixed_validation_dataset_dict['attention_mask'].shape, 
          mixed_validation_dataset_dict['label'].shape,
          "Data types: ", 
          mixed_validation_dataset_dict['input_ids'].dtype, 
          mixed_validation_dataset_dict['attention_mask'].dtype, 
          mixed_validation_dataset_dict['label'].dtype
          )
    print(mixed_train_dataset_dict['label'].shape, mixed_train_dataset_dict['label'])
    return mixed_train_dataset_dict, mixed_validation_dataset_dict

