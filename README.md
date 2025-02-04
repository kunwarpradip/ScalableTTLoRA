**Note: Models and tokenizers are supposed to be saved in the folder structure as ./models/{model_name}-models and ./models/{model_name}-tokenizer (the options for model_name are: "roberta-base", "llama-3-8b", "llama-3-70b"**

**Note: Best Trained Checkpoints are supposed to be save in the folder structure as ./trained_checkpoints/{model_name}/experts/**

**Note: The best checkpoints in the folder for each expert is supposed to be one but not multiple best checkpoints**

**Step 1: Train Individual Experts**
File: _train_individual_experts.py
Wrapper file: __TTLoRAWrapper_TensorMultiplication.py
Note: All query shape, value shape, rank, learning rate, alpha are set as per the TT-LoRA paper for roberta, llama-3-8b and for llama-3-70b the settings are same as of 8b
- Just before config, change model_name for which you would like to train the experts (the options for now are roberta-base, llama-3-8b, llama-3-70b)
- Change the config's dataset_name one of these: [mrpc, cola, qnli, sst2]
- Run the code repeatedly using different dataset_names to create multiple experts
- This should save the best checkpoint at f"./trained_checkpoints/{config["model_name"]}/experts/{config["dataset_name"]}
- These folders are supposed to store only one best checkpoint all the time; if you want to train the same model for same dataset which you have trained earlier then it creates multiple best checkpoints. So, make sure to delete other checkpoints and just keep a single best checkpoint

**Step 2: (optional-step of sanity checking) Test if the saved checkpoints are loaded successfully and cores are adapted to the base model successfully**
File: _test_individual_experts.py
Wrapper file: __TTLoRAWrapper_TensorMultiplication.py
- Note: All query shape, value shape, rank, learning rate, alpha are set as per the TT-LoRA paper for roberta, llama-3-8b and for llama-3-70b the settings are same as of 8b
- Just before config, change model_name for which you would like to train the experts (the options for now are roberta-base, llama-3-8b, llama-3-70b)
- Change the config's dataset_name one of these: [mrpc, cola, qnli, sst2]
- Run the code repeatedly using different dataset_names to check multiple experts

**Step 3: Training the Router for multiple tasks**
File: _train_ttmoe_multitask.py
Wrapper file: __TTMoE_multitask_wrapper.py
- You should have best checkpoints saved for the model you want to traint the router with
- change the model_name
- change the config: router_type, gumbel_temperature, learning rate
- chage the config: dataload_type (single to train on a single datatype, multiple to train on multitasking)
- change the config: if single dataload_type then change the dataset_name; if multiple dataload_type then change the multiple_datasets combination
- run the code
