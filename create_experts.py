import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
from collections import defaultdict
import os

# Save TT-LoRA adapter weights
def save_adapter_weights(model, save_path):
    adapter_weights = {}
    for name, param in model.named_parameters():
        if "query.ttlora_cores" in name or "value.ttlora_cores" in name:  # Adjust condition for adapter-specific layers
            adapter_weights[name] = param.cpu().clone()
    torch.save(adapter_weights, save_path)
    print(f"Adapter weights saved at {save_path}")

def parse_expert_files(directory_path):
    """
    Parses all .pth files in a directory and organizes the experts into a nested dictionary.
    Args:
        directory_path (str): Path to the directory containing .pth files.
    Returns:
        dict: Nested dictionary containing organized experts for all task types.
    """
    # Nested dictionary to hold all tasks and their experts
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))

    # Iterate through all .pth files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pth"):
            task_name = filename.replace(".pth", "")  # Extract task type from filename
            file_path = os.path.join(directory_path, filename)

            # Load the .pth file
            expert_data = torch.load(file_path, map_location="cpu", weights_only=True)

            # Parse the expert data
            for full_key, tensor in expert_data.items():
                # Parse the key: 'roberta.encoder.layer.11.attention.self.value.ttlora_cores.7'
                tensor.requires_grad = False
                parts = full_key.split(".")
                
                # Extract components
                try:
                    layer = f"layer_{parts[3].split('_')[-1]}"  # Extract layer index
                    attention_type = parts[6]  # 'query' or 'value'
                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'
                    
                    # Insert tensor into the structured dictionary
                    all_experts[task_name][layer][attention_type][f'ttlora_core_{ttlora_core}'] = tensor
                except IndexError:
                    print(f"Skipping invalid key: {full_key}")
    return all_experts

def print_experts_details(all_experts):
    print(all_experts.keys())
    for task_type, values in all_experts.items():
        print("-"*50,'\n', task_type)
        print(values.keys())
        for layer, layer_data in values.items():
            print("-"*50,'\n',layer)
            print(layer_data.keys())
            for attention_type, cores in layer_data.items():
                print(attention_type)
                print(cores.keys())
                for core_name, tensor in cores.items():
                    print(core_name)
                    print(tensor.shape)
                    # if tensor.requires_grad:
                    #     print(f"Tensor {core_name} in {attention_type} of {layer} for task {task_type} requires grad.")
                    # else:
                    #     print(f"Tensor {core_name} in {attention_type} of {layer} for task {task_type} does not require grad.")
                    break
                
if __name__== "__main__":
    # Load all adapter weights
    all_experts = parse_expert_files("./saved_adapters")
    print_experts_details(all_experts)