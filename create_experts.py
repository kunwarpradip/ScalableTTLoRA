import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
from collections import defaultdict
import os
import sys

# sys.stdout = open('output_expert.log', 'w')
# sys.stderr = open('output_expert.log', 'w')

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
        print("-"*50,'\n',"Expert Type: ", task_type)
        # print(values.keys())
        for layer, layer_data in values.items():
            print("Layer:", layer)
            for attention_type, cores in layer_data.items():
                print(attention_type)
                # print(cores.keys())
                for core_name, tensor in cores.items():
                    print(core_name,":",tensor)
                    break
                    # print()
                    # if tensor.requires_grad:
                    #     print(f"Tensor {core_name} in {attention_type} of {layer} for task {task_type} requires grad.")
                    # else:
                    #     print(f"Tensor {core_name} in {attention_type} of {layer} for task {task_type} does not require grad.")
                    # break

if __name__== "__main__":
    # Load all adapter weights
    all_experts = parse_expert_files("./saved_adapters")
    print_experts_details(all_experts)
    
    # Create a dictionary of experts
    experts_dict = {
        "expert_1": {
            "layer_0": {
                "query": {"ttloracore_0": torch.randn(1, 1, 1), 
                          "ttloracore_1": torch.randn(1, 2, 1), 
                          "ttloracore_2": torch.randn(1, 3, 1), 
                          "ttloracore_3": torch.randn(1, 4, 1)}, 
                "value": {"ttloracore_0": torch.randn(1, 9, 2), 
                          "ttloracore_1": torch.randn(1, 10, 2), 
                          "ttloracore_2": torch.randn(1, 11, 2), 
                          "ttloracore_3": torch.randn(1, 12, 2)}
                        },
            "layer_1": {
                "query": {"ttloracore_0": torch.randn(2, 1, 1), 
                          "ttloracore_1": torch.randn(2, 2, 1), 
                          "ttloracore_2": torch.randn(2, 3, 1), 
                          "ttloracore_3": torch.randn(2, 4, 1)}, 
                "value": {"ttloracore_0": torch.randn(2, 9, 2), 
                          "ttloracore_1": torch.randn(2, 10, 2), 
                          "ttloracore_2": torch.randn(2, 11, 2), 
                          "ttloracore_3": torch.randn(2, 12, 2)}
                        },
            "layer_2": {
                "query": {"ttloracore_0": torch.randn(3, 1, 1), 
                          "ttloracore_1": torch.randn(3, 2, 1), 
                          "ttloracore_2": torch.randn(3, 3, 1), 
                          "ttloracore_3": torch.randn(3, 4, 1)}, 
                "value": {"ttloracore_0": torch.randn(3, 9, 2), 
                          "ttloracore_1": torch.randn(3, 10, 2), 
                          "ttloracore_2": torch.randn(3, 11, 2), 
                          "ttloracore_3": torch.randn(3, 12, 2)}
                        },
            },
        "expert_2": {
            "layer_0": {
                "query": {"ttloracore_0": torch.randn(1, 5, 1), 
                          "ttloracore_1": torch.randn(1, 6, 1), 
                          "ttloracore_2": torch.randn(1, 7, 1), 
                          "ttloracore_3": torch.randn(1, 8, 1)}, 
                "value": {"ttloracore_0": torch.randn(1, 13, 2), 
                          "ttloracore_1": torch.randn(1, 14, 2), 
                          "ttloracore_2": torch.randn(1, 15, 2), 
                          "ttloracore_3": torch.randn(1, 16, 2)}
                        },
            "layer_1": {
                "query": {"ttloracore_0": torch.randn(2, 5, 1), 
                          "ttloracore_1": torch.randn(2, 6, 1), 
                          "ttloracore_2": torch.randn(2, 7, 1), 
                          "ttloracore_3": torch.randn(2, 8, 1)}, 
                "value": {"ttloracore_0": torch.randn(2, 13, 2), 
                          "ttloracore_1": torch.randn(2, 14, 2), 
                          "ttloracore_2": torch.randn(2, 15, 2), 
                          "ttloracore_3": torch.randn(2, 16, 2)}
                        },
            "layer_2": {
                "query": {"ttloracore_0": torch.randn(3, 5, 1), 
                          "ttloracore_1": torch.randn(3, 6, 1), 
                          "ttloracore_2": torch.randn(3, 7, 1), 
                          "ttloracore_3": torch.randn(3, 8, 1)}, 
                "value": {"ttloracore_0": torch.randn(3, 13, 2), 
                          "ttloracore_1": torch.randn(3, 14, 2), 
                          "ttloracore_2": torch.randn(3, 15, 2), 
                          "ttloracore_3": torch.randn(3, 16, 2)}
                        },
                    },
                }
    for expert_name, expert_data in experts_dict.items():
        for layer_name, layer_data in expert_data.items():
            for attention_type, cores in layer_data.items():
                for core_name, tensor in cores.items():
                    # print(f"{expert_name} {layer_name} {attention_type}: {tensor.shape}")
                    pass
    
    
    layers = len(next(iter(all_experts.values())))
    # print("Number of layers:", layers)
    
    for layer in range(layers):
        # print("layer",layer)
        num_experts = len(all_experts)
        # print("Number of experts:", num_experts)
        access_first_expert = next(iter(all_experts.values()))
        # print(access_first_expert.keys())
        query_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer}"]["query"]))]
        # print(len(query_cores))
        for expert in all_experts.values():
            for i, (core_name, tensor) in enumerate(expert[f"layer_{layer}"]["query"].items()):
                # print(i, core_name, tensor.shape)
                query_cores[i].append(tensor)

        

        # print("length of query core list",len(query_cores))        
        # example_core = query_cores[0]
        # print("Example core length:", len(example_core))
        # # num_experts = example_core.shape[0]
        # # print("Number of experts:", num_experts)
        
        # Convert list of lists of tensors into list of tensors
        query_cores = [torch.stack(core_list) for core_list in query_cores]
        example_core = query_cores[7]
        # print("Example core shape:", example_core.shape)
        num_experts = example_core.shape[0]
        # print("Number of experts:", num_experts)

        for i, core_list in enumerate(query_cores):
            # print("*"*20, f"Core {i}:", "*"*20)
            # print(type(core_list), core_list.shape)
            pass
        break
        
   
    
    