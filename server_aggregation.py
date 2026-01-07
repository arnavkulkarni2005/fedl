import torch
import copy

def perform_fedalt_aggregation(all_clients_individual_weights):
    """
    Implements Equation 3: A_k^R = (1 / K-1) * sum_{m != k} A_m^L
    
    Args:
        all_clients_individual_weights: A list of state_dicts. 
        Each state_dict contains the 'A_local' and 'B_local' weights from one client.
    """
    K = len(all_clients_individual_weights)
    row_weights_for_all_clients = []

    # 1. First, calculate the sum of ALL Individual LoRAs
    # (It's mathematically faster to sum all and subtract the current one)
    global_sum = {}
    for key in all_clients_individual_weights[0].keys():
        global_sum[key] = torch.stack([c[key] for c in all_clients_individual_weights]).sum(dim=0)

    # 2. For each client 'k', compute their specific RoW LoRA
    for k in range(K):
        client_k_row_dict = {}
        for key in global_sum.keys():
            # Equation 3: (Total_Sum - Client_k_Local) / (K - 1)
            # This results in the average of everyone EXCEPT client k
            row_val = (global_sum[key] - all_clients_individual_weights[k][key]) / (K - 1)
            
            # Rename the key from 'local' to 'row' so it matches the model architecture
            row_key = key.replace("individual_lora", "row_lora")
            client_k_row_dict[row_key] = row_val
            
        row_weights_for_all_clients.append(client_k_row_dict)

    return row_weights_for_all_clients