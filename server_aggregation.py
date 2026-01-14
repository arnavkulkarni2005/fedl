import torch

def perform_fedalt_aggregation(all_clients_individual_weights):
    K = len(all_clients_individual_weights)
    row_weights_for_all_clients = []

    # 1. Sum all weights
    global_sum = {}
    for key in all_clients_individual_weights[0].keys():
        global_sum[key] = torch.stack([c[key] for c in all_clients_individual_weights]).sum(dim=0)

    # 2. Compute RoW with Modality Scaling
    for k in range(K):
        client_k_row_dict = {}
        for key in global_sum.keys():
            base_avg = (global_sum[key] - all_clients_individual_weights[k][key]) / (K - 1)
            
            if "vision_tower" in key:
                scaled_weight = base_avg * 1.2 
            else:
                scaled_weight = base_avg * 1.0
                
            row_key = key.replace("individual_lora", "row_lora")
            client_k_row_dict[row_key] = scaled_weight
            
        row_weights_for_all_clients.append(client_k_row_dict)

    return row_weights_for_all_clients