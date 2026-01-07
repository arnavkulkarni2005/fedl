import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loading_llama import load_llama_base  #  file
from integrate_fedalt import apply_fedalt_to_llama #  file
from loading_data import assign_tasks_to_8_clients, get_task_specific_data #  file
from server_aggregation import perform_fedalt_aggregation #  file
from config import LOCAL_EPOCHS,BATCH_SIZE,LEARNING_RATE,ROUNDS,NUM_CLIENTS


def train_one_client(client_id, model, tokenizer, train_data):
    """Performs local training for a single client."""
    model.train()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Simple training loop for the client's subset
    for epoch in range(LOCAL_EPOCHS):
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i : i + BATCH_SIZE]
            
            # Format and tokenize
            prompts = [get_task_specific_data(ex) for ex in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass (Only updates Individual LoRA and Mixer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    print(f"Client {client_id} training complete. Loss: {loss.item():.4f}")
    
    # Extract only the TRAINABLE local weights to send to server
    local_weights = {k: v.cpu().clone() for k, v in model.state_dict().items() if "individual_lora" in k}
    return local_weights

def main():
    # 1. Setup Base Model and Data
    model, tokenizer = load_llama_base()
    model = apply_fedalt_to_llama(model)
    all_datasets = assign_tasks_to_8_clients() # Tasks assigned here!
    
    # Dictionary to keep track of each client's personalized Mixer and Individual LoRA
    # Since these stay with the client across rounds
    client_local_states = {i: None for i in range(NUM_CLIENTS)}

    for r in range(ROUNDS):
        print(f"\n--- STARTING FEDERATED ROUND {r+1} ---")
        round_local_weights = []

        for client_id in range(NUM_CLIENTS):
            # Load this specific client's state (Individual LoRA + Mixer) into the model
            if client_local_states[client_id] is not None:
                model.load_state_dict(client_local_states[client_id], strict=False)
            
            # Perform Local Training
            local_lora_weights = train_one_client(client_id, model, tokenizer, all_datasets[client_id])
            
            # Save the updated full local state (including the Mixer) for next round
            client_local_states[client_id] = {k: v.cpu().clone() for k, v in model.state_dict().items() 
                                              if "individual_lora" in k or "mixer" in k}
            
            round_local_weights.append(local_lora_weights)

        # 2. SERVER AGGREGATION (Equation 3)
        print("Server computing Rest-of-World (RoW) weights...")
        new_row_weights = perform_fedalt_aggregation(round_local_weights)

        # 3. UPDATE RoW LoRAs
        # For the next round, each client will have a new frozen RoW component
        for client_id in range(NUM_CLIENTS):
            client_local_states[client_id].update(new_row_weights[client_id])

    print("\nFederated Training Finished!")

    # Save the entire personalized dictionary (all 8 clients)
    torch.save(client_local_states, "fedalt_final_client_states.pt")
    print("Saved all client states to fedalt_final_client_states.pt")

if __name__ == "__main__":
    main()