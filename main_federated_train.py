import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from loading_llama import load_llama_base 
from integrate_fedalt import apply_fedalt_to_vlm
from loading_data import assign_tasks_to_8_clients
from server_aggregation import perform_fedalt_aggregation
from config import *

def train_one_client_vlm(client_id, model, processor, train_data):
    """
    Trains a single client on their local data (VLM Mode).
    Includes Novelty: Orthogonal Regularization.
    """
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    ORTHO_LAMBDA = 0.05 
    
    for epoch in range(LOCAL_EPOCHS):
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i : i + BATCH_SIZE]
      
            images = [b['image'] for b in batch]
            texts = [b['text'] for b in batch]
  
            prompts = [f"USER: <image>\nDescribe this image.\nASSISTANT: {t}" for t in texts]
            
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(DEVICE)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            task_loss = outputs.loss
            
            ortho_loss = 0.0
            count = 0
            for name, module in model.named_modules():
                if hasattr(module, "individual_lora") and hasattr(module, "row_lora"):

                    A_local = module.individual_lora.A
                    A_row = module.row_lora.A
                    
                    sim = F.cosine_similarity(A_local.flatten(), A_row.flatten(), dim=0)
                    
                    ortho_loss += torch.abs(sim)
                    count += 1
            
            if count > 0:
                ortho_loss = ortho_loss / count
            total_loss = task_loss + (ORTHO_LAMBDA * ortho_loss)
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    print(f"Client {client_id} finished. Task Loss: {task_loss.item():.4f} | Ortho Penalty: {ortho_loss.item():.4f}")
    
    local_weights = {k: v.cpu().clone() for k, v in model.state_dict().items() if "individual_lora" in k}
    return local_weights

def main():
    print(f"Loading VLM Base: {MODEL_ID}...")
    model, _ = load_llama_base()
    
    print("Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = apply_fedalt_to_vlm(model)
    
    # 2. Assign Data to Clients
    print("Partitioning Data...")
    client_datasets = assign_tasks_to_8_clients()
    
    # Dictionary to store persistent client states (Mixers + Local LoRAs)
    client_local_states = {i: None for i in range(NUM_CLIENTS)}

    # 3. Federated Training Loop
    for r in range(ROUNDS):
        print(f"\n--- STARTING FEDERATED ROUND {r+1} ---")
        round_local_weights = []
        
        for client_id in range(NUM_CLIENTS):
            # Load Client's Personal Brain (if it exists)
            if client_local_states[client_id] is not None:
                model.load_state_dict(client_local_states[client_id], strict=False)
            
            # Train Client
            print(f"Training Client {client_id}...")
            weights = train_one_client_vlm(client_id, model, processor, client_datasets[client_id])
            
            # Save Client's Personal Brain
            client_local_states[client_id] = {
                k: v.cpu().clone() for k, v in model.state_dict().items() 
                if "individual_lora" in k or "mixer" in k
            }
            round_local_weights.append(weights)

        # 4. Server Aggregation (Novelty: Modality-Aware Scaling is inside here)
        print("Server computing Rest-of-World (RoW) weights...")
        new_row_weights = perform_fedalt_aggregation(round_local_weights)

        # 5. Distribute Global Knowledge back to clients
        for client_id in range(NUM_CLIENTS):
            client_local_states[client_id].update(new_row_weights[client_id])

    print("\nFederated Training Finished!")
    torch.save(client_local_states, "fedalt_vlm_final.pt")
    print("Saved all client states to fedalt_vlm_final.pt")

if __name__ == "__main__":
    main()