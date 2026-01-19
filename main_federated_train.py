# main_federated_train.py
import torch
import torch.nn.functional as F
import bitsandbytes as bnb # Using 8-bit optimizer for memory
from transformers import AutoProcessor
from loading_llama import load_phi3_vision_base 
from integrate_fedalt import apply_fedalt_to_vlm
from loading_data import assign_tasks_to_8_clients
from config import *
import matplotlib.pyplot as plt
import os
import gc

def aggregate_fedalt_server(client_local_states):
    """
    FedALT ROW (Rest-of-World) Logic:
    For each client, the ROW stream becomes the average of all OTHER clients' 
    Individual LoRA weights.
    """
    new_row_states = {}
    client_ids = list(client_local_states.keys())
    
    for target_cid in client_ids:
        # Collect individual lora weights from everyone EXCEPT the target client
        others = [client_local_states[cid] for cid in client_ids 
                  if cid != target_cid and client_local_states[cid] is not None]
        
        if not others:
            continue
            
        row_state = {}
        first_other = others[0]
        for key in first_other.keys():
            if "individual_lora" in key:
                # Map 'individual_lora' key from neighbors to 'row_lora' for this client
                target_key = key.replace("individual_lora", "row_lora")
                stacked = torch.stack([state[key] for state in others])
                row_state[target_key] = torch.mean(stacked, dim=0)
        
        new_row_states[target_cid] = row_state
        
    return new_row_states

def train_one_client_vlm(client_id, model, processor, train_data):
    model.train()
    # PagedAdamW8bit is essential for staying under 11GB VRAM
    optimizer = bnb.optim.PagedAdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    print(f"DEBUG: Client {client_id} has {len(train_data)} samples.")
    total_epoch_loss = 0
    num_steps = 0

    for epoch in range(LOCAL_EPOCHS):
        for step, item in enumerate(train_data):
            # Prompt template for Phi-3.5 Vision
            prompt = f"<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>\n{item['text']}<|end|>"
            inputs = processor(text=prompt, images=item['image'], return_tensors="pt").to(DEVICE)
            
            labels = inputs["input_ids"].clone()
            
            # Masking logic to prevent CUDA Assert Errors and improve quality
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            
            labels[labels == 32128] = -100 # Mask Phi-3.5 Vision Image Tokens
            
            # User Prompt Masking: Model only learns to predict the Assistant's text
            assistant_token_id = processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)[-1]
            for i in range(labels.shape[1]):
                if labels[0, i] == assistant_token_id:
                    labels[0, :i+1] = -100 
                    break
            
            outputs = model(**inputs, labels=labels)
            task_loss = outputs.loss
            
            # FedALT Orthogonal Regularization: Forces local expertise to differ from global context
            ortho_loss = 0.0
            for name, module in model.named_modules():
                if hasattr(module, "individual_lora") and hasattr(module, "row_lora"):
                    A_local = module.individual_lora.A.flatten().to(torch.float16)
                    A_row = module.row_lora.A.flatten().to(torch.float16)
                    ortho_loss += torch.abs(F.cosine_similarity(A_local, A_row, dim=0))
            
            total_loss = task_loss + (0.05 * ortho_loss)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_epoch_loss += task_loss.item()
            num_steps += 1

            if step % 5 == 0:
                print(f"    [Step {step}] Loss: {task_loss.item():.4f}")
                
    avg_loss = total_epoch_loss / num_steps if num_steps > 0 else 0
    # Return Individual LoRA and Mixer weights (we don't save the ROW weights locally)
    weights = {k: v.cpu().clone() for k, v in model.state_dict().items() 
               if "individual_lora" in k or "mixer" in k}
    return weights, avg_loss

def save_loss_plot(history, round_num):
    plt.figure(figsize=(10, 6))
    for client_id, losses in history.items():
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Client {client_id}')
    plt.title(f"FedALT Training History (Round {round_num})")
    plt.xlabel("Global Round")
    plt.ylabel("Avg Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"fedalt_loss_round_{round_num}.png")
    plt.close()

def quick_eval(model, processor, client_id, task_data):
    """
    Performs a single inference check while bypassing the DynamicCache error.
    """
    original_cache_state = model.config.use_cache
    original_mode = model.training
    
    try:
        model.eval()
        # FIX: Explicitly manage the cache to avoid 'seen_tokens' error
        model.config.use_cache = True 
        
        test_item = task_data[0]
        prompt = f"<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>"
        
        inputs = processor(text=prompt, images=test_item['image'], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # generate_ids logic with specific parameters for Phi-3.5 Vision
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=32,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                # Force use_cache to True here specifically
                use_cache=True 
            )
            
            prediction = processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
        print(f"\n--- [Client {client_id} Inference Check] ---")
        print(f"Target: {test_item['text']}")
        print(f"Model:  {prediction.strip()}\n")

    except Exception as e:
        # If the cache still fails, try one last time with use_cache=False
        print(f"[!] Quick Eval with cache failed: {e}. Retrying without cache...")
        try:
            model.config.use_cache = False
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=32, use_cache=False)
                prediction = processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                print(f"Model (No Cache): {prediction.strip()}\n")
        except Exception as e2:
            print(f"[!!] Total Eval Failure: {e2}")
    
    finally:
        # CRITICAL: Always reset to False for training or the next step will crash
        model.config.use_cache = False 
        if original_mode:
            model.train()

def main():
    # 1. Setup
    model, _ = load_phi3_vision_base()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = apply_fedalt_to_vlm(model)
    
    client_datasets = assign_tasks_to_8_clients()
    
    # State tracking
    client_local_states = {i: None for i in range(NUM_CLIENTS)}
    client_row_states = {i: None for i in range(NUM_CLIENTS)}
    history = {i: [] for i in range(NUM_CLIENTS)}

    for r in range(ROUNDS):
        print(f"\n" + "="*25 + f" GLOBAL ROUND {r+1} " + "="*25)
        
        for client_id in range(NUM_CLIENTS):
            # Load Client's Private Expertise
            if client_local_states[client_id]:
                model.load_state_dict(client_local_states[client_id], strict=False)
            
            # Load Server's Aggregated 'Rest-of-World' knowledge
            if client_row_states.get(client_id):
                model.load_state_dict(client_row_states[client_id], strict=False)
            
            print(f"Training Client {client_id}...")
            weights, avg_loss = train_one_client_vlm(client_id, model, processor, client_datasets[client_id])
            
            # CRITICAL: Store weights in CPU memory immediately
            if weights:
                client_local_states[client_id] = weights
                print(f"DEBUG: Weights captured for Client {client_id}")
            
            history[client_id].append(avg_loss)
            quick_eval(model, processor, client_id, client_datasets[client_id])
            
            # Manual memory cleanup to keep 1080 Ti stable
            gc.collect()
            torch.cuda.empty_cache()

        # 2. Server Aggregation: Prepare ROW weights for the NEXT round
        print(f"\n--- Server: Aggregating Rest-of-World knowledge ---")
        client_row_states = aggregate_fedalt_server(client_local_states)

        # 3. Intermediate Backups
        torch.save({'local': client_local_states, 'row': client_row_states}, f"fedalt_checkpoint_r{r+1}.pt")
        save_loss_plot(history, r+1)

    # Final Export
    print("\nTraining Complete. Saving Final Model...")
    torch.save(client_local_states, "fedalt_phi3_final.pt")

if __name__ == "__main__":
    main()