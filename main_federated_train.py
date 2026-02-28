import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import os
import gc

import time

from loading_phi3 import load_phi3_vision_base 
from integrate_fedalt import apply_fedalt_to_vlm
from loading_data import assign_tasks_to_3_clients
from server_aggregation import perform_fedalt_aggregation
from config import *

def setup_directories():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    print("DEBUG: Created 'checkpoints/' and 'plots/' directories.")

def save_loss_plot(history, round_num):
    plt.figure(figsize=(12, 8))
    for client_id, losses in history.items():
        if len(losses) > 0:
            rounds = range(1, len(losses) + 1)
            plt.plot(rounds, losses, marker='o', label=f'Client {client_id}')
    
    plt.title(f"FedALT Training Loss (Up to Round {round_num})")
    plt.xlabel("Global Round")
    plt.ylabel("Average Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f"plots/loss_round_{round_num}.png"
    plt.savefig(filename)
    plt.close()
    print(f"DEBUG: Loss plot saved to {filename}")

def train_one_client_vlm(client_id, model, processor, train_data):
    model.train()
    model.config.use_cache = False 
    
    optimizer = bnb.optim.PagedAdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    task_name = train_data[0].get('task', 'Unknown')
    print(f"   -> Client {client_id} [Task: {task_name}] | Samples: {len(train_data)}")
    
    total_epoch_loss = 0
    num_steps = 0
    
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_1|>")
    if image_token_id is None: image_token_id = 32044 
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|assistant|>")

    optimizer.zero_grad()

    for epoch in range(LOCAL_EPOCHS):
        for step, item in enumerate(train_data):
            try:
                # 1. Inputs
                prompt = f"<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>\n{item['text']}<|end|>"
                inputs = processor(text=prompt, images=item['image'], return_tensors="pt").to(DEVICE)
                
                # 2. Masking
                labels = inputs["input_ids"].clone()
                if processor.tokenizer.pad_token_id is not None:
                    labels[labels == processor.tokenizer.pad_token_id] = -100
                labels[labels == image_token_id] = -100 
                
                for b in range(labels.shape[0]):
                    matches = (labels[b] == assistant_token_id).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        start_idx = matches[-1] + 1 
                        labels[b, :start_idx] = -100
                
                # 3. Forward
                outputs = model(**inputs, labels=labels)
                task_loss = outputs.loss
                
                # 4. Orthogonal Loss
                ortho_loss = 0.0
                for name, module in model.named_modules():
                    if hasattr(module, "individual_lora") and hasattr(module, "row_lora"):
                        A_local = module.individual_lora.A.flatten()
                        A_row = module.row_lora.A.flatten()
                        ortho_loss += torch.abs(F.cosine_similarity(A_local, A_row, dim=0, eps=1e-8))
                
                total_loss = task_loss + (0.05 * ortho_loss)
                
                # 5. Backward (Accumulated)
                total_loss = total_loss / GRAD_ACCUM_STEPS
                total_loss.backward()
                
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
    
                total_epoch_loss += task_loss.item()
                num_steps += 1

                if step % 20 == 0:
                    print(f"      [Step {step}] Loss: {task_loss.item():.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"      [WARN] OOM at step {step}. Skipping.")
                    torch.cuda.empty_cache()
                else:
                    raise e
                
    avg_loss = total_epoch_loss / num_steps if num_steps > 0 else 0
    weights = {k: v.cpu().clone() for k, v in model.state_dict().items() 
               if "individual_lora" in k or "mixer" in k}
    
    return weights, avg_loss

def quick_eval(model, processor, client_id, task_data):
    model.eval()
    model.config.use_cache = False 
    try:
        test_item = task_data[0]
        prompt = f"<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>"
        inputs = processor(text=prompt, images=test_item['image'], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=30,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=False 
            )
        
        pred = processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        print(f"   [Eval C{client_id}]: {pred.strip()}")
        
    except Exception as e:
        print(f"   [!] Eval failed: {e}")
    finally:
        model.train()
        model.config.use_cache = False

def main():
    setup_directories()
    
    print("DEBUG: Loading Base Model...")
    model, _ = load_phi3_vision_base()

    processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    num_crops=4 
)
    model = apply_fedalt_to_vlm(model, rank=LORA_RANK)
    
    print("DEBUG: Loading Data...")
    client_datasets = assign_tasks_to_3_clients()
    
    client_local_states = {i: None for i in range(NUM_CLIENTS)}
    client_row_states = {i: None for i in range(NUM_CLIENTS)}
    history = {i: [] for i in range(NUM_CLIENTS)}

    for r in range(ROUNDS):
        start_time = time.time()
        print(f"\n{'='*20} GLOBAL ROUND {r+1}/{ROUNDS} {'='*20}")
        
        for client_id in range(NUM_CLIENTS):
            
            client_ckpt_path = f"checkpoints/round_{r+1}_client_{client_id}_done.pt"

            if os.path.exists(client_ckpt_path):
                print(f"   -> Found checkpoint for Client {client_id}. Loading and SKIPPING training...")
                
                saved_data = torch.load(client_ckpt_path)
                client_local_states[client_id] = saved_data['weights']
                history[client_id].append(saved_data['loss'])
                
                
                continue 

            if client_local_states[client_id]:
                model.load_state_dict(client_local_states[client_id], strict=False)
            if client_row_states[client_id]:
                model.load_state_dict(client_row_states[client_id], strict=False)
            
            weights, avg_loss = train_one_client_vlm(client_id, model, processor, client_datasets[client_id])
            
            client_local_states[client_id] = weights
            history[client_id].append(avg_loss)
            
            print(f"      [Checkpoint] Saving progress for Client {client_id}...")
            torch.save({
                'weights': weights,
                'loss': avg_loss,
                
            }, client_ckpt_path)
            
            quick_eval(model, processor, client_id, client_datasets[client_id])
            gc.collect()
            torch.cuda.empty_cache()

        print("\nDEBUG: Server Aggregating...")
        
        client_weights_list = [client_local_states[i] for i in range(NUM_CLIENTS)]
        
        if any(w is None for w in client_weights_list):
            print("CRITICAL: Some clients failed or data is missing. Skipping aggregation.")
            continue

        row_weights_list = perform_fedalt_aggregation(client_weights_list)
        client_row_states = {i: row_weights_list[i] for i in range(NUM_CLIENTS)}

        ckpt_path = f"checkpoints/fedalt_round_{r+1}.pt"
        torch.save({'local': client_local_states, 'row': client_row_states}, ckpt_path)
        save_loss_plot(history, r+1)
        
        print(f"DEBUG: Round {r+1} finished in {(time.time() - start_time)/60:.2f} mins.")

    print("\nTRAINING COMPLETE.")
    torch.save(client_local_states, "checkpoints/fedalt_final.pt")

if __name__ == "__main__":
    main()