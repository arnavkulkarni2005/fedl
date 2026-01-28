import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import evaluate
from loading_data import get_task_specific_data
from integrate_fedalt import apply_fedalt_to_vlm
from config import MODEL_ID, DEVICE, LORA_RANK
import numpy as np
import os

# --- CONFIGURATION ---
TEST_SAMPLES = 20        
SKIP_TRAIN_SAMPLES = 200 
# Point to the BIG 805MB file
COMBINED_CHECKPOINT = "/home/kulkarni/projects/fedl/fedl/checkpoints/fedalt_round_3.pt"

# --- CLIENT CONFIGURATION MAP ---
CLIENT_CONFIG = {
    0: {"task": "CommonSense", "metric": "ROUGE", "prompt": "<|user|>\n<|image_1|>\nWhat is happening in this scene?<|end|>\n<|assistant|>"},
    1: {"task": "Coreference", "metric": "ACCURACY", "prompt": "<|user|>\n<|image_1|>\nIdentify the main object/region.<|end|>\n<|assistant|>"},
    2: {"task": "NLI", "metric": "ACCURACY", "prompt": "<|user|>\n<|image_1|>\nIs the statement true based on the image? Answer Yes or No.<|end|>\n<|assistant|>"},
    3: {"task": "Paraphrase", "metric": "ROUGE", "prompt": "<|user|>\n<|image_1|>\nDescribe this image in detail.<|end|>\n<|assistant|>"},
    4: {"task": "ReadingComp", "metric": "ACCURACY", "prompt": "<|user|>\n<|image_1|>\nAnswer the question presented in the image/context.<|end|>\n<|assistant|>"},
    5: {"task": "Sentiment", "metric": "ACCURACY", "prompt": "<|user|>\n<|image_1|>\nIs this image offensive/hateful? Answer Yes or No.<|end|>\n<|assistant|>"},
    6: {"task": "StructToText", "metric": "ROUGE", "prompt": "<|user|>\n<|image_1|>\nSummarize the data shown in this chart.<|end|>\n<|assistant|>"},
    7: {"task": "TextClass", "metric": "ACCURACY", "prompt": "<|user|>\n<|image_1|>\nRead the text in the image.<|end|>\n<|assistant|>"}
}

def clean_answer(text):
    text = text.replace("{'answer':", "").replace("'answer':", "").replace("{'user':", "")
    text = text.replace("}", "").replace("]", "").replace("[", "").replace("'", "").replace('"', "")
    return text.strip().lower()

def run_benchmark():
    print(f"--- STARTING FINAL BENCHMARK (Single File Mode) ---")
    rouge = evaluate.load("rouge")
    
    # 1. Load Model
    print(f"Loading Base Model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map=DEVICE, 
        _attn_implementation='eager' 
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = apply_fedalt_to_vlm(model, rank=LORA_RANK)
    
    # 2. Load the BIG Checkpoint (Once)
    print(f"Loading Combined Checkpoint: {COMBINED_CHECKPOINT}...")
    if not os.path.exists(COMBINED_CHECKPOINT):
        print("CRITICAL: Checkpoint file not found!")
        return

    full_checkpoint = torch.load(COMBINED_CHECKPOINT, map_location=DEVICE)
    print("   Checkpoint loaded into RAM.")
    
    # Debug: Print structure to verify
    if isinstance(full_checkpoint, list):
        print(f"   Structure detected: LIST with {len(full_checkpoint)} elements.")
    elif isinstance(full_checkpoint, dict):
        keys = list(full_checkpoint.keys())
        print(f"   Structure detected: DICT with keys: {keys[:5]}...")
    
    results_table = []

    # 3. Iterate Clients
    for client_id, config in CLIENT_CONFIG.items():
        task_name = config["task"]
        metric_type = config["metric"]
        prompt = config["prompt"]
        
        print(f"\n" + "-"*60)
        print(f"Benchmarking Client {client_id} | Task: {task_name}")
        
        # --- EXTRACT CLIENT WEIGHTS ---
        client_state = None
        
        # Strategy A: It's a list (Index = Client ID)
        if isinstance(full_checkpoint, list):
            if client_id < len(full_checkpoint):
                client_state = full_checkpoint[client_id]
        
        # Strategy B: It's a dict (Key = Client ID)
        elif isinstance(full_checkpoint, dict):
            # Try all common key formats
            potential_keys = [client_id, str(client_id), f"client_{client_id}", f"client{client_id}"]
            for key in potential_keys:
                if key in full_checkpoint:
                    client_state = full_checkpoint[key]
                    break
        
        if client_state is None:
            print(f"   [!] Could not find weights for Client {client_id} in the file.")
            continue

        # Map internal keys (local/row) to model keys (individual/row)
        try:
            new_model_state = {}
            # If the checkpoint has 'local' and 'row' sub-dicts
            if 'local' in client_state and 'row' in client_state:
                for k, v in client_state['local'].items():
                    new_model_state[k.replace("local", "individual")] = v
                for k, v in client_state['row'].items():
                    new_model_state[k] = v
                model.load_state_dict(new_model_state, strict=False)
            else:
                # Maybe it's already flat
                model.load_state_dict(client_state, strict=False)
                
            print(f"   Weights loaded successfully.")
        except Exception as e:
            print(f"   [!] Error loading state dict: {e}")
            continue

        model.eval()
        
        # --- INFERENCE ---
        raw_data = get_task_specific_data(task_name, num_samples=SKIP_TRAIN_SAMPLES + TEST_SAMPLES)
        test_data = raw_data[SKIP_TRAIN_SAMPLES:]
        
        if not test_data:
            print("   [!] Not enough data.")
            continue
            
        predictions, references = [], []
        
        for i, item in enumerate(test_data):
            inputs = processor(prompt, item['image'], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, 
                    max_new_tokens=60, 
                    use_cache=False, 
                    pad_token_id=processor.tokenizer.pad_token_id, 
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            input_len = inputs.input_ids.shape[1]
            pred_raw = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]
            predictions.append(clean_answer(pred_raw))
            references.append(clean_answer(item['text']))

        # --- SCORING ---
        score = 0.0
        if metric_type == "ACCURACY":
            correct = sum([1 for p, r in zip(predictions, references) if r in p or p in r])
            score = (correct / len(predictions)) * 100
            print(f"   >> Accuracy: {score:.2f}%")
        elif metric_type == "ROUGE":
            if predictions:
                scores = rouge.compute(predictions=predictions, references=references)
                score = scores['rougeL'] * 100
                print(f"   >> ROUGE-L: {score:.2f}")

        results_table.append((client_id, task_name, score))

    print("\n" + "="*40 + "\nFINAL RESULTS\n" + "="*40)
    for res in results_table:
        print(f"Client {res[0]} ({res[1]}): {res[2]:.2f}")

if __name__ == "__main__":
    run_benchmark()