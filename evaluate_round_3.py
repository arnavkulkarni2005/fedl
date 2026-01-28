import torch
import os
import re
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm

# Import project modules
from config import MODEL_ID, DEVICE, LORA_RANK
from integrate_fedalt import apply_fedalt_to_vlm
from loading_data import get_task_specific_data

# --- CLIENT CONFIGURATION ---
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

def simple_rouge_1(prediction, reference):
    """
    A lightweight ROUGE-1 F1 score implementation (Unigram Overlap).
    """
    pred_tokens = re.findall(r'\w+', prediction.lower())
    ref_tokens = re.findall(r'\w+', reference.lower())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    
    overlap = len(pred_set.intersection(ref_set))
    
    precision = overlap / len(pred_set) if len(pred_set) > 0 else 0
    recall = overlap / len(ref_set) if len(ref_set) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1 * 100

def evaluate_client(model, processor, client_id, config, num_samples=30):
    """
    Runs evaluation for a single client using their specific task and metric.
    """
    task_name = config["task"]
    metric_type = config["metric"]
    custom_prompt = config["prompt"]
    
    test_data = get_task_specific_data(task_name, num_samples=num_samples)
    
    if not test_data:
        print(f"   [Warning] No data found for task {task_name}")
        return 0.0

    score_sum = 0
    count = 0
    
    # --- CRITICAL FIX: DISABLE CACHE ---
    model.eval()
    model.config.use_cache = False 
    # -----------------------------------
    
    print(f"   Evaluating {len(test_data)} samples (Metric: {metric_type})...")
    
    for item in tqdm(test_data, desc=f"Client {client_id}", leave=False):
        image = item['image']
        
        # Robust Ground Truth Handling
        raw_gt = item.get('text', "")
        if isinstance(raw_gt, dict):
            ground_truth = str(raw_gt).strip()
        else:
            ground_truth = str(raw_gt).strip()
        
        inputs = processor(text=custom_prompt, images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=64, 
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=False # CRITICAL FIX: Explicitly disable cache in generate
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = generate_ids[:, input_len:]
        prediction = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        
        if metric_type == "ACCURACY":
            is_correct = (ground_truth.lower() in prediction.lower()) or (prediction.lower() in ground_truth.lower())
            score_sum += 1 if is_correct else 0
            
        elif metric_type == "ROUGE":
            score_sum += simple_rouge_1(prediction, ground_truth)
            
        count += 1

    avg_score = score_sum / count if count > 0 else 0
    return avg_score

def main():
    CHECKPOINT_PATH = "checkpoints/fedalt_round_3.pt"
    
    print(f"--- FedALT Evaluation: Round 3 ---")
    
    print(f"Loading Base Model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation='eager'
    )
    # Ensure cache is disabled globally
    model.config.use_cache = False
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = apply_fedalt_to_vlm(model, rank=LORA_RANK)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"CRITICAL: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    print(f"Loading Weights from {CHECKPOINT_PATH}...")
    checkpoint_data = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    local_states = checkpoint_data.get('local', {})
    row_states = checkpoint_data.get('row', {})

    print(f"\n{'Client':<8} | {'Task':<15} | {'Metric':<10} | {'Score':<10}")
    print("-" * 55)
    
    for client_id, config in CLIENT_CONFIG.items():
        if client_id in local_states and local_states[client_id] is not None:
            model.load_state_dict(local_states[client_id], strict=False)
        if client_id in row_states and row_states[client_id] is not None:
            model.load_state_dict(row_states[client_id], strict=False)
            
        score = evaluate_client(model, processor, client_id, config, num_samples=5)
        
        print(f"{client_id:<8} | {config['task']:<15} | {config['metric']:<10} | {score:>6.2f}")

    print("-" * 55)
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()