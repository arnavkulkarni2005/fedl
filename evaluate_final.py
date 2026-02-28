import torch
import gc
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from loading_data import get_task_specific_data
from integrate_fedalt import apply_fedalt_to_vlm
from config import MODEL_ID, DEVICE, LORA_RANK

from transformers.cache_utils import DynamicCache


# --- SETTINGS ---
TEST_SAMPLES = 50        
SKIP_TRAIN_SAMPLES = 200
COMBINED_CHECKPOINT = "/home/kulkarni/projects/fedl/fedl/checkpoints/fedalt_final.pt"
OUTPUT_DIR = "benchmarking_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIENT_CONFIG = {
    0: {"task": "CommonSense", "metric_group": "nlg", "prompt": "<|user|>\n<|image_1|>\nWhat is happening in this scene?<|end|>\n<|assistant|>"},
    1: {"task": "Coreference", "metric_group": "clf", "prompt": "<|user|>\n<|image_1|>\nIdentify the main object/region.<|end|>\n<|assistant|>"},
    2: {"task": "NLI", "metric_group": "clf", "prompt": "<|user|>\n<|image_1|>\nIs the statement true based on the image? Answer Yes or No.<|end|>\n<|assistant|>"}
}

def clean_answer(text):
    if isinstance(text, dict):
        text = str(text.get('answer', text.get('text', next(iter(text.values())))))
    text = str(text).lower().strip()
    for noise in ["{'answer':", "'answer':", "{'user':", "}", "]", "[", "'", '"', "<|assistant|>", "<|end|>"]:
        text = text.replace(noise, "")
    return text.strip()

def evaluate_model(model, processor, rouge, mode="Final"):
    results = []
    for client_id, config in CLIENT_CONFIG.items():
        print(f"   -> Testing Client {client_id} | Task: {config['task']} | Mode: {mode}...")
        test_data = get_task_specific_data(config['task'], num_samples=SKIP_TRAIN_SAMPLES + TEST_SAMPLES)[SKIP_TRAIN_SAMPLES:]
        
        preds, refs = [], []
        for i, item in enumerate(test_data):
            print(f"      [Step {i+1}/{TEST_SAMPLES}] Processing...", end="\r")
            
            # Prepare inputs
            inputs = processor(config["prompt"], item['image'], return_tensors="pt").to(DEVICE)
            
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, "seen_tokens"):
                DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, 
                    max_new_tokens=30,
                    use_cache=False, 
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            input_len = inputs.input_ids.shape[1]
            pred_raw = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]
            preds.append(clean_answer(pred_raw))
            refs.append(clean_answer(item['text']))
            
            del inputs, gen_ids
            if i % 10 == 0: torch.cuda.empty_cache()

        if config["metric_group"] == "clf":
            score = (sum([1 for p, r in zip(preds, refs) if r in p]) / len(preds)) * 100
            label = "Accuracy"
        else:
            score = rouge.compute(predictions=preds, references=refs)['rougeL'] * 100
            label = "ROUGE-L"
            
        results.append({"Client": client_id, "Task": config["task"], "Model": mode, "Metric": label, "Score": score})
        print(f"\n      >> Done. Score: {score:.2f}")
    return results

def run_benchmark():
    rouge = evaluate.load("rouge")

    print("\n[PHASE 1] Zero-Shot Baseline (4-Bit Quantized)...")
    q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, quantization_config=q_config, 
        device_map=DEVICE, _attn_implementation='eager'
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    zs_results = evaluate_model(model, processor, rouge, mode="Zero-Shot")

    print("\n[PURGE] Clearing GPU for Phase 2...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 2: FEDALT (FULL PRECISION) ---
    print("\n[PHASE 2] FedALT Final (Bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, 
        device_map=DEVICE, _attn_implementation='eager'
    )
    model = apply_fedalt_to_vlm(model, rank=LORA_RANK)
    
    ckpt_results = []
    if os.path.exists(COMBINED_CHECKPOINT):
        full_ckpt = torch.load(COMBINED_CHECKPOINT, map_location=DEVICE)
        for cid in [0, 1, 2]:
            state = full_ckpt.get(cid) or full_ckpt.get(str(cid))
            if state:
                model.load_state_dict(state, strict=False)
                res = evaluate_model(model, processor, rouge, mode="FedALT Final")
                ckpt_results.append([r for r in res if r["Client"] == cid][0])
    
    # --- SAVE & PLOT ---
    df = pd.DataFrame(zs_results + ckpt_results)
    df.to_csv(f"{OUTPUT_DIR}/comparison.csv", index=False)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="Task", y="Score", hue="Model", palette="magma")
    plt.title("Evaluation: Zero-Shot vs. FedALT", fontsize=14, fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/final_plot.png", dpi=300)
    print(f"\n[FINISH] Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_benchmark()