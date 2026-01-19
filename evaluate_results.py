# evaluate_results.py
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from loading_llama import load_phi3_vision_base
from integrate_fedalt import apply_fedalt_to_vlm
from loading_data import get_task_specific_data, TASK_MAPPING
from config import LORA_RANK, PHI3_VISION_ID, DEVICE

def calculate_accuracy(model, processor, test_data):
    """
    Computes accuracy using Phi-3 Vision generation.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            # Update: Phi-3 specific prompt format
            prompt = f"<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>"
            
            inputs = processor(text=prompt, images=item['image'], return_tensors="pt").to(DEVICE)
            
            # Generate prediction
            outputs = model.generate(**inputs, max_new_tokens=32)
            # Decode only the assistant's part (skipping inputs)
            full_text = processor.decode(outputs[0], skip_special_tokens=True)
            prediction = full_text.split("assistant")[-1]

            if item['text'].strip().lower() in prediction.strip().lower():
                correct += 1
            total += 1

    return (correct / total) * 100 if total > 0 else 0

def generate_fedalt_table():
    tasks = list(TASK_MAPPING.keys())
    
    # 1. Initialize Phi-3 Model and FedALT
    model, _ = load_phi3_vision_base()
    model = apply_fedalt_to_vlm(model, rank=LORA_RANK)
    processor = AutoProcessor.from_pretrained(PHI3_VISION_ID, trust_remote_code=True)

    # 2. Load trained states
    print("Loading personalized weights...")
    try:
        client_states = torch.load("fedalt_phi3_final.pt")
    except:
        print("Final states not found. Using initialized weights.")
        client_states = {i: None for i in range(8)}

    results = {}
    for i, task_name in enumerate(tasks):
        print(f"\n--- Evaluating Client {i} on Task: {task_name} ---")
        
        # Load the personalized state for this client
        if client_states[i] is not None:
            model.load_state_dict(client_states[i], strict=False)
        
        # Evaluate on task-specific test data
        test_data = get_task_specific_data(task_name, num_samples=50)
        score = calculate_accuracy(model, processor, test_data)
        results[task_name] = score

    # --- PRINT THE RESULTS ---
    print("\n" + "="*50)
    print(f"{'Task Name':<20} | {'Sub-Dataset':<15} | {'Score':<10}")
    print("-" * 50)
    for task, score in results.items():
        subset = TASK_MAPPING[task]
        print(f"{task:<20} | {subset:<15} | {score:>8.2f}")
    print("="*50)

if __name__ == "__main__":
    generate_fedalt_table()