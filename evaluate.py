import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from config import MODEL_ID, DEVICE, SUBSET_NAME, DATASET_NAME
# Ensure your model architecture helper is imported
from integrate_fedalt import apply_fedalt_to_vlm 

def load_evaluation_model(checkpoint_path, client_id):
    """Loads base model and injects FedALT weights for a specific client."""
    print(f"Loading Base Model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map=DEVICE, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        _attn_implementation='eager' # Required for 1080 Ti
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 1. Apply the FedALT architecture (Parallel LoRAs + Mixer)
    model = apply_fedalt_to_vlm(model)

    # 2. Load the saved federated state
    print(f"Loading FedALT Weights from {checkpoint_path}")
    all_states = torch.load(checkpoint_path, map_location=DEVICE)
    
    # NEW: Safety check for NoneType
    if client_id not in all_states or all_states[client_id] is None:
        print(f"Error: No weight data found for Client {client_id} in {checkpoint_path}")
        print(f"Available clients with data: {[k for k,v in all_states.items() if v is not None]}")
        exit(1)
    
    model.load_state_dict(all_states[client_id], strict=False)
    model.eval()
    return model, processor

def run_accuracy_test(model, processor, dataset, num_samples=50):
    """Calculates Exact Match accuracy for VQA tasks."""
    correct = 0
    total = min(len(dataset), num_samples)
    
    print(f"Evaluating {total} samples...")
    model.config.use_cache = True # Enable for inference speed

    for i in tqdm(range(total)):
        item = dataset[i]
        image = item['image']
        # Extract question based on your dataset schema (e.g., The Cauldron/OK-VQA)
        question = item.get('question', "Describe this image.") 
        ground_truth = item.get('text', "").lower().strip()
        
        prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=32,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            # Decode only the generated part
            raw_pred = processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )[0]
            prediction = raw_pred.lower().strip()

        # Simple Exact Match or Keyword Match
        if prediction == ground_truth or ground_truth in prediction:
            correct += 1
            
    accuracy = (correct / total) * 100
    return accuracy

if __name__ == "__main__":
    CHECKPOINT = "/home/kulkarni/fedl/fedalt_phi3_final.pt"
    TEST_CLIENT = 1 # Check Client 0's personalized accuracy
    
    # Load model with weights
    model, processor = load_evaluation_model(CHECKPOINT, TEST_CLIENT)
    
    # You would typically load a 'test' split here
    # For now, we use the training set to verify if the 0.5 loss is real
    from main_federated_train import client_datasets 
    test_data = client_datasets[TEST_CLIENT]

    acc = run_accuracy_test(model, processor, test_data)
    print(f"\n--- Final Results ---")
    print(f"Client {TEST_CLIENT} Accuracy: {acc:.2f}%")