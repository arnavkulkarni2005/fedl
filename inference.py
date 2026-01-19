import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import io

# Import your project modules
from integrate_fedalt import apply_fedalt_to_vlm
from config import MODEL_ID, DEVICE, LORA_RANK

def run_test():
    # 1. Load Base Model (Frozen)
    print(f"--- Loading Base Model: {MODEL_ID} ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        _attn_implementation='eager',
        device_map=DEVICE
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2. Re-create the Architecture
    model = apply_fedalt_to_vlm(model, rank=LORA_RANK)

    # 3. Load Your Trained Weights
    checkpoint_path = "checkpoints/fedalt_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL ERROR: {checkpoint_path} not found!")
        return

    print(f"--- Loading FedALT Weights from {checkpoint_path} ---")
    all_client_states = torch.load(checkpoint_path, map_location=DEVICE)

    # 4. Load Local Image
    print("--- Loading Local Image... ---")
    image_path = "test_image.jpg"  # Ensure you downloaded this file manually!
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found. Please download an image and name it 'test_image.jpg'.")
        return
        
    image = Image.open(image_path)

    # 5. Compare Client Personalities
    test_clients = [0, 1]
    
    prompt = "<|user|>\n<|image_1|>\nDescribe the main activity in this image.<|end|>\n<|assistant|>"
    inputs = processor(prompt, image, return_tensors="pt").to(DEVICE)

    print("\n" + "="*50)
    print("      FEDERATED DUAL-STREAM COMPARISON      ")
    print("="*50)

    for client_id in test_clients:
        print(f"\n[Switching to Client {client_id} Weights...]")
        
        # Load specific Individual LoRA + Mixer
        model.load_state_dict(all_client_states[client_id], strict=False)
        model.eval()
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=80,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=False 
            )

        # CRITICAL FIX: Slice off the input tokens before decoding!
        # This prevents the tokenizer from choking on the negative Image IDs.
        input_token_len = inputs.input_ids.shape[1]
        generated_tokens = generate_ids[:, input_token_len:]
        
        response_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        print(f"CLIENT {client_id} OUTPUT:")
        print(f"'{response_text.strip()}'")
        print("-" * 30)

if __name__ == "__main__":
    run_test()