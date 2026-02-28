# loading_llama.py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_ID

def load_phi3_vision_base(model_id=MODEL_ID):
    print(f"Loading Phi-3 Vision: {model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        _attn_implementation='eager' # Compatibility for GTX 1080 Ti
    )

    # Required fixes for training Phi-3 Vision
    model.config.use_cache = False  # Fixes 'DynamicCache' AttributeError
    model.gradient_checkpointing_enable() # Saves VRAM by not storing activations
    model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = False

    return model, None