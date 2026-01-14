import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
from config import MODEL_ID

def load_llama_base(model_id=MODEL_ID):
    """
    Loads the LLaVA-1.5 VLM with 4-bit quantization.
    """
    print(f"Loading VLM model: {model_id}...")

    # 1. 4-bit Quantization Config (Critical for VRAM efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load the VLM (LLaVA)
    # We use LlavaForConditionalGeneration, not AutoModelForCausalLM
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 3. Freeze Base Parameters
    # We only want to train the FedALT adapters we add later
    for param in model.parameters():
        param.requires_grad = False

    print("LLaVA model loaded successfully (4-bit) and base parameters frozen.")
    
    # We return None for tokenizer because VLMs use a 'Processor' instead
    return model, None