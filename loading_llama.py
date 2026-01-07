import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODEL_ID

def load_llama_base(model_id=MODEL_ID):
    """
    Loads Llama 2 7B with 4-bit quantization and freezes all parameters.
    """
    print(f"Loading model: {model_id}...")

    # 1. Configuration for 4-bit quantization (Saves massive VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" # Recommended for Llama training

    # 3. Load the Base Model (W0)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. Freeze all parameters of the base model
    # FedALT only trains the adapters and the mixer, not the core model
    for param in model.parameters():
        param.requires_grad = False

    print("Llama 2 model loaded successfully (4-bit) and base parameters frozen.")
    
    return model, tokenizer