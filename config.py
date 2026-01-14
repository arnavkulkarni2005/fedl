# config.py
import torch

# --- PHASE 2: MULTI-MODAL SETTINGS ---
# We use LLaVA-1.5-7b because it uses Llama as the decoder, 
# making it compatible with your existing 4-bit quantization code.
MODEL_ID = "llava-hf/llava-1.5-7b-hf" 
USE_4BIT = True

# FedALT Hyperparameters
NUM_CLIENTS = 8
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 2 # Reduced batch size for VLM (Images take VRAM)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# LoRA & Mixer Settings
LORA_RANK = 8
LORA_ALPHA = 16

# TARGET MODULES: This is the "Dual-Stream" Logic
# 'q_proj', 'v_proj' -> Targets the Language Decoder (Llama)
# 'q_proj', 'v_proj' inside 'vision_tower' -> Targets the Vision Encoder (CLIP)
TARGET_MODULES = ["q_proj", "v_proj"] 

# Dataset Settings
DATASET_NAME = "HuggingFaceM4/the_cauldron" # Multi-modal dataset collection
SUBSET_NAME = "coco_caption" # Simplest task for prototyping
MAX_SEQ_LENGTH = 128
MAX_IMAGE_TOKEN = 576 # Standard for LLaVA
SAMPLES_PER_CLIENT = 200 

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"