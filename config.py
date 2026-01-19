# config.py
import torch

# --- PHASE 2: MULTI-MODAL SETTINGS ---
# Switching to Phi-3.5 Vision for 11GB VRAM compatibility (GTX 1080 Ti)
PHI3_VISION_ID = "microsoft/Phi-3.5-vision-instruct"
MODEL_ID = PHI3_VISION_ID  # Retained for compatibility with existing import statements
USE_4BIT = True

# FedALT Hyperparameters
NUM_CLIENTS = 8
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 1 # Images take significant VRAM
GRAD_ACCUM_STEPS = 4 # Simulate batch size of 4
LEARNING_RATE = 2e-4 # Slightly higher for LoRA

# LoRA & Mixer Settings
LORA_RANK = 16

# TARGET MODULES: Updated for Phi-3 Architecture
# Unlike LLaVA, Phi-3 typically packs attention into 'qkv_proj'
TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"] 

# Dataset Settings
# config.py
DATASET_NAME = "HuggingFaceM4/the_cauldron"
SUBSET_NAME = "okvqa"  # Changed from "coco_caption" which was not found
MAX_SEQ_LENGTH = 128
# Phi-3.5 supports high resolution; 1024 is a safe limit for your 11GB VRAM
MAX_IMAGE_TOKEN = 256 

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"