import torch

# Model Settings
MODEL_ID = "meta-llama/Llama-2-7b-hf"
USE_4BIT = True

# FedALT Hyperparameters (From the Paper)
NUM_CLIENTS = 8
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# LoRA Settings
LORA_RANK = 8
LORA_ALPHA = 16  # Scaling factor
TARGET_MODULES = ["q_proj", "v_proj"]

# Dataset Settings
DATASET_NAME = "SirNeural/flan_v2"
MAX_SEQ_LENGTH = 512
SAMPLES_PER_CLIENT = 500  # Adjust based on your GPU speed

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"