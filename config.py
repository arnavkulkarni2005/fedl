import torch
PHI3_VISION_ID = "microsoft/Phi-3.5-vision-instruct"
MODEL_ID = PHI3_VISION_ID
USE_4BIT = True

NUM_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4 
LEARNING_RATE = 2e-4 

LORA_RANK = 32

TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"] 

DATASET_NAME = "HuggingFaceM4/the_cauldron"
SUBSET_NAME = "okvqa"
MAX_SEQ_LENGTH = 128
MAX_IMAGE_TOKEN = 256 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"