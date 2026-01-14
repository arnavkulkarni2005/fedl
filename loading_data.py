# loading_data.py
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np

def get_multimodal_data(num_samples=100):
    """
    Fetches image-text pairs. 
    Uses 'HuggingFaceM4/the_cauldron' (subset: coco_caption) or generates dummies.
    """
    data = []
    try:
        # Try loading a small streaming dataset
        ds = load_dataset("HuggingFaceM4/the_cauldron", "coco_caption", split="train", streaming=True)
        iterable = iter(ds)
        
        for _ in range(num_samples):
            item = next(iterable)
            # Structure: {'images': [PIL.Image], 'texts': ["Caption"]}
            if len(item['images']) > 0:
                data.append({
                    "image": item['images'][0], 
                    "text": item['texts'][0]
                })
    except Exception as e:
        print(f"Dataset load failed/skipped ({e}). Using synthetic dummy data for testing.")
        # Create Dummy Data for debugging pipeline
        for i in range(num_samples):
            # Create a random noise image
            img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
            data.append({
                "image": img,
                "text": f"This is a caption for image {i}"
            })
            
    return data

def assign_tasks_to_8_clients():
    """
    Distributes VLM tasks. 
    For VLM, we might split by 'context' (e.g., Client 1 gets food images, Client 2 gets animal images)
    For now, we shuffle and split the generic pool.
    """
    full_data = get_multimodal_data(num_samples=800) # 100 per client
    
    federated_data = {}
    chunk_size = len(full_data) // 8
    
    for i in range(8):
        start = i * chunk_size
        federated_data[i] = full_data[start : start + chunk_size]
        print(f"Client {i} assigned {len(federated_data[i])} multi-modal samples.")
        
    return federated_data