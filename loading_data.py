# loading_data.py
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np

# Task to Dataset Mapping for 'the_cauldron'
TASK_MAPPING = {
    "CommonSense": "aokvqa",
    "Coreference": "visual7w",
    "NLI": "nlvr2"
}

def get_task_specific_data(task_name, num_samples=200):
    """
    Fetches image-text pairs for a specific task subset.
    """
    subset = TASK_MAPPING.get(task_name, "okvqa")
    data = []
    try:
        # Load specific subset in streaming mode
        ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True)
        iterable = iter(ds)
        
        for _ in range(num_samples):
            item = next(iterable)
            # The Cauldron standard format: {'images': [PIL], 'texts': [str]}
            if len(item['images']) > 0 and len(item['texts']) > 0:
                data.append({
                    "image": item['images'][0], 
                    "text": item['texts'][0],
                    "task": task_name
                })
    except Exception as e:
        print(f"Task {task_name} ({subset}) failed: {e}. Falling back to dummy.")
        for i in range(num_samples):
            img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
            data.append({"image": img, "text": f"Dummy {task_name} answer {i}", "task": task_name})
            
    return data

def assign_tasks_to_3_clients():
    """
    Assigns each client a unique task from the multi-task list.
    """
    tasks = list(TASK_MAPPING.keys())
    federated_data = {}
    
    for i, task_name in enumerate(tasks):
        print(f"Assigning Client {i} to TASK: {task_name}...")
        federated_data[i] = get_task_specific_data(task_name, num_samples=200)
        
    return federated_data