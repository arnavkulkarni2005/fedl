from datasets import load_dataset

def get_task_specific_data(task_name, num_samples=500):
    """
    Returns a dataset filtered for a specific task.
    """
    ds = load_dataset("SirNeural/flan_v2", split="train", streaming=True)
    
    # Use the .filter() method to only keep rows for this task
    filtered_ds = ds.filter(lambda x: x['task'] == task_name)
    
    # Shuffle and take a specific amount
    return list(filtered_ds.shuffle(seed=42).take(num_samples))

def assign_tasks_to_8_clients():
    # Example task names usually found in Flan-v2
    # You should update these based on the output of the "Available tasks" script above
    target_tasks = [
        "cot",           # Chain of Thought (Reasoning)
        "natural_lang",   # Natural Language Inference
        "sentiment",     # Sentiment Analysis
        "summarization", # Summarization
        "translation",   # Translation
        "question_gen",  # Question Generation
        "dialogue",      # Dialogue/Chat
        "common_sense"   # Commonsense Reasoning
    ]
    
    federated_data = {}
    for i, task in enumerate(target_tasks):
        print(f"Loading data for Client {i} (Task: {task})...")
        federated_data[i] = get_task_specific_data(task)
        
    return federated_data