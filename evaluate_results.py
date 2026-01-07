import torch
from tqdm import tqdm
from loading_llama import load_llama_base
from integrate_fedalt import apply_fedalt_to_llama
from loading_data import get_task_specific_data
from config import LORA_RANK

def calculate_accuracy(model, tokenizer, test_data):
    """
    Computes the accuracy for a specific task.
    For LLMs, we check if the model's highest probability prediction 
    matches the ground truth label.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating"):
            # Prepare inputs and labels
            prompt = item['instruction'] + "\n" + item['input']
            label = item['output']
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            
            # Get model prediction
            outputs = model.generate(**inputs, max_new_tokens=10)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Simple string matching for evaluation (Standard for Flan-v2 tasks)
            if label.strip().lower() in prediction.strip().lower():
                correct += 1
            total += 1

    return (correct / total) * 100

def generate_fedalt_table(client_states):
    """
    Runs evaluation for all 8 tasks and prints a table matching the paper's format.
    """
    tasks = [
        "CommonSense", "Coreference", "NLI", "Paraphrase", 
        "ReadingComp", "Sentiment", "StructToText", "TextClass"
    ]
    
    # Initialize Model
    model, tokenizer = load_llama_base()
    model = apply_fedalt_to_llama(model, rank=LORA_RANK)

    results = {}

    for i, task_name in enumerate(tasks):
        print(f"\n--- Evaluating Task: {task_name} ---")
        
        # 1. Load the trained brain for this specific client
        if client_states[i] is not None:
            model.load_state_dict(client_states[i], strict=False)
        
        # 2. Load separate TEST data (distinct from training data)
        test_data = get_task_specific_data(task_name, num_samples=200) # num_samples for speed
        
        # 3. Calculate Score
        score = calculate_accuracy(model, tokenizer, test_data)
        results[task_name] = score

    # --- PRINT THE MATHEMATICAL TABLE ---
    print("\n" + "="*50)
    print(f"{'Task Name':<25} | {'FedALT Score':<15}")
    print("-" * 50)
    
    total_score = 0
    for task, score in results.items():
        print(f"{task:<25} | {score:>14.2f}")
        total_score += score
    
    print("-" * 50)
    print(f"{'AVERAGE':<25} | {total_score/len(tasks):>14.2f}")
    print("="*50)


def run_full_evaluation():
    # 1. Load the states we saved in the main training file
    print("Loading personalized weights...")
    client_states = torch.load("fedalt_final_client_states.pt")

    # 2. Call the table generation function (the one I gave you earlier)
    # This will loop through each client, load their weights, and calculate accuracy
    generate_fedalt_table(client_states)

if __name__ == "__main__":
    run_full_evaluation()