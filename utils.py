import matplotlib.pyplot as plt
import json

def save_training_meta(metrics, filename="metrics.json"):
    with open(filename, "w") as f:
        json.dump(metrics, f)

def plot_losses(client_losses):
    """
    client_losses: Dict where keys are client_ids and values are lists of loss values
    """
    plt.figure(figsize=(10, 6))
    for client_id, losses in client_losses.items():
        plt.plot(losses, label=f"Client {client_id}")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("FedALT Training Progress per Client")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()