# FedALT-VLM: Federated Multi-Task Fine-Tuning for Vision-Language Models

## Overview
**FedALT-VLM** is a specialized Federated Learning framework designed to fine-tune Vision-Language Models (VLMs) on non-IID (non-independent and identically distributed) multi-modal data.

Built from scratch using **PyTorch** and **Hugging Face Transformers**, this project addresses the challenge of training a single foundation model (**Phi-3.5 Vision**) on diverse downstream tasks—such as Visual Question Answering (VQA), Image Captioning, and Sentiment Analysis—without centralizing raw user data.

The core innovation is the implementation of the **FedALT (Federated Adaptive LoRA Tuning)** architecture, which dynamically balances personalized local learning with global knowledge aggregation.

---

## Architecture: The FedALT Layer
Unlike standard Low-Rank Adaptation (LoRA), which learns a single delta weight $\Delta W$, this project implements a **Dual-Stream Adaptive Mechanism**. The target linear layers of the Phi-3.5 Vision model are replaced with custom `FedALTLayer` modules.

**Mathematical Formulation**
For any linear layer output $y$ and input $x$, the output is computed as:
$$y = W_{frozen}x + \alpha_{loc}(x) \cdot (B_{loc}A_{loc}x) + \alpha_{row}(x) \cdot (B_{row}A_{row}x)$$

* **$W_{frozen}$**: The quantized 4-bit weights of the pre-trained base model.
* **Individual LoRA ($A_{loc}, B_{loc}$)**: Captures client-specific task knowledge (updated locally).
* **Rest-of-World (RoW) LoRA ($A_{row}, B_{row}$)**: Aggregates weights from all *other* clients (frozen during local training).
* **Adaptive Mixer ($\alpha$)**: A learned gating network that outputs dynamic scalar weights ($\alpha_{loc}, \alpha_{row}$) to contextually blend local and global streams per token.

---

## Key Features & Technical Highlights

### Custom Parameter-Efficient Fine-Tuning (PEFT)
* **Manual Implementation**: Implemented custom `torch.nn.Module` classes (`IndividualLoRA`, `RestOfWorldLoRA`) instead of using off-the-shelf libraries to support the specific dual-stream requirement.
* **Targeted Adaptation**: Specifically targets `qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj` layers within the Phi-3.5 attention blocks.

### Federated Learning Simulation
* **Client Heterogeneity**: Simulates **8 distinct clients**, each assigned a unique task from the **HuggingFaceM4/the_cauldron** dataset (e.g., OK-VQA, TextCaps, ScienceQA).
* **"Rest-of-World" Aggregation**: Implemented a custom server-side aggregation logic that calculates a unique global context for each client by excluding their own weights from the sum ($GlobalSum - ClientWeight$).

### Hardware-Aware Optimization
* **Consumer GPU Compatibility**: Engineered to run on 11GB VRAM (e.g., GTX 1080 Ti) by leveraging **4-bit NF4 quantization** (`bitsandbytes`) and **Gradient Checkpointing**.
* **Orthogonal Regularization**: A custom loss term enforces orthogonality between Local and RoW matrices to prevent feature redundancy.

---

## Repository Structure

* `fedalt_layer.py`: Defines the composite layer combining Base, Individual LoRA, RoW LoRA, and the Mixer.
* `fedalt_modules.py`: Low-level implementation of the rank-decomposition matrices and gating network.
* `main_federated_train.py`: The central engine that orchestrates the training loop, client updates, and server communication.
* `server_aggregation.py`: Contains the logic for the "Leave-One-Out" weight aggregation strategy.
* `integrate_fedalt.py`: A utility to dynamically hot-swap standard Linear layers in Phi-3 with `FedALTLayer`.
* `loading_data.py`: Handles streaming multi-task data from The Cauldron dataset.
* `config.py`: Central configuration for hyperparameters (Rank=16, Learning Rate=2e-4).

---

## Installation

**Prerequisites**
* Python 3.10+
* NVIDIA GPU with CUDA support

**Setup**
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies include: `torch`, `transformers`, `bitsandbytes`, `accelerate`, `datasets`)*

---

## Usage

### Start Federated Training
To initialize the model and begin the federated training simulation (defaults to 5 rounds):
```bash
python main_federated_train.py
