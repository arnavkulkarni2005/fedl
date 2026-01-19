# FedALT-VLM: Privacy-Preserving AI on Consumer Hardware

## The Elevator Pitch
**FedALT-VLM** is a system that allows multiple AI models to learn together without sharing their private data.

Imagine 8 different hospitals interacting with an AI. One hospital asks about X-rays, another about MRI scans. Instead of sending sensitive patient data to a central server, our system allows each hospital to train a local adapter (a small "brain") on their own data. They then share *only* the mathematical updates—not the images—to build a smarter collective intelligence.

We built this from scratch to run on a standard **11GB consumer GPU** (like a GTX 1080 Ti), proving that advanced Multi-Modal AI doesn't require massive data centers.

---

## Why This Project Stands Out
Most student projects simply fine-tune a model using existing libraries. Here is how **FedALT-VLM** demonstrates advanced engineering skills suitable for industry applications:

### 1. We Built the Engine, We Didn't Just Drive It
* **The Norm:** Most practitioners use the `peft` library to add standard LoRA adapters.
* **Our Work:** We manually implemented the Low-Rank Adaptation (LoRA) mathematics in PyTorch. We created a custom `FedALTLayer` that dynamically mixes "Local" knowledge with "Global" knowledge using a learned gating mechanism.

### 2. "Impossible" Hardware Optimization
* **The Challenge:** Vision-Language Models (like Phi-3.5 Vision) usually require 40GB+ of VRAM to train.
* **Our Solution:** We engineered a pipeline using **4-bit quantization (NF4)** and **Gradient Checkpointing** to fit the entire training process into just **11GB of VRAM**. This demonstrates a deep understanding of low-level memory management and optimization.

### 3. Complex Distributed Logic
* **The Algorithm:** We implemented a "Leave-One-Out" aggregation strategy. When the server updates Client A, it aggregates knowledge from Clients B, C, D... but *excludes* Client A's own previous contributions to prevent feedback loops. This is far more complex than standard Federated Averaging.

---

## Technical Architecture

The architecture splits the learning process into two distinct streams to balance personalization with generalization.

### The Dual-Stream Mechanism
1.  **The "Individual" Stream:**
    * **Goal:** Become an expert at the local task (e.g., answering questions about charts).
    * **Mechanism:** A private adapter that updates *only* on local data.
2.  **The "Rest-of-World" Stream:**
    * **Goal:** Learn general knowledge from everyone else (e.g., recognizing objects).
    * **Mechanism:** A frozen adapter that represents the collective intelligence of all other clients.
3.  **The Mixer:**
    * A lightweight neural network decides, token-by-token, whether to trust the **Individual** expert or the **Rest-of-World** generalist.

### Mathematical Formulation
For any linear layer output $y$ and input $x$, the output is computed as:

$$y = W_{frozen}x + \alpha_{loc}(x) \cdot (B_{loc}A_{loc}x) + \alpha_{row}(x) \cdot (B_{row}A_{row}x)$$

Where:
* **$W_{frozen}$**: The quantized 4-bit weights of the pre-trained base model.
* **Individual LoRA ($A_{loc}, B_{loc}$)**: Captures client-specific task knowledge.
* **Rest-of-World LoRA ($A_{row}, B_{row}$)**: Aggregates weights from all other clients.
* **Adaptive Mixer ($\alpha$)**: A learned gating network that outputs dynamic scalar weights ($\alpha_{loc}, \alpha_{row}$).

---

## Tech Stack
* **Frameworks:** PyTorch, Hugging Face Transformers
* **Model:** Microsoft Phi-3.5 Vision (Quantized)
* **Techniques:** Federated Learning, Low-Rank Adaptation (LoRA), 4-bit Quantization
* **Hardware Target:** Single NVIDIA GTX 1080 Ti (11GB VRAM)

---

## Project Structure
* `fedalt_layer.py`: The custom neural network layer we built from scratch.
* `server_aggregation.py`: The logic for combining weights from different clients using the "Leave-One-Out" strategy.
* `main_federated_train.py`: The main script that simulates 8 clients training in parallel rounds.
* `integrate_fedalt.py`: A script that surgically replaces standard layers in the model with our custom layers.

---

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt

### 2. Run the simulation
```bash
python main_federated_train.py

This will download the model, set up 8 virtual clients, and begin the federated training process.

### 3. Evaluate Success
```bash
python evaluate_results.py

Check how well the model learned specific tasks compared to a baseline.
