import torch
import torch.nn as nn
import torch.nn.functional as F

class IndividualLoRA(nn.Module):
    """Captures locally learned information; updated exclusively during fine-tuning."""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A) @ self.B

class RestOfWorldLoRA(nn.Module):
    """Aggregates information from all other clients; remains fixed during local training."""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, rank), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(rank, out_features), requires_grad=False)

    def forward(self, x):
        return (x @ self.A) @ self.B

class AdaptiveMixer(nn.Module):
    """Learns input-specific weightings (alpha) between Individual and RoW LoRA."""
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Linear(in_features, 2)

    def forward(self, x):
        # Returns alpha_local and alpha_row
        return F.softmax(self.gate(x), dim=-1)