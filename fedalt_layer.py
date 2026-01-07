import torch.nn as nn
from fedalt_modules import IndividualLoRA, RestOfWorldLoRA, AdaptiveMixer

class FedALTLayer(nn.Module):
    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base_layer = base_layer # The frozen Llama layer (W0) 
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        
        self.individual_lora = IndividualLoRA(in_dim, out_dim, rank)
        self.row_lora = RestOfWorldLoRA(in_dim, out_dim, rank)
        self.mixer = AdaptiveMixer(in_dim)

    def forward(self, x):
        # Equation: y = W0x + alpha_local(x) * BA_local + alpha_row(x) * BA_row 
        base_out = self.base_layer(x)
        weights = self.mixer(x)
        
        alpha_local = weights[..., 0:1]
        alpha_row = weights[..., 1:2]
        
        local_out = self.individual_lora(x)
        row_out = self.row_lora(x)
        
        return base_out + (alpha_local * local_out) + (alpha_row * row_out)