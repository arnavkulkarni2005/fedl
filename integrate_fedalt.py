# integrate_fedalt.py
from fedalt_layer import FedALTLayer
from config import TARGET_MODULES
import torch

def apply_fedalt_to_vlm(model, rank=8):
    modules_to_replace = []
    for name, module in model.named_modules():
        if any(target in name for target in TARGET_MODULES):
            if "lm_head" in name: continue
            modules_to_replace.append((name, module))

    device = next(model.parameters()).device
    dtype = torch.float16 

    for name, module in modules_to_replace:
        parent_path = name.rsplit('.', 1)
        if len(parent_path) == 2:
            parent_name, child_name = parent_path
            parent = model.get_submodule(parent_name)
            
            # Move to GPU/Float16 during creation
            new_layer = FedALTLayer(module, rank=rank).to(device=device, dtype=dtype)
            setattr(parent, child_name, new_layer)
            
    for n, p in model.named_parameters():
        p.requires_grad = ("individual_lora" in n or "mixer" in n)
            
    return model