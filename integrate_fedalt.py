# integrate_fedalt.py
from fedalt_layer import FedALTLayer
from config import TARGET_MODULES

def apply_fedalt_to_vlm(model, rank=8):
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if any(target in name for target in TARGET_MODULES):
            if "lm_head" in name: 
                continue
            stream_type = "VISION" if "vision_tower" in name else "LANGUAGE"
            modules_to_replace.append((name, module, stream_type))


    for name, module, stream_type in modules_to_replace:
        parent_path = name.rsplit('.', 1)
        if len(parent_path) == 2:
            parent_name, child_name = parent_path
            parent = model.get_submodule(parent_name)
            
            # Replace with FedALT Layer
            # This automatically creates a unique Mixer for this layer
            setattr(parent, child_name, FedALTLayer(module, rank=rank))
            
    print(f"FedALT successfully integrated! Total layers adapted: {len(modules_to_replace)}")

    # 3. Freeze everything EXCEPT the FedALT adapters (Individual & Mixer)
    for n, p in model.named_parameters():
        if "individual_lora" in n or "mixer" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    return model