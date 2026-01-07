from fedalt_layer import FedALTLayer

def apply_fedalt_to_llama(model, rank=8):
    """
    Replaces Llama's query and value projections with FedALT layers.
    The paper targets multiple layers; q_proj and v_proj are standard.
    """
    # 1. Target layers (q_proj and v_proj are standard for LoRA)
    target_modules = ["q_proj", "v_proj"]
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            # Get the parent module and the attribute name
            parent_path = name.rsplit('.', 1)
            if len(parent_path) == 2:
                parent_name, child_name = parent_path
                parent = model.get_submodule(parent_name)
            else:
                child_name = parent_path[0]
                parent = model
            
            # 2. Replace the original Linear layer with FedALTLayer
            setattr(parent, child_name, FedALTLayer(module, rank=rank))

    # 3. Ensure only relevant parts are trainable [cite: 138]
    for n, p in model.named_parameters():
        if "individual_lora" in n or "mixer" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    print("FedALT architecture successfully integrated into Llama.")
    return model