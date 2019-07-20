import torch
import torch.nn as nn

def get_model_param_count(model):
    if model and not isinstance(model, nn.Module):
        raise TypeError('Model must be nn.Module, but got {}.'.format(type(model)))
    param_count = 0
    for p in model.parameters():
        val = p.size(0)
        if len(p.size()) > 1:
            val *= p.size(1)
        param_count += val
    return param_count
