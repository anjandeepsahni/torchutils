import torch
import torch.optim as optim

def _validate_optimizer(val):
    if val and not isinstance(val, optim.Optimizer):
        raise TypeError('Optimizer must be optim.Optimizer, but got {}.'.format(type(val)))

def get_current_lr(optimizer):
    _validate_optimizer(optimizer)
    return optimizer.param_groups[0]['lr']

def set_current_lr(optimizer, lr):
    _validate_optimizer(optimizer)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
