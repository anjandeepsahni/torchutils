import torch
from _validate import _validate_param

def get_current_lr(optimizer):
    """
    Get learning rate.

    Parameters
    ----------
    optimizer : optim.Optimizer
        PyTorch optimizer.

    Returns
    -------
    float
        Learning rate of the optimizer.
    """

    _validate_param(optimizer, 'optimizer', 'optimizer')
    return optimizer.param_groups[0]['lr']

def set_current_lr(optimizer, lr):
    """
    Set learning rate.

    Parameters
    ----------
    optimizer : optim.Optimizer
        PyTorch optimizer.
    lr : float
        New learning rate value.

    Returns
    -------
    optim.Optimizer
        PyTorch optimizer.
    """

    _validate_param(optimizer, 'optimizer', 'optimizer')
    _validate_param(lr, 'lr', 'float')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
