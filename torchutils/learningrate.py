import torch
from ._validate import _validate_param

def get_lr(optimizer):
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
    return get_current_lr(optimizer)

def set_lr(optimizer, lr):
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
    return set_current_lr(optimizer, lr)

########################################################################
#################### BACKWARD COMPATIBILITY SECTION ####################
########################################################################

def get_current_lr(optimizer):
    raise DeprecationWarning(('torchutils.get_current_lr is depracated. '
                                'This will be an error in future releases. '
                                'Please use torchutils.get_lr instead.'))
    _validate_param(optimizer, 'optimizer', 'optimizer')
    return optimizer.param_groups[0]['lr']

def set_current_lr(optimizer, lr):
    raise DeprecationWarning(('torchutils.set_current_lr is depracated. '
                                'This will be an error in future releases. '
                                'Please use torchutils.set_lr instead.'))
    _validate_param(optimizer, 'optimizer', 'optimizer')
    _validate_param(lr, 'lr', 'float')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
