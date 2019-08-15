import warnings as _warnings

from ._validate import _validate_param

__all__ = ['get_lr', 'set_lr', 'get_current_lr', 'set_current_lr']


def get_lr(optimizer):
    """Get learning rate.

    Args:
        optimizer (optim.Optimizer): PyTorch optimizer.

    Returns:
        float: Learning rate of the optimizer.

    Example::

        import torchvision
        import torchutils as tu
        import torch.optim as optim

        model = torchvision.models.alexnet()
        optimizer = optim.Adam(model.parameters())
        current_lr = tu.get_lr(optimizer)
        print('Current learning rate:', current_lr)

    Out::

        Current learning rate: 0.001

    """

    _validate_param(optimizer, 'optimizer', 'optimizer')
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer, lr):
    """Set learning rate.

    Args:
        optimizer (optim.Optimizer): PyTorch optimizer.
        lr (float): New learning rate value.

    Returns:
        optim.Optimizer: PyTorch optimizer.

    Example::

        import torchvision
        import torchutils as tu
        import torch.optim as optim

        model = torchvision.models.alexnet()
        optimizer = optim.Adam(model.parameters())
        current_lr = tu.get_lr(optimizer)
        print('Current learning rate:', current_lr)

        optimizer = tu.set_lr(optimizer, current_lr*0.1)
        revised_lr = tu.get_lr(optimizer)
        print('Revised learning rate:', revised_lr)

    Out::

        Current learning rate: 0.001
        Revised learning rate: 0.0001

    """

    _validate_param(optimizer, 'optimizer', 'optimizer')
    _validate_param(lr, 'lr', 'float')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


########################################################################
#                    BACKWARD COMPATIBILITY SECTION                    #
########################################################################


def get_current_lr(optimizer):
    _warnings.simplefilter('always', DeprecationWarning)
    _warnings.warn(('torchutils.get_current_lr is deprecated. '
                    'This will be an error in future releases. '
                    'Please use torchutils.get_lr instead.'),
                   DeprecationWarning)
    _validate_param(optimizer, 'optimizer', 'optimizer')
    return optimizer.param_groups[0]['lr']


def set_current_lr(optimizer, lr):
    _warnings.simplefilter('always', DeprecationWarning)
    _warnings.warn(('torchutils.set_current_lr is deprecated. '
                    'This will be an error in future releases. '
                    'Please use torchutils.set_lr instead.'),
                   DeprecationWarning)
    _validate_param(optimizer, 'optimizer', 'optimizer')
    _validate_param(lr, 'lr', 'float')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
