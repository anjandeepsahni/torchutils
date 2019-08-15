import numpy as _np
import torch as _torch

from ._validate import _validate_param

__all__ = ['set_random_seed']


def set_random_seed(seed):
    """Set random seed for numpy, torch and torch.cuda.

    Args:
        seed (int): Seed value.

    Returns:
        None: Returns nothing.

    Example::

        import torchutils as tu

        tu.set_random_seed(2222)

    """

    _validate_param(seed, 'seed', 'int')
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed(seed)
