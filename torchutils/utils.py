import torch
import numpy as np
from ._validate import _validate_param

def set_random_seed(seed):
    """
    Set random seed for numpy, torch and torch.cuda.

    Parameters
    ----------
    seed : int
        Seed value.

    Returns
    -------
    None
        Nothing.
    """

    _validate_param(seed, 'seed', 'int')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
