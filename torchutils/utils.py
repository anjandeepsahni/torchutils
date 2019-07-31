import torch
import numpy as np

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

    if not isinstance(seed, int):
        raise TypeError('Seed value must be int, but got {}.'.format(type(seed)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
