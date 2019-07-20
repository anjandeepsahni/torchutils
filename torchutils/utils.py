import torch
import numpy as np

def set_random_seed(seed):
    if not isinstance(seed, int):
        raise TypeError('Seed value must be int, but got {}.'.format(type(seed)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
