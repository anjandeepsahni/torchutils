import torch.nn as _nn
import torch.optim as _optim


def _validate_param(val, name, param_type):
    supported_types = {
        'int': int,
        'str': str,
        'float': float,
        'model': _nn.Module,
        'optimizer': _optim.Optimizer,
        'scheduler': [_optim.lr_scheduler._LRScheduler,
                      _optim.lr_scheduler.ReduceLROnPlateau]
        }
    if param_type not in supported_types:
        raise TypeError(('INTERNAL ERROR: param_type must be one of {}, but '
                         'got {}. Please raise an issue on github.').format(
                         supported_types, type(val)))
    cur_type = supported_types[param_type]
    if not isinstance(cur_type, list):
        cur_type = [cur_type]
    for ct in cur_type:
        if isinstance(val, ct):
            return
    type_str = str(cur_type[0])
    raise TypeError('{} must be of type {}, but got {}.'.format(
                    name, type_str, type(val)))
