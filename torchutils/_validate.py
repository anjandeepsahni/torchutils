import torch as _torch
import torch.nn as _nn
import torch.optim as _optim


def _validate_param(val, name, param_type):
    supported_types = {
        'int': int,
        'str': str,
        'list': list,
        'bool': bool,
        'float': float,
        'tuple': tuple,
        'none': type(None),
        'model': _nn.Module,
        'tensor': _torch.Tensor,
        'optimizer': _optim.Optimizer,
        'scheduler': [
            _optim.lr_scheduler._LRScheduler,
            _optim.lr_scheduler.ReduceLROnPlateau
        ],
        'dataloader': _torch.utils.data.DataLoader
    }
    if not isinstance(param_type, list):
        param_type = [param_type]
    for pt in param_type:
        if pt not in supported_types:
            raise TypeError(('INTERNAL ERROR: param_type must be one of {}, '
                             'but got {}. Please raise an issue on '
                             'github.').format(supported_types, type(val)))
        cur_type = supported_types[pt]
        if not isinstance(cur_type, list):
            cur_type = [cur_type]
        for ct in cur_type:
            if isinstance(val, ct):
                return
    type_str = str(cur_type[0])
    raise TypeError('{} must be of type {}, but got {}.'.format(
        name, type_str, type(val)))
