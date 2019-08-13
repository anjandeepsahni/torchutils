from ._validate import _validate_param

__all__ = ['get_model_param_count']


def get_model_param_count(model, param_type='total'):
    """
    Count number of parameters in the PyTorch model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    param_type : str
        Type of parameters. Available arguments are 'total', 'train' and
        'nontrain'. Defaults to 'total'.

    Returns
    -------
    int
        Number of parameters in the model.
    """

    _validate_param(model, 'model', 'model')
    _validate_param(param_type, 'param_type', 'str')

    param_count = 0
    for p in model.parameters():
        if param_type == 'total':
            val = p.size(0)
            if len(p.size()) > 1:
                val *= p.size(1)
            param_count += val

        elif param_type == 'train':
            if p.requires_grad:
                val = p.size(0)
                if len(p.size()) > 1:
                    val *= p.size(1)
                param_count += val

        else:
            if not p.requires_grad:
                val = p.size(0)
                if len(p.size()) > 1:
                    val *= p.size(1)
                param_count += val

        return param_count
