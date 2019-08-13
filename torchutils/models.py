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

    total_params = 0
    train_params = 0

    for p in model.parameters():
        val = p.size(0)
        if len(p.size()) > 1:
            val *= p.size(1)
        train_params += val
        if p.requires_grad:
            train_params += val

    if param_type == 'total':
        return total_params
    elif param_type == 'train':
        return train_params
    else:
        return total_params - train_params
