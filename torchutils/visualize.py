import os as _os

import matplotlib.pyplot as _plt

from ._validate import _validate_param

__all__ = ['plot_gradients']


def plot_gradients(model, file_path, include_bias=False, plot_max=False,
                   plot_type='line', ylim=(-1.0, -1.0)):
    """Plot (average) gradients for each layer in model.

    Useful for debugging vanishing gradient problem. This API should be called
    after loss.backward() and before  optimizer.step().

    Args:
        model (nn.Module): PyTorch model.
        file_path (str): File path (including file name) to save plot.
        include_bias (bool): Include/exclude bias gradients from plot.
            (default: False)
        plot_max (bool): Plot max gradients also. (default: False)
        plot_type (str): Type of plot. Must be one of ('line', 'bar').
            (default: 'line')
        ylim (tuple): Limit the y-axis (gradient values) of the plot.
            Useful for zooming into low gradient regions. Must be tuple
            (low, high). Negative low/high value will plot entire y-limit.
            (default: (-1.0, -1.0))

    Returns:
        None: Returns nothing.

    Example::

        import torch
        import torchvision
        import torchutils as tu

        criterion = torch.nn.CrossEntropyLoss()
        net = torchvision.models.alexnet(num_classes=10)
        out = net(torch.rand(1, 3, 224, 224))
        ground_truth = torch.randint(0, 10, (1, ))
        loss = criterion(out, ground_truth)
        loss.backward()
        tu.plot_gradients(net, './grad_figures/grad_01.png', plot_type='line')

    """

    _validate_param(model, 'model', 'model')
    _validate_param(file_path, 'file_path', 'str')
    _validate_param(include_bias, 'include_bias', 'bool')
    _validate_param(plot_max, 'plot_max', 'bool')
    _validate_param(plot_type, 'plot_type', 'str')
    _validate_param(ylim, 'ylim', 'tuple')
    # validate plot type
    if plot_type not in {'line', 'bar'}:
        raise ValueError(
            'plot_type must be one of (\'line\', \'bar\') but got \'{}\''.
            format(plot_type))
    if len(ylim) != 2:
        raise ValueError(
            'ylim must be tuple of size 2 but got tuple of size {}'.format(
                len(ylim)))
    # get gradient for each layer
    layers, ave_grads, max_grads = [], [], []
    for n, p in model.named_parameters():
        if p.requires_grad and (include_bias or "bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(-10)
                max_grads.append(-10)
            else:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
    fig = _plt.figure()
    count = len(max_grads)
    count_range = range(0, count, 1)
    if (plot_type == 'line'):
        if plot_max:
            _plt.plot(max_grads, alpha=0.4, color="b", label='Max')
        _plt.plot(ave_grads, alpha=0.4, color="r", label='Mean')
    else:
        if plot_max:
            _plt.bar(count_range, max_grads, alpha=0.4, lw=1, color="c",
                     label="Max")
        _plt.bar(count_range, ave_grads, alpha=0.4, lw=1, color="b",
                 label="Mean")
    _plt.hlines(0, 0, count + 1, lw=2, color="k")
    _plt.xticks(count_range, layers, rotation="vertical")
    _plt.xlim(xmin=0, xmax=count)
    # zoom in on the lower gradient regions
    if ylim[0] >= 0 and ylim[1] > 0:
        _plt.ylim(bottom=ylim[0], top=ylim[1])
    _plt.xlabel("Layers")
    _plt.ylabel("Gradient")
    _plt.title("Gradient Flow")
    _plt.grid(True)
    _plt.tight_layout()
    _plt.legend(framealpha=1.0)
    dir_path = _os.path.dirname(file_path)
    _os.makedirs(dir_path, exist_ok=True)
    fig.savefig(file_path)
    _plt.close()
