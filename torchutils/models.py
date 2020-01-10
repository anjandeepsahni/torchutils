from collections import OrderedDict as _OrderedDict

import numpy as _np
import torch as _torch

from ._flops import _compute_flops
from ._validate import _validate_param

__all__ = ['get_model_param_count', 'get_model_flops', 'get_model_summary']


class _ModelSummary():

    def __init__(self, model, *x, compact=False, **kwargs):
        # prepare module names
        self.module_names = {}
        self.prep_names_dict(model)

        # create properties
        self.summary = _OrderedDict()
        self.total_model_flops = 0
        self.hooks = []
        self.x = x
        self.compact = compact

        # register hook
        model.apply(self.register_hook)

        # make a forward pass
        try:
            with _torch.no_grad():
                model(*x, **kwargs)
        finally:
            for hook in self.hooks:
                hook.remove()

    def prep_names_dict(self, module, parent_name=""):
        for key, m in module.named_children():
            num_named_children = len(list(m.named_children()))
            if parent_name and num_named_children > 0:
                name = parent_name + "." + key
            elif parent_name:
                cls_name = str(m.__class__).split(".")[-1].split("'")[0]
                name = parent_name + "." + cls_name + "_" + key
            else:
                name = key
            self.module_names[name] = m

            if isinstance(m, _torch.nn.Module):
                self.prep_names_dict(m, parent_name=name)

    def register_hook(self, module):
        # ignore Sequential and ModuleList
        if not module._modules:
            self.hooks.append(module.register_forward_hook(self.hook))

    def hook(self, module, inp, out):
        module_idx = len(self.summary)

        # Lookup name in dict that includes parents
        for name, item in self.module_names.items():
            if item == module:
                m_key = "{}_{}".format(module_idx, name)
                break

        m_info = _OrderedDict()
        m_info["id"] = id(module)

        # store output size
        if isinstance(out, (list, tuple)):
            try:
                m_info["out"] = list(out[0].size())
            except AttributeError:
                # pack_padded_seq and pad_packed_seq store
                # feature into data attribute
                m_info["out"] = list(out[0].data.size())
        else:
            m_info["out"] = list(out.size())

        m_info["ksize"] = "-"
        m_info["params_t"], m_info["params"], m_info["flops"] = 0, 0, 0

        for name, param in module.named_parameters():
            m_info["params"] += param.nelement()
            m_info["params_t"] += param.nelement() * param.requires_grad
            if name == "weight":
                ksize = list(param.size())
                # to make [in_shape, out_shape, ksize, ksize]
                if len(ksize) > 1:
                    ksize[0], ksize[1] = ksize[1], ksize[0]
                m_info["ksize"] = ksize

        # if the current module is already-used, this is recursive.
        # check if this module has params
        if list(module.named_parameters()):
            for v in self.summary.values():
                if m_info["id"] == v["id"]:
                    m_info["params"] = 0
                    m_info["params_t"] = 0
                    break

        # compute module flops
        m_info["flops"] = _compute_flops(module, inp, out)

        self.summary[m_key] = m_info
        self.total_model_flops += int(m_info["flops"])

    def show(self):
        # calculate max field lengths
        max_layer, max_out = len('Layer'), len('Output')
        max_params, max_ksize = len('Params'), len('Kernel')
        max_flops = len('FLOPs')
        total_params, total_output, trainable_params, total_flops = 0, 0, 0, 0
        for layer in self.summary:
            max_layer = max(max_layer, len(layer))
            max_out = max(max_out, len(str(self.summary[layer]["out"])))
            max_params = max(
                max_params, len("{0:,}".format(self.summary[layer]["params"])))
            max_ksize = max(max_ksize, len(str(self.summary[layer]["ksize"])))
            max_flops = max(max_flops,
                            len("{0:,}".format(self.summary[layer]["flops"])))
            total_params += self.summary[layer]["params"]
            total_output += _np.prod(self.summary[layer]["out"])
            total_flops += int(self.summary[layer]["flops"])
            trainable_params += self.summary[layer]["params_t"]

        # calculate total values.
        total_input_size = _np.sum([
            _np.prod(list(_x.size())) * _x.element_size()
            for _x in self.x if isinstance(_x, _torch.Tensor)
        ]) / (1024**2.)
        # assumes default float, 4 bytes as size.
        ib = 4
        # x2 output size for gradients
        total_output_size = (2. * total_output * ib) / (1024**2.)
        total_params_size = (total_params * ib) / (1024**2.)
        total_size = total_params_size + total_output_size + total_input_size
        if total_flops > (1e9):
            total_flops_size = total_flops / (1e9)
            tfs_str = "GFLOPs"
        elif total_flops > (1e6):
            total_flops_size = total_flops / (1e6)
            tfs_str = "MFLOPs"
        else:
            total_flops_size = total_flops / (1e3)
            tfs_str = "KFLOPs"

        # calculate total line length
        total_line_len = max_layer + max_out
        total_line_len += 3  # adjust for spaces
        if not self.compact:
            total_line_len += max_params + max_ksize + max_flops
            total_line_len += 3 * 3  # adjust for spaces

        # prepare summary lines
        _lines = {}
        _lines['param'] = "Total params: {:,}".format(total_params)
        _lines['param_t'] = "Trainable params: {:,}".format(trainable_params)
        _lines['param_nt'] = "Non-trainable params: {:,}".format(
            total_params - trainable_params)
        _lines['flops'] = "Total FLOPs: {:,} / {:.2f} {}".format(
            total_flops, total_flops_size, tfs_str)
        _lines['inp_size'] = "Input size (MB): {:.2f}".format(total_input_size)
        _lines['pass_size'] = "Forward/backward pass size (MB): {:.2f}".format(
            total_output_size)
        _lines['param_size'] = "Params size (MB): {:.2f}".format(
            total_params_size)
        _lines['est_total'] = "Estimated Total Size (MB): {:.2f}".format(
            total_size)

        for k, v in _lines.items():
            total_line_len = max(total_line_len, len(v))

        _lines['-'] = "-" * total_line_len
        _lines['='] = "=" * total_line_len

        # print summary
        print(_lines['='])
        if self.compact:
            header = "{:<{ml}}   {:>{mo}}".format(
                "Layer", "Output", ml=max_layer,
                mo=max(max_out, total_line_len - max_layer - 3))
        else:
            header = "{:<{ml}}   {:^{mk}}   {:^{mo}}" \
                     "   {:^{mp}}   {:>{mf}}".format(
                      "Layer", "Kernel", "Output", "Params", "FLOPs",
                      ml=max_layer, mk=max_ksize, mo=max_out, mp=max_params,
                      mf=max_flops)
        print(header)
        print(_lines['='])
        for layer in self.summary:
            if self.compact:
                line = "{:<{ml}}   {:>{mo}}".format(
                    layer, str(self.summary[layer]["out"]), ml=max_layer,
                    mo=max(max_out, total_line_len - max_layer - 3))
            else:
                line = "{:<{ml}}   {:>{mk}}   {:>{mo}}" \
                       "   {:>{mp}}   {:>{mf}}".format(
                        layer, str(self.summary[layer]["ksize"]),
                        str(self.summary[layer]["out"]),
                        "{0:,}".format(self.summary[layer]["params"]),
                        "{0:,}".format(int(self.summary[layer]["flops"])),
                        ml=max_layer, mk=max_ksize, mo=max_out,
                        mp=max_params, mf=max_flops)
            print(line)
        print(_lines['='])
        print("{}\n{}\n{}\n{}".format(_lines['param'], _lines['param_t'],
                                      _lines['param_nt'], _lines['flops']))
        print(_lines['-'])
        print("{}\n{}\n{}\n{}".format(_lines['inp_size'], _lines['pass_size'],
                                      _lines['param_size'],
                                      _lines['est_total']))
        print(_lines['='])


def get_model_summary(model, *input, compact=False, **kwargs):
    """Print model summary.

    Args:
        model (nn.Module): PyTorch model.
        input (user dependent): Input(s) for model. Shape: [N, \\*].
            Input dtype and device must match to the model.
            Can be comma separated inputs for multi-input models.
        compact (bool): To print compact summary, only layer and output shape.
            (default: False)
        **kwargs: Other keyword arguments used in model.forward function.

    Returns:
        None: Returns nothing.

    Example::

        import torch
        import torchvision
        import torchutils as tu

        model = torchvision.models.alexnet()
        tu.get_model_summary(model, torch.rand((1, 3, 224, 224)), compact=True)

    Out::

        ===========================================
        Layer                                Output
        ===========================================
        0_features.Conv2d_0         [1, 64, 55, 55]
        1_features.ReLU_1           [1, 64, 55, 55]
        2_features.MaxPool2d_2      [1, 64, 27, 27]
        3_features.Conv2d_3        [1, 192, 27, 27]
        4_features.ReLU_4          [1, 192, 27, 27]
        5_features.MaxPool2d_5     [1, 192, 13, 13]
        6_features.Conv2d_6        [1, 384, 13, 13]
        7_features.ReLU_7          [1, 384, 13, 13]
        8_features.Conv2d_8        [1, 256, 13, 13]
        9_features.ReLU_9          [1, 256, 13, 13]
        10_features.Conv2d_10      [1, 256, 13, 13]
        11_features.ReLU_11        [1, 256, 13, 13]
        12_features.MaxPool2d_12     [1, 256, 6, 6]
        13_classifier.Dropout_0           [1, 9216]
        14_classifier.Linear_1            [1, 4096]
        15_classifier.ReLU_2              [1, 4096]
        16_classifier.Dropout_3           [1, 4096]
        17_classifier.Linear_4            [1, 4096]
        18_classifier.ReLU_5              [1, 4096]
        19_classifier.Linear_6            [1, 1000]
        ===========================================
        Total params: 61,100,840
        Trainable params: 61,100,840
        Non-trainable params: 0
        Total FLOPs: 773,286,232 / 773.29 MFLOPs
        -------------------------------------------
        Input size (MB): 0.57
        Forward/backward pass size (MB): 8.31
        Params size (MB): 233.08
        Estimated Total Size (MB): 241.96
        ===========================================

    """

    _validate_param(model, 'model', 'model')
    _validate_param(input, 'input', 'tuple')
    _validate_param(compact, 'compact', 'bool')
    summary = _ModelSummary(model, *input, compact=compact, **kwargs)
    summary.show()


def get_model_flops(model, *input, unit='FLOP', **kwargs):
    """Count total FLOPs for the PyTorch model.

    Args:
        model (nn.Module): PyTorch model.
        input (user dependent): Input(s) for model. Shape: [N, \\*].
            Input dtype and device must match to the model.
            Can be comma separated inputs for multi-input models.
        unit (str): FLOPs unit. Can be 'FLOP', 'MFLOP' or 'GFLOP'.
            (default: 'FLOP')
        **kwargs: Other keyword arguments used in model.forward function.

    Returns:
        float: Number of FLOPs.

    Example::

        import torch
        import torchvision
        import torchutils as tu

        model = torchvision.models.alexnet()
        total_flops = tu.get_model_flops(model, torch.rand((1, 3, 224, 224)))
        print('Total model FLOPs: {:,}'.format(total_flops))

    Out::

        Total model FLOPs: 773,304,664

    """

    _validate_param(model, 'model', 'model')
    _validate_param(input, 'input', 'tuple')
    _validate_param(unit, 'unit', 'str')
    assert (unit in {'GFLOP', 'MFLOP', 'FLOP'})
    summary = _ModelSummary(model, *input, compact=False, **kwargs)
    flops = summary.total_model_flops
    if unit == 'GFLOP':
        flops /= 1e9
    elif unit == 'MFLOP':
        flops /= 1e6
    return round(flops, 2)


def get_model_param_count(model, trainable=None):
    """Count total parameters in the PyTorch model.

    Args:
        model (nn.Module): PyTorch model.
        trainable (None or bool): Pass ``None``: total, ``True``: trainable,
            or ``False``: non-trainable parameters.

    Returns:
        int: Number of parameters in the model.

    Example::

        import torchvision
        import torchutils as tu

        model = torchvision.models.alexnet()
        total_params = tu.get_model_param_count(model)
        print('Total model params: {:,}'.format(total_params))

    Out::

        Total model params: 61,100,840

    """

    _validate_param(model, 'model', 'model')
    _validate_param(trainable, 'trainable', ['bool', 'none'])

    total_params = 0
    train_params = 0

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            train_params += p.numel()

    if trainable is None:
        return total_params
    elif trainable:
        return train_params
    else:
        return total_params - train_params
