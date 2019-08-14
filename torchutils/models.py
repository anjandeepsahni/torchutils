import numpy as _np
import torch as _torch
from ._flops import _compute_flops
from ._validate import _validate_param
from collections import OrderedDict as _OrderedDict

__all__ = ['get_model_param_count', 'get_model_flops', 'get_model_summary']


class _ModelSummary():
    def __init__(self, model, input_size, batch_size=-1, device='cpu',
                 input_type='float32'):
        assert (device.lower() in {'cuda', 'cpu'})
        assert (not ((device.lower() == "cuda") ^ _torch.cuda.is_available()))
        self.model = model
        if -1 < int(batch_size) <= 0:
            batch_size = -1
        else:
            batch_size = max(int(batch_size), -1)
        self.batch_size = batch_size

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]
        self.input_size = input_size

        valid_input_types = {'float32': 4., 'float16': 2., 'float8': 1.}
        self.input_bytes = valid_input_types[input_type]

        # dummy batch_size of 2 for batchnorm
        inp = [_torch.rand(2, *in_size).to(device) for in_size in input_size]
        self.model = self.model.to(device)

        # create properties
        self.summary = _OrderedDict()
        self.total_model_flops = 0
        self.hooks = []

        # register hook
        self.model.apply(self.register_hook)

        # make a forward pass
        with _torch.no_grad():
            self.model(*inp)

        # remove hooks
        for h in self.hooks:
            h.remove()

    def register_hook(self, module):
        if len(list(module.children())) == 0:
            self.hooks.append(module.register_forward_hook(self.hook))

    def hook(self, module, inp, out):
        # print('module {}, inp {}, out {}'.format(module, inp ,out))
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(self.summary)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        self.summary[m_key] = _OrderedDict()
        if isinstance(out, (list, tuple)):
            self.summary[m_key]["out_shape"] = [[self.batch_size] +
                                                list(o.size())[1:]
                                                for o in out]
        else:
            self.summary[m_key]["out_shape"] = [self.batch_size] + list(
                out.size())[1:]

        # compute module parameters
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += \
                _torch.prod(_torch.Tensor(list(module.weight.size()))).item()
            self.summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += \
                _torch.prod(_torch.Tensor(list(module.bias.size()))).item()
        self.summary[m_key]["params"] = int(params)

        # compute module flops
        flops = _compute_flops(module, inp, out) // 2  # used dummy batch of 2
        flops = int(flops) * max(self.batch_size, 1)
        self.summary[m_key]["flops"] = flops
        self.total_model_flops += flops

    def show(self):
        print("----------------------------------------"
              "---------------------------------------")
        line = "{:>20}  {:>25} {:>15} {:>15}".format("Layer", "Output",
                                                     "Params", "FLOPs")
        print(line)
        print("========================================"
              "=======================================")
        total_params, total_output, trainable_params, total_flops = 0, 0, 0, 0
        for layer in self.summary:
            line = "{:>20}  {:>25} {:>15} {:>15}".format(
                layer, str(self.summary[layer]["out_shape"]),
                "{0:,}".format(self.summary[layer]["params"]),
                "{0:,}".format(self.summary[layer]["flops"]))
            total_params += self.summary[layer]["params"]
            total_output += _np.prod(self.summary[layer]["out_shape"])
            total_flops += self.summary[layer]["flops"]
            if "trainable" in self.summary[layer]:
                if self.summary[layer]["trainable"]:
                    trainable_params += self.summary[layer]["params"]
            print(line)

        # calculate total values
        ib = self.input_bytes
        total_input_size = abs(
            sum([_np.prod(input_item) for input_item in self.input_size]) *
            self.batch_size * ib / (1024**2.))
        # x2 output size for gradients
        total_output_size = abs(2. * total_output * ib / (1024**2.))
        total_params_size = abs(total_params * ib / (1024**2.))
        total_size = total_params_size + total_output_size + total_input_size
        total_flops_size = abs(total_flops / (1e9))

        print("========================================"
              "=======================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params -
                                                   trainable_params))
        print("Total FLOPs: {0:,} / {1:.2f} GFLOPs".format(
            total_flops, total_flops_size))
        print("----------------------------------------"
              "---------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------"
              "---------------------------------------")


def get_model_summary(model, input_size, batch_size=-1, device='cpu'):
    """Print model summary.

    Args:
        model (nn.Module): PyTorch model.
        input_size (tuple): Input size to the model.
            Can be a list of multiple inputs.
        batch_size (int): Batch size. (default: -1)
        device (str): Device to map the checkpoint, "cpu" or "cuda".
            (default: 'cpu')

    Returns:
        None: Returns nothing.

    """

    _validate_param(model, 'model', 'model')
    _validate_param(input_size, 'input_size', ['tuple', 'list'])
    _validate_param(batch_size, 'batch_size', 'int')
    _validate_param(device, 'device', 'str')
    assert (device.lower() in {'cpu', 'cuda'})
    summary = _ModelSummary(model, input_size, batch_size, device.lower())
    summary.show()


def get_model_flops(model, input_size, device='cpu', unit='FLOP'):
    """Count total FLOPs for the PyTorch model.

    Args:
        model (nn.Module): PyTorch model.
        input_size (tuple): Input size to the model.
            Can be a list of multiple inputs.
        device (str): Device to map the checkpoint, "cpu" or "cuda".
            (default: 'cpu')
        unit (str): FLOPs unit. Can be 'FLOP', 'MFLOP' or 'GFLOP'.
            (default: 'FLOP')

    Returns:
        float: Number of FLOPs.

    """

    _validate_param(model, 'model', 'model')
    _validate_param(input_size, 'input_size', ['tuple', 'list'])
    _validate_param(device, 'device', 'str')
    _validate_param(unit, 'unit', 'str')
    assert (device.lower() in {'cpu', 'cuda'})
    assert (unit in {'GFLOP', 'MFLOP', 'FLOP'})
    summary = _ModelSummary(model, input_size, device=device.lower())
    flops = summary.total_model_flops
    if unit == 'GFLOP':
        flops /= 1e9
    elif unit == 'MFLOP':
        flops /= 1e6
    return round(flops, 2)


def get_model_param_count(model):
    """Count total parameters in the PyTorch model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Number of parameters in the model.

    """

    _validate_param(model, 'model', 'model')
    param_count = 0
    for p in model.parameters():
        val = p.size(0)
        if len(p.size()) > 1:
            val *= p.size(1)
        param_count += val
    return param_count
