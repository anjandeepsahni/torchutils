import torch as _torch
import torch.nn as _nn


def _zero_flops(module, inp, out):
    return 0


def _convNd_flops(module, inp, out):
    kernel_ops = module.weight.size()[2:].numel()  # k_h x k_w
    bias_ops = 1 if module.bias is not None else 0
    # (batch x out_c x out_h x out_w) x  (in_c x k_h x k_w + bias)
    total_ops = out.nelement() * \
        (module.in_channels // module.groups * kernel_ops + bias_ops)
    return total_ops


def _bn_flops(module, inp, out):
    nelements = inp.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements
    return total_ops


def _relu_flops(module, inp, out):
    return inp.numel()


def _softmax_flops(module, inp, out):
    batch_size, nfeatures = inp.size()
    # exp: nfeatures, add: nfeatures-1, div: nfeatures
    total_ops = batch_size * (3 * nfeatures - 1)
    return total_ops


def _avgpool_flops(module, inp, out):
    # pool: kernel size, avg: 1
    kernel_ops = _torch.prod(_torch.Tensor([module.kernel_size]))
    total_ops = (kernel_ops + 1) * out.numel()
    return total_ops


def _adap_avgpool_flops(module, inp, out):
    # pool: kernel size, avg: 1
    kernel_size = _torch.Tensor(list(inp.size()[2:])) // \
                  _torch.Tensor(list((module.output_size,))).squeeze()
    kernel_ops = _torch.prod(kernel_size)
    total_ops = (kernel_ops + 1) * out.numel()
    return total_ops


def _linear_flops(module, inp, out):
    mul = module.in_features
    add = module.in_features - 1
    total_ops = (mul + add) * out.numel()
    return total_ops


def _upsample_flops(module, inp, out):
    if module.mode not in ("nearest", "linear", "bilinear", "bicubic",
                           "trilinear"):
        factor = _zero_flops(module, inp, out)
    elif module.mode == "nearest":
        factor = _zero_flops(module, inp, out)
    elif module.mode == "linear":
        factor = 5  # 2 muls + 3 add
    elif module.mode == "bilinear":
        factor = 13  # 6 muls + 7 adds
    elif module.mode == "bicubic":
        # Product matrix [4x4] x [4x4] x [4x4]
        # 224 = 128 muls + 96 adds, 35 # 16 muls + 12 adds + 4 muls + 3 adds
        factor = (224 + 35)
    elif module.mode == "trilinear":
        # can viewed as 2 bilinear + 1 linear
        factor = (13 * 2 + 5)
    total_ops = out.numel() * factor
    return total_ops


def _compute_flops(module, inp, out):
    flop_func = {
        _nn.Conv1d: _convNd_flops,
        _nn.Conv2d: _convNd_flops,
        _nn.Conv3d: _convNd_flops,
        _nn.ConvTranspose1d: _convNd_flops,
        _nn.ConvTranspose2d: _convNd_flops,
        _nn.ConvTranspose3d: _convNd_flops,
        _nn.BatchNorm1d: _bn_flops,
        _nn.BatchNorm2d: _bn_flops,
        _nn.BatchNorm3d: _bn_flops,
        _nn.ReLU: _zero_flops,
        _nn.ReLU6: _zero_flops,
        _nn.LeakyReLU: _relu_flops,
        _nn.MaxPool1d: _zero_flops,
        _nn.MaxPool2d: _zero_flops,
        _nn.MaxPool3d: _zero_flops,
        _nn.AdaptiveMaxPool1d: _zero_flops,
        _nn.AdaptiveMaxPool2d: _zero_flops,
        _nn.AdaptiveMaxPool3d: _zero_flops,
        _nn.AvgPool1d: _avgpool_flops,
        _nn.AvgPool2d: _avgpool_flops,
        _nn.AvgPool3d: _avgpool_flops,
        _nn.AdaptiveAvgPool1d: _adap_avgpool_flops,
        _nn.AdaptiveAvgPool2d: _adap_avgpool_flops,
        _nn.AdaptiveAvgPool3d: _adap_avgpool_flops,
        _nn.Linear: _linear_flops,
        _nn.Dropout: _zero_flops,
        _nn.Upsample: _upsample_flops,
        _nn.UpsamplingBilinear2d: _upsample_flops,
        _nn.UpsamplingNearest2d: _upsample_flops,
        _nn.Softmax: _softmax_flops
    }
    if type(module) in flop_func:
        if isinstance(inp, tuple):
            inp = inp[0]
        return flop_func[type(module)](module, inp, out)
    else:
        return 0
