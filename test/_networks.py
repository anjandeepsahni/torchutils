import torch as _torch
import torch.nn as _nn


class _CoverageNetwork(_nn.Module):

    def __init__(self):
        super(_CoverageNetwork, self).__init__()
        self.features = _nn.Sequential(
            _nn.Upsample(scale_factor=1, mode='nearest'),
            _nn.Upsample(scale_factor=1, mode='bilinear'),
            _nn.Upsample(scale_factor=1, mode='bicubic'),
            _nn.Conv2d(3, 64, kernel_size=7), _nn.LeakyReLU(inplace=True),
            _nn.BatchNorm2d(64))
        self.avgpool = _nn.AvgPool2d(kernel_size=7)
        self.classifier = _nn.Sequential(_nn.Dropout(), _nn.Linear(576, 10),
                                         _nn.Softmax(-1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = _torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class _RecursiveNetwork(_nn.Module):

    def __init__(self):
        super(_RecursiveNetwork, self).__init__()
        self.l1 = _nn.Linear(10, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l1(x)
        x = self.l1(x)
        return x


class _MultipleInputNetwork(_nn.Module):

    def __init__(self):
        super(_MultipleInputNetwork, self).__init__()
        self.conv = _nn.Conv2d(3, 16, 3)

    def forward(self, inp1, inp2):
        inp = inp1 * inp2
        out = self.conv(inp)
        return out
