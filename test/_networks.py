import torch as _torch
import torch.nn as _nn


class _SampleNetwork(_nn.Module):

    def __init__(self, num_classes=1000):
        super(_SampleNetwork, self).__init__()
        self.features = _nn.Sequential(
            _nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            _nn.ReLU(inplace=True),
            _nn.BatchNorm2d(64),
            _nn.AvgPool2d(kernel_size=3, stride=2),
            _nn.Conv2d(64, 192, kernel_size=5, padding=2),
            _nn.LeakyReLU(inplace=True),
            _nn.MaxPool2d(kernel_size=3, stride=2),
            _nn.Conv2d(192, 384, kernel_size=3, padding=1),
            _nn.ReLU(inplace=True),
            _nn.Conv2d(384, 256, kernel_size=3, padding=1),
            _nn.ReLU(inplace=True),
            _nn.Conv2d(256, 256, kernel_size=3, padding=1),
            _nn.ReLU(inplace=True),
            _nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = _nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = _nn.Sequential(_nn.Dropout(),
                                         _nn.Linear(256 * 6 * 6, 4096),
                                         _nn.LeakyReLU(inplace=True),
                                         _nn.Dropout(), _nn.Linear(4096, 4096),
                                         _nn.LeakyReLU(inplace=True),
                                         _nn.Linear(4096, num_classes),
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
