import unittest as _unittest

import torch as _torch
import torch.nn as _nn
import torchvision as _torchvision

import torchutils as _tu


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


class _TestModels(_unittest.TestCase):

    def test_get_model_param_count(self):
        model = _torchvision.models.alexnet()
        alexnet_params = 61100840
        total_params = _tu.get_model_param_count(model)
        trainable_params = _tu.get_model_param_count(model, True)
        nontrainable_params = _tu.get_model_param_count(model, False)
        # verify parameters for alexnet, it has 0 non-trainable params.
        self.assertEqual(total_params, alexnet_params)
        self.assertEqual(trainable_params, alexnet_params)
        self.assertEqual(nontrainable_params, 0)

    def test_get_model_flops(self):
        model = _SampleNetwork()
        alexnet_flops = 774416847
        alexnet_mflops = 774.42
        alexnet_gflops = 0.77
        flops = _tu.get_model_flops(model, _torch.rand((1, 3, 224, 224)))
        mflops = _tu.get_model_flops(model, _torch.rand((1, 3, 224, 224)),
                                     'MFLOP')
        gflops = _tu.get_model_flops(model, _torch.rand((1, 3, 224, 224)),
                                     'GFLOP')
        self.assertAlmostEqual(flops, alexnet_flops)
        self.assertAlmostEqual(mflops, alexnet_mflops)
        self.assertAlmostEqual(gflops, alexnet_gflops)


if __name__ == '__main__':
    _unittest.main()
