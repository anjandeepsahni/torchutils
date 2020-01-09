import unittest as _unittest
from contextlib import redirect_stdout as _redirect_stdout
from io import StringIO as _StringIO

import torch as _torch
import torchvision as _torchvision

import torchutils as _tu

from ._networks import _RecursiveNetwork, _SampleNetwork


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
        samplenet_flops = 815360463
        samplenet_mflops = 815.36
        samplenet_gflops = 0.82
        flops = _tu.get_model_flops(model, _torch.rand((1, 3, 224, 224)))
        mflops = _tu.get_model_flops(model, _torch.rand((1, 3, 224, 224)),
                                     'MFLOP')
        gflops = _tu.get_model_flops(model, _torch.rand((1, 3, 224, 224)),
                                     'GFLOP')
        self.assertAlmostEqual(flops, samplenet_flops)
        self.assertAlmostEqual(mflops, samplenet_mflops)
        self.assertAlmostEqual(gflops, samplenet_gflops)

    def test_get_model_summary(self):
        # suppress summary print output
        with _redirect_stdout(_StringIO()) as _:
            # test compact representation
            model = _torchvision.models.alexnet()
            _tu.get_model_summary(model, _torch.rand((1, 3, 224, 224)),
                                  compact=True)
            # test with deeper network
            model = _torchvision.models.resnet152()
            _tu.get_model_summary(model, _torch.rand((1, 3, 224, 224)))
            # test recursive network
            model = _RecursiveNetwork()
            _tu.get_model_summary(model, _torch.rand((1, 10)))
        # pass if no error in printing summary
        self.assertTrue(True)


if __name__ == '__main__':
    _unittest.main()
