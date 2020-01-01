import unittest as _unittest

import torch as _torch
import torchvision as _torchvision

import torchutils as _tu


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
        model = _torchvision.models.alexnet()
        alexnet_flops = 773304664
        alexnet_mflops = 773.3
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
