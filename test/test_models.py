import unittest as _unittest
import warnings as _warnings
from contextlib import redirect_stdout as _redirect_stdout
from io import StringIO as _StringIO

import torch as _torch
import torchvision as _torchvision

import torchutils as _tu

from ._networks import (_CoverageNetwork, _MultipleInputNetwork,
                        _RecursiveNetwork, _SequenceNetwork)


class _TestModels(_unittest.TestCase):

    def test_get_model_param_count(self):
        model = _torchvision.models.alexnet()
        alexnet_params = 61100840
        total_params = _tu.get_model_param_count(model)
        # verify parameters for alexnet, it has 0 non-trainable params.
        self.assertEqual(total_params, alexnet_params)

    def test_get_model_param_count_trainable(self):
        model = _torchvision.models.alexnet()
        alexnet_params = 61100840
        trainable_params = _tu.get_model_param_count(model, True)
        # verify parameters for alexnet, it has 0 non-trainable params.
        self.assertEqual(trainable_params, alexnet_params)

    def test_get_model_param_count_nontrainable(self):
        model = _torchvision.models.alexnet()
        nontrainable_params = _tu.get_model_param_count(model, False)
        # verify parameters for alexnet, it has 0 non-trainable params.
        self.assertEqual(nontrainable_params, 0)

    def test_get_model_flops(self):
        # to suppress userwarning for upsample
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=UserWarning)
            model = _CoverageNetwork()
            samplenet_flops = 5395219
            flops = _tu.get_model_flops(model, _torch.rand((1, 3, 28, 28)))
            self.assertAlmostEqual(flops, samplenet_flops)

    def test_get_model_mflops(self):
        # to suppress userwarning for upsample
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=UserWarning)
            model = _CoverageNetwork()
            samplenet_mflops = 5.4
            mflops = _tu.get_model_flops(model, _torch.rand((1, 3, 28, 28)),
                                         unit='MFLOP')
            self.assertAlmostEqual(mflops, samplenet_mflops)

    def test_get_model_gflops(self):
        # to suppress userwarning for upsample
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=UserWarning)
            model = _CoverageNetwork()
            samplenet_gflops = 0.01
            gflops = _tu.get_model_flops(model, _torch.rand((1, 3, 28, 28)),
                                         unit='GFLOP')
            self.assertAlmostEqual(gflops, samplenet_gflops)

    def test_get_model_flops_lstm(self):
        model = _SequenceNetwork(mode='lstm')
        lstm_flops = 42239200
        sentence = _torch.randint(1, 10, (200, )).long()
        flops = _tu.get_model_flops(model, [sentence], [len(sentence)])
        self.assertAlmostEqual(flops, lstm_flops)

    def test_get_model_flops_gru(self):
        model = _SequenceNetwork(mode='gru')
        lstm_flops = 32357600
        sentence = _torch.randint(1, 10, (200, )).long()
        flops = _tu.get_model_flops(model, [sentence], [len(sentence)])
        self.assertAlmostEqual(flops, lstm_flops)

    def test_get_model_flops_rnn(self):
        model = _SequenceNetwork(mode='rnn')
        lstm_flops = 12479200
        sentence = _torch.randint(1, 10, (200, )).long()
        flops = _tu.get_model_flops(model, [sentence], [len(sentence)])
        self.assertAlmostEqual(flops, lstm_flops)

    def test_get_model_summary_compact(self):
        # suppress summary print output
        with _redirect_stdout(_StringIO()) as _:
            # test compact representation
            model = _torchvision.models.alexnet()
            _tu.get_model_summary(model, _torch.rand((1, 3, 224, 224)),
                                  compact=True)
        # pass if no error in printing summary
        self.assertTrue(True)

    def test_get_model_summary_recursive(self):
        # suppress summary print output
        with _redirect_stdout(_StringIO()) as _:
            # test recursive network
            model = _RecursiveNetwork()
            _tu.get_model_summary(model, _torch.rand((1, 10)))
        # pass if no error in printing summary
        self.assertTrue(True)

    def test_get_model_summary_multi_input(self):
        # suppress summary print output
        with _redirect_stdout(_StringIO()) as _:
            # test multiple input network
            model = _MultipleInputNetwork()
            _tu.get_model_summary(model, _torch.rand((1, 3, 28, 28)),
                                  _torch.rand((1, 3, 28, 28)))
        # pass if no error in printing summary
        self.assertTrue(True)


if __name__ == '__main__':
    _unittest.main()
