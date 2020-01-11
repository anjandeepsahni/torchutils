import unittest as _unittest

import torch as _torch

import torchutils as _tu
from ._networks import _DummyDataset


class _TestDatasets(_unittest.TestCase):

    def test_get_dataset_stats(self):
        dataset = _DummyDataset()
        trainloader = _torch.utils.data.DataLoader(dataset, batch_size=10,
                                                   num_workers=1,
                                                   shuffle=False)
        stats = _tu.get_dataset_stats(trainloader)
        true_mean = _torch.Tensor([10000.0, 10000.0, 10000.0])
        true_std = _torch.Tensor([1.0, 1.0, 1.0])
        self.assertTrue(_torch.allclose(stats['mean'], true_mean, atol=1e-2))
        self.assertTrue(_torch.allclose(stats['std'], true_std, atol=1e-2))


if __name__ == '__main__':
    _unittest.main()
