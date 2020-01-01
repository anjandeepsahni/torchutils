import random as _random
import unittest as _unittest

import torch.optim as _optim
import torchvision as _torchvision

import torchutils as _tu


class _TestLearningRate(_unittest.TestCase):

    def test_get_lr(self):
        model = _torchvision.models.alexnet()
        optimizer = _optim.Adam(model.parameters())
        default_adamlr = 0.001
        current_lr = _tu.get_lr(optimizer)
        self.assertAlmostEqual(current_lr, default_adamlr)

    def test_set_lr(self):
        model = _torchvision.models.alexnet()
        optimizer = _optim.Adam(model.parameters())
        new_lr = round(_random.uniform(0.0001, 0.0009), 4)
        optimizer = _tu.set_lr(optimizer, new_lr)
        revised_lr = _tu.get_lr(optimizer)
        self.assertTrue(False)  # FIXME: For travis testing.
        self.assertAlmostEqual(new_lr, revised_lr)


if __name__ == '__main__':
    _unittest.main()
