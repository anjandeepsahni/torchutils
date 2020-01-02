import unittest as _unittest

import torch as _torch

import torchutils as _tu


class _TestMetrics(_unittest.TestCase):

    def test_accuracy_0d(self):
        acc = _tu.Accuracy()
        correct_predictions = 0
        for i in range(10):
            # generate random
            targets = _torch.randint(1, 10, (10, ))
            predictions = _torch.randint(-1, 0, (10, ))
            num_dup = _torch.randint(1, 10, (1, )).item()
            dup_idx = _torch.randperm(10)[:num_dup]
            predictions[dup_idx] = targets[dup_idx]
            correct_predictions += num_dup
            batch_acc = acc.update(targets, predictions)
            self.assertAlmostEqual(batch_acc, num_dup * 10)
        self.assertAlmostEqual(correct_predictions, acc.accuracy)

    def test_accuracy_1d(self):
        acc = _tu.Accuracy()
        correct_predictions = 0
        for i in range(10):
            # generate random
            targets = _torch.randint(1, 10, (
                10,
                3,
            ))
            predictions = _torch.randint(-1, 0, (
                10,
                3,
            ))
            num_dup = _torch.randint(1, 10, (1, )).item()
            dup_idx = _torch.randperm(10)[:num_dup]
            predictions[dup_idx] = targets[dup_idx]
            correct_predictions += num_dup
            batch_acc = acc.update(targets, predictions)
            self.assertAlmostEqual(batch_acc, num_dup * 10)
        self.assertAlmostEqual(correct_predictions, acc.accuracy)

    def test_accuracy_2d(self):
        acc = _tu.Accuracy()
        correct_predictions = 0
        for i in range(10):
            # generate random
            targets = _torch.randint(1, 10, (10, 3, 2))
            predictions = _torch.randint(-1, 0, (10, 3, 2))
            num_dup = _torch.randint(1, 10, (1, )).item()
            dup_idx = _torch.randperm(10)[:num_dup]
            predictions[dup_idx] = targets[dup_idx]
            correct_predictions += num_dup
            batch_acc = acc.update(targets, predictions)
            self.assertAlmostEqual(batch_acc, num_dup * 10)
        self.assertAlmostEqual(correct_predictions, acc.accuracy)

    def test_accuracy_3d(self):
        acc = _tu.Accuracy()
        correct_predictions = 0
        for i in range(10):
            # generate random
            targets = _torch.randint(1, 10, (10, 3, 2, 2))
            predictions = _torch.randint(-1, 0, (10, 3, 2, 2))
            num_dup = _torch.randint(1, 10, (1, )).item()
            dup_idx = _torch.randperm(10)[:num_dup]
            predictions[dup_idx] = targets[dup_idx]
            correct_predictions += num_dup
            batch_acc = acc.update(targets, predictions)
            self.assertAlmostEqual(batch_acc, num_dup * 10)
        self.assertAlmostEqual(correct_predictions, acc.accuracy)


if __name__ == '__main__':
    _unittest.main()
