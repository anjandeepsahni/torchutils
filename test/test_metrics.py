import unittest as _unittest
from contextlib import redirect_stdout as _redirect_stdout
from io import StringIO as _StringIO

import numpy as _np
import torch as _torch

import torchutils as _tu


class _TestMetrics(_unittest.TestCase):

    def test_accuracy_0d(self):
        # suppress summary print output
        with _redirect_stdout(_StringIO()) as _:
            acc = _tu.Accuracy(keep_hist=True)
            acc.reset()
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
                print(acc)  # coverage for printing
                self.assertAlmostEqual(batch_acc, num_dup * 10)
            self.assertAlmostEqual(correct_predictions, acc.accuracy)
            # check history size
            history = acc.history
            self.assertEqual(len(history["metric"]), 10)
            self.assertEqual(len(history["iteration"]), 10)

    def test_accuracy_1d(self):
        acc = _tu.Accuracy(keep_hist=True, hist_size=5)
        correct_predictions = 0
        test_acc = _torch.randint(1, 10, (10, ))
        for i in range(10):
            # generate random
            targets = _torch.randint(1, 10, (10, 3))
            predictions = _torch.randint(-1, 0, (10, 3))
            num_dup = test_acc[i].item()
            dup_idx = _torch.randperm(10)[:num_dup]
            predictions[dup_idx] = targets[dup_idx]
            correct_predictions += num_dup
            batch_acc = acc.update(targets, predictions)
            self.assertAlmostEqual(batch_acc, num_dup * 10)
        self.assertAlmostEqual(correct_predictions, acc.accuracy)
        # check history size
        history = acc.history
        self.assertEqual(len(history["metric"]), 5)
        self.assertEqual(len(history["iteration"]), 5)
        # check history values
        test_acc = (test_acc[-5:] * 10.0).float().numpy()
        result_acc = _np.array(history["metric"])
        # to str for floating point error
        self.assertTrue(str(test_acc) == str(result_acc))

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

    def test_running_loss(self):
        running_loss = _tu.RunningLoss(keep_hist=True)
        running_loss.reset()
        # dummy list of loss values for testing
        test_losses = _np.random.random(10)
        for i in range(10):
            avg_loss = running_loss.update(test_losses[i])
            true_avg_loss = _np.mean(test_losses[:i + 1])
            self.assertAlmostEqual(avg_loss, true_avg_loss)
        # check history size
        history = running_loss.history
        self.assertEqual(len(history["metric"]), 10)
        self.assertEqual(len(history["iteration"]), 10)

    def test_hamming_loss(self):
        hamming_loss = _tu.HammingLoss(keep_hist=True)
        test_loss = _torch.randint(1, 50, (10, ))
        for i in range(10):
            # generate random
            targets = _torch.randint(1, 10, (5, 10)) > 7
            targets = targets.type(_torch.uint8)
            predictions = targets.clone()
            # setup incorrect predictions.
            incorrect = _torch.randperm(50)[:test_loss[i]]
            for loc in incorrect:
                loc_item = loc.item()
                row = int(loc_item / 10)
                col = loc_item % 10
                predictions[row][col] = int(not predictions[row][col])
            batch_loss = hamming_loss.update(targets, predictions)
            self.assertAlmostEqual(batch_loss, test_loss[i].item() / 50)
        test_final_loss = test_loss.sum().item() / (len(test_loss) * 50)
        result_final_loss = hamming_loss.loss
        self.assertAlmostEqual(result_final_loss, test_final_loss)
        # check history size
        history = hamming_loss.history
        self.assertEqual(len(history["metric"]), 10)
        self.assertEqual(len(history["iteration"]), 10)
        # check history values
        test_loss = (test_loss / 50.0).float().numpy().round(4)
        result_loss = _np.array(history["metric"]).round(4)
        # to str for floating point error
        self.assertTrue(str(test_loss) == str(result_loss))


if __name__ == '__main__':
    _unittest.main()
