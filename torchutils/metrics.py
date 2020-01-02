import collections as _collections

import torch as _torch

from ._validate import _validate_param


class Accuracy():
    """Calculate and track accuracy of predictions.

    Example::

        import torch
        import torchutils as tu

        acc = tu.Accuracy()
        # This is sample loop over batches
        for i in range(5):
            # generate random batch (only for example)
            targets = torch.randint(1, 10, (10, 3, 2, 2))
            predictions = torch.randint(-1, 0, (10, 3, 2, 2))
            # set 5 predictions equal to targets
            dup_idx = _torch.randperm(10)[:5]
            predictions[dup_idx] = targets[dup_idx]
            # track accuracy
            batch_acc = acc.update(targets, predictions)
            print('Running accuracy: {}'.format(acc.accuracy))
        print('Total accuracy: {}'.format(acc.accuracy))

    Out::

        Running accuracy: 50.0
        Running accuracy: 50.0
        Running accuracy: 50.0
        Running accuracy: 50.0
        Running accuracy: 50.0
        Total accuracy: 50.0

    """

    def __init__(self, keep_hist=False, hist_size=0, hist_freq=1):
        """Initialize accuracy object.

        Args:
            keep_hist (bool): Set as True to save accuracy history.
                              (default:False)
            hist_size (int): Size of accuracy history buffer.
                             (default:0 means infinite)
            hist_freq (int): Frequency of storing the history.
                             (default:1 means store every iteration)

        Returns:
            None: Returns nothing.

        """

        _validate_param(keep_hist, 'keep_hist', 'bool')
        _validate_param(hist_size, 'hist_size', 'int')
        _validate_param(hist_freq, 'hist_freq', 'int')
        self._size = None
        self._iter_count = 0
        self._num_samples = 0
        self._keep_hist = keep_hist
        self._hist_size = hist_size
        self._hist_freq = hist_freq
        self._history_start_iter = 1
        self._correct_predictions = 0
        self._history = _collections.deque()

    def update(self, targets, predictions):
        """Update accuracy tracker.

        Args:
            targets (torch.Tensor): Targets. Must be (N, \\*).
            predictions (torch.Tensor): Model predictions. Must be (N, \\*).

        Returns:
            float: Accuracy of current batch of predictions.

        """

        _validate_param(targets, 'targets', 'tensor')
        _validate_param(predictions, 'predictions', 'tensor')
        tsize, psize = targets.size(), predictions.size()
        if tsize != psize:
            raise ValueError("targets {} and predictions {} must be of "
                             "same shape.".format(tuple(tsize), tuple(psize)))
        # extract non-batch dimensions
        if not self._size:
            self._size = tsize[1:]
        # verify that current input size is same as previous inputs
        # excluding batch dimension
        if self._size != tsize[1:]:
            raise ValueError(
                "Excluding batch dimension, previous input size was {}, but"
                " current input size is {}.".format(
                    tuple(self._size), tuple(tsize[1:])))
        # find number of correct predictions in the batch
        batch_size = tsize[0]
        num_dims = len(tsize) - 1
        if num_dims == 0:
            targets = targets.reshape(-1, 1)
            predictions = predictions.reshape(-1, 1)
            num_dims = 1
        matches = (targets == predictions).all(-1)
        for _ in range(num_dims - 1):
            matches = matches.all(-1)
        correct_predictions = matches.sum().item()
        iter_acc = (correct_predictions / batch_size) * 100
        self._num_samples += batch_size
        self._correct_predictions += correct_predictions
        # store in history depending on frequency
        if self._keep_hist and ((self._iter_count % self._hist_freq) == 0):
            self._history.append(iter_acc)
            if len(self._history) > self._hist_size:
                self._history.popleft()
                self._history_start_iter += self._hist_freq
        self._iter_count += 1
        return iter_acc

    @property
    def accuracy(self):
        """accuracy (float): Current running accuracy."""
        return (self._correct_predictions / self._num_samples) * 100

    @accuracy.setter
    def accuracy(self, val):
        raise NotImplementedError('Modifying accuracy is not supported.')

    @property
    def history(self):
        """history (dict): Accuracy values for past iterations.
                "acc" (list): Accuracy values.
                "iter" (list): Iteration numbers.
        """
        hist_dict = {}
        hist_dict["acc"] = list(self._history)
        hist_dict["iter"] = list(
            range(self._history_start_iter,
                  self._history_start_iter + len(self._history),
                  self._hist_freq))
        return hist_dict

    @history.setter
    def history(self, val):
        raise NotImplementedError('Modifying history is not supported.')

    def reset(self):
        """Reset accuracy tracker.

        Returns:
            None: Returns nothing.
        """
        self._size = None
        self._iter_count = 0
        self._num_samples = 0
        self._history_start_iter = 1
        self._correct_predictions = 0
        self._history.clear()
