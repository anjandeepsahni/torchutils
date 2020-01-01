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

    def __init__(self):
        self._size = None
        self._num_samples = 0
        self._correct_predictions = 0

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
        self._num_samples += batch_size
        self._correct_predictions += correct_predictions
        return (correct_predictions / batch_size) * 100

    @property
    def accuracy(self):
        """accuracy (float): Current running accuracy."""
        return (self._correct_predictions / self._num_samples) * 100

    @accuracy.setter
    def accuracy(self, val):
        raise NotImplementedError('Modifying accuracy is not supported.')

    def reset(self):
        """Reset accuracy tracker.

        Returns:
            None: Returns nothing.
        """
        self._size = None
        self._num_samples = 0
        self._correct_predictions = 0
