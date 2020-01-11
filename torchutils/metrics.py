import collections as _collections
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

from ._validate import _validate_param

__all__ = ['Accuracy', 'HammingLoss', 'RunningLoss']


class _MetricTracker(_ABC):

    def __init__(self, name, fmt=':f', keep_hist=False, hist_size=0,
                 hist_freq=1):
        _validate_param(name, 'name', 'str')
        _validate_param(fmt, 'fmt', 'str')
        _validate_param(keep_hist, 'keep_hist', 'bool')
        _validate_param(hist_size, 'hist_size', 'int')
        _validate_param(hist_freq, 'hist_freq', 'int')
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._fmt = fmt
        self._size = None
        self._name = name
        self._iter_count = 0
        self._num_samples = 0
        self._keep_hist = keep_hist
        self._hist_size = hist_size
        self._hist_freq = hist_freq
        self._history_start_iter = 1
        self._correct_predictions = 0
        self._history = _collections.deque()

    def __str__(self):
        fmtstr = '{_name} - Val: {_val' + self._fmt \
                    + '} Avg: {_avg' + self._fmt + '}'
        return fmtstr.format(**self.__dict__)

    @_abstractmethod
    def reset(self):
        self._size = None
        self._iter_count = 0
        self._num_samples = 0
        self._history_start_iter = 1
        self._correct_predictions = 0
        self._history.clear()

    @_abstractmethod
    def update(self, targets, predictions):
        pass

    @property
    @_abstractmethod
    def history(self):
        hist_dict = {}
        hist_dict["metric"] = list(self._history)
        hist_dict["iteration"] = list(
            range(self._history_start_iter,
                  self._history_start_iter + len(self._history),
                  self._hist_freq))
        return hist_dict

    def _validate_update_inputs(self, targets, predictions):
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

    def _update_history(self, iter_val):
        if self._keep_hist and ((self._iter_count % self._hist_freq) == 0):
            self._history.append(iter_val)
            if self._hist_size != 0 and len(self._history) > self._hist_size:
                self._history.popleft()
                self._history_start_iter += self._hist_freq


class Accuracy(_MetricTracker):
    """Calculate and track accuracy of predictions.

    Args:
        keep_hist (bool): Set as True to save accuracy history.
                            (default:False)
        hist_size (int): Size of accuracy history buffer.
                            (default:0 means infinite)
        hist_freq (int): Frequency of storing the history.
                            (default:1 means store every iteration)

    Example::

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        import torchvision
        import torchvision.transforms as transforms
        import torchutils as tu

        # define your network
        model = MyNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        trainset = torchvision.datasets.MNIST(root='./data/', train=True,
                                            download=True,
                                            transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=60,
                                                shuffle=True, num_workers=2,
                                                drop_last=True)
        n_epochs = 1
        model.train()
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch + 1, n_epochs))
            acc_tracker = tu.Accuracy()
            for batch_idx, (data, target) in enumerate(trainloader):
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                acc_tracker.update(target, predicted)
                if batch_idx % 100 == 0:
                    print(acc_tracker)

    Out::

        Epoch: 1/1
        Accuracy - Val: 10.0000 Avg: 10.0000
        Accuracy - Val: 91.6667 Avg: 70.9406
        Accuracy - Val: 86.6667 Avg: 79.5937
        Accuracy - Val: 93.3333 Avg: 83.1063
        Accuracy - Val: 90.0000 Avg: 85.4032
        Accuracy - Val: 88.3333 Avg: 86.9627
        Accuracy - Val: 95.0000 Avg: 88.1364
        Accuracy - Val: 95.0000 Avg: 89.1702
        Accuracy - Val: 93.3333 Avg: 89.9459
        Accuracy - Val: 95.0000 Avg: 90.5161

    """

    def __init__(self, keep_hist=False, hist_size=0, hist_freq=1):
        super().__init__(name="Accuracy", fmt=":.4f", keep_hist=keep_hist,
                         hist_size=hist_size, hist_freq=hist_freq)

    def reset(self):
        """Reset accuracy tracker.

        Returns:
            None: Returns nothing.
        """

        super().reset()

    def update(self, targets, predictions):
        """Update accuracy tracker.

        Args:
            targets (torch.Tensor): Targets. Must be (N, \\*).
            predictions (torch.Tensor): Model predictions. Must be (N, \\*).

        Returns:
            float: Accuracy of current batch of predictions (percentage).

        """

        self._validate_update_inputs(targets, predictions)
        tsize = targets.size()
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
        self._val = iter_acc
        self._num_samples += batch_size
        self._correct_predictions += correct_predictions
        self._avg = self.accuracy
        # store in history depending on frequency
        self._update_history(iter_acc)
        self._iter_count += 1
        return iter_acc

    @property
    def accuracy(self):
        """float: Current running accuracy (percentage)."""

        return (self._correct_predictions / self._num_samples) * 100

    @property
    def history(self):
        """dict {"metric" -> list of accuracy values, \
            "iteration" -> list of iteration numbers} \
            : Accuracy values for past iterations.
        """

        return super().history


class HammingLoss(_MetricTracker):
    """Calculate and track hamming loss of predictions.

    Hamming loss is an evaluation metric for multilabel classification \
    problem. It is the fraction of labels that are incorrectly predicted.

    Args:
        keep_hist (bool): Set as True to save accuracy history.
                            (default:False)
        hist_size (int): Size of accuracy history buffer.
                            (default:0 means infinite)
        hist_freq (int): Frequency of storing the history.
                            (default:1 means store every iteration)

    Example::

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchutils as tu

        # define your network and trainloader
        model = MyNet()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MultiLabelSoftMarginLoss()

        n_epochs = 1
        model.train()
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch + 1, n_epochs))
            ham_tracker = tu.HammingLoss()
            for batch_idx, (data, target) in enumerate(trainloader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                predicted = torch.sigmoid(output) > 0.5
                ham_tracker.update(target, predicted)
                if batch_idx % 100 == 0:
                    print(ham_tracker)

    Out::

        Epoch: 1/1
        Hamming Loss - Val: 0.6667 Avg: 0.6667
        Hamming Loss - Val: 1.0000 Avg: 0.8333
        Hamming Loss - Val: 1.0000 Avg: 0.8889
        Hamming Loss - Val: 0.0000 Avg: 0.6667
        Hamming Loss - Val: 0.6667 Avg: 0.6667
        Hamming Loss - Val: 0.6667 Avg: 0.6667
        Hamming Loss - Val: 1.0000 Avg: 0.7143
        Hamming Loss - Val: 0.6667 Avg: 0.7083
        Hamming Loss - Val: 0.0000 Avg: 0.6296
        Hamming Loss - Val: 0.0000 Avg: 0.5667

    """

    def __init__(self, keep_hist=False, hist_size=0, hist_freq=1):
        super().__init__(name="Hamming Loss", fmt=":.4f", keep_hist=keep_hist,
                         hist_size=hist_size, hist_freq=hist_freq)

    def reset(self):
        """Reset hamming loss tracker.

        Returns:
            None: Returns nothing.
        """

        super().reset()

    def update(self, targets, predictions):
        """Update hamming loss tracker.

        Args:
            targets (bool torch.Tensor): Targets. Must be (N, Classes).
            predictions (bool torch.Tensor): Model predictions.
                                             Must be (N, Classes).

        Returns:
            float: Hamming loss of current batch of predictions.

        """

        self._validate_update_inputs(targets, predictions)
        tsize = targets.size()
        # verify that input is (N,Classes)
        if len(self._size) != 1:
            raise ValueError(
                "Expected targets shape to be (N, Classes) but got {}".format(
                    tuple(tsize)))
        # find number of correct predictions in the batch
        batch_size = tsize[0]
        num_labels = self._size[0]
        wrong_pred = (targets.byte() ^ predictions.byte()).sum().item()
        correct_predictions = batch_size * num_labels - wrong_pred
        iter_loss = wrong_pred / (num_labels * batch_size)
        self._val = iter_loss
        self._num_samples += batch_size
        self._correct_predictions += correct_predictions
        self._avg = self.loss
        # store in history depending on frequency
        self._update_history(iter_loss)
        self._iter_count += 1
        return iter_loss

    @property
    def loss(self):
        """float: Current running hamming loss."""

        num_labels = self._size[0]
        return (1 - (self._correct_predictions /
                     (num_labels * self._num_samples)))

    @property
    def history(self):
        """dict {"metric" -> list of hamming loss values, \
            "iteration" -> list of iteration numbers} \
            : Hamming loss values for past iterations.
        """

        return super().history


class RunningLoss(_MetricTracker):
    """Track and maintain running average of loss.

    Args:
        keep_hist (bool): Set as True to save accuracy history.
                            (default:False)
        hist_size (int): Size of accuracy history buffer.
                            (default:0 means infinite)
        hist_freq (int): Frequency of storing the history.
                            (default:1 means store every iteration)

    Example::

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision
        import torchvision.transforms as transforms
        import torchutils as tu

        # define your network
        model = MyNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        trainset = torchvision.datasets.MNIST(root='./data/', train=True,
                                            download=True,
                                            transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=60,
                                                shuffle=True, num_workers=2,
                                                drop_last=True)
        n_epochs = 1
        model.train()
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch + 1, n_epochs))
            loss_tracker = tu.RunningLoss()
            for batch_idx, (data, target) in enumerate(trainloader):
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss_tracker.update(loss.item())
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(loss_tracker)

    Out::

        Epoch: 1/1
        Loss - Val: 2.2921 Avg: 2.2921
        Loss - Val: 0.5084 Avg: 0.9639
        Loss - Val: 0.6027 Avg: 0.6588
        Loss - Val: 0.1817 Avg: 0.5255
        Loss - Val: 0.1005 Avg: 0.4493
        Loss - Val: 0.2982 Avg: 0.3984
        Loss - Val: 0.3103 Avg: 0.3615
        Loss - Val: 0.0940 Avg: 0.3296
        Loss - Val: 0.0957 Avg: 0.3071
        Loss - Val: 0.0229 Avg: 0.2875

    """

    def __init__(self, keep_hist=False, hist_size=0, hist_freq=1):
        super().__init__(name="Loss", fmt=":.4f", keep_hist=keep_hist,
                         hist_size=hist_size, hist_freq=hist_freq)

    def reset(self):
        """Reset running loss tracker.

        Returns:
            None: Returns nothing.
        """

        super().reset()

    def update(self, val):
        """Update running loss tracker.

        Args:
            val (float): Loss value.

        Returns:
            float: Running (average) loss after latest update.

        """

        _validate_param(val, 'val', 'float')
        self._val = val
        self._num_samples += 1
        self._sum += val
        self._avg = self.loss
        # store in history depending on frequency
        self._update_history(val)
        self._iter_count += 1
        return self.loss

    @property
    def loss(self):
        """float: Current running (average) loss."""

        return self._sum / self._num_samples

    @property
    def history(self):
        """dict {"metric" -> list of loss values, \
            "iteration" -> list of iteration numbers} \
            : Loss values for past iterations.
        """

        return super().history
