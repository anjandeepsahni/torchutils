import torch as _torch

from ._validate import _validate_param

__all__ = ['RunningStat', 'get_dataset_stats']


class RunningStat():
    """Calculate and track statistics of data.

    Calculates mean, standard deviation and variance. Uses Welford's \
    algorithm for computing the statistics. See Knuth TAOCP Vol 2, \
    3rd edition, page 232.

    Args:
        dims (int): Number of dimensions of stat. Example, 3 for RGB images.
                    (default: 3)

    Example::

        import torchutils as tu

        # define your dataset and dataloader
        sample, _ = loader.dataset[0]
        running_stat = tu.RunningStat(dims=sample.size(1))
        for batch_idx, (data, _) in enumerate(loader):
            # data must be (N,C)
            running_stat.update(data)
        print('Mean:', running_stat.mean)
        print('Std:', running_stat.std)

    Out::

        Mean: tensor([10000.0029,  9999.9941, 10000.0137])
        Std: tensor([1.0037, 1.0009, 0.9997])

    """

    def __init__(self, dims=3):
        _validate_param(dims, 'dims', 'int')
        self._n = 0
        self._dims = dims
        self._mean = _torch.zeros(dims)
        self._var = _torch.zeros(dims)

    def reset(self):
        """Reset stats tracker.

        Returns:
            None: Returns nothing.
        """

        self.__init__(dims=self._dims)

    def update(self, data):
        """Update running stats tracker.

        Args:
            data (torch.Tensor): Input data. Must be (N, dims).

        Returns:
            dict {"mean" -> Mean, "std" -> Standard deviation, \
            "var" -> Variance} \
            : Current stats of entire data.

        """

        _validate_param(data, 'data', 'tensor')
        # validate data size
        dsize = data.size()
        if len(dsize) != 2:
            raise ValueError("data must be of shape (N,{})"
                             " but got {}.".format(self._dims, tuple(dsize)))
        # loop over all data points
        for i in range(dsize[0]):
            self._n += 1
            if self._n == 1:
                self._mean = data[i]  # first data point
            else:
                diff = data[i] - self._mean
                self._mean += diff / self._n
                self._var += (diff * (data[i] - self._mean))
        stats = {'mean': self.mean, 'std': self.std, 'var': self.var}
        return stats

    @property
    def num_data_points(self):
        """int: Number of data points seen till now."""

        return self._n

    @property
    def mean(self):
        """torch.Tensor: Mean of data seen till now."""

        return self._mean[:]

    @property
    def var(self):
        """torch.Tensor: Variance of data seen till now."""

        _var = self._var[:]
        _var = _var / (self._n - 1) if self._n > 1 else _var
        return _var

    @property
    def std(self):
        """torch.Tensor: Standard deviation of data seen till now."""

        return _torch.sqrt(self.var)


def get_dataset_stats(loader, verbose=False):
    """Get statistics of dataset.

    Calculates mean, standard deviation and variance. Supports data of shape \
        (N,C) and (N,C,H,W).

    Args:
        loader (torch.utils.data.DataLoader): PyTorch dataloader.
        verbose (bool): Enable/disable print statements.

    Returns:
        dict {"mean" -> Mean, "std" -> Standard deviation, \
            "var" -> Variance} \
            : Stats of entire dataset.

    Example::

        import torch
        import torchutils as tu

        # define your dataset and dataloader
        dataset = MyDataset()
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  num_workers=1,
                                                  shuffle=False)
        stats = tu.get_dataset_stats(trainloader, verbose=True)
        print('Mean:', stats['mean'])
        print('Std:', stats['std'])

    Out::

        Calculating dataset stats...
        Batch 100/100
        Mean: tensor([10000.0098,  9999.9795,  9999.9893])
        Std: tensor([0.9969, 1.0003, 0.9972])

    """

    _validate_param(loader, 'loader', 'dataloader')
    _validate_param(verbose, 'verbose', 'bool')
    sample, _ = loader.dataset[0]
    sample.unsqueeze_(0)
    _validate_param(sample, 'Dataset sample', 'tensor')
    if len(sample.size()) not in {2, 4}:
        raise ValueError("Only data with shapes (N,C) and (N,C,H,W) are"
                         " currently supported. Recieved data with "
                         "shape {}.".format(tuple(sample.size())))
    running_stat = RunningStat(dims=sample.size(1))
    nbatches = int(len(loader))
    if nbatches == 0:
        raise ValueError("Dataloader must have at least one batch.")
    nbatches_char = len(str(nbatches))
    if verbose:
        print('Calculating dataset stats...')
    for batch_idx, (data, _) in enumerate(loader):
        if verbose:
            print(
                'Batch {:>{ch}}/{}'.format(batch_idx + 1, nbatches,
                                           ch=nbatches_char), end="\r")
        if len(data.size()) == 4:
            data = data.reshape(data.size(0), data.size(1), -1)
            data = data.permute(0, 2, 1)
            data = data.reshape(-1, data.size(2))
        # data is now (N,C)
        stats = running_stat.update(data)
    if verbose:
        print("")
    return stats
