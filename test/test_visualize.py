import os as _os
import unittest as _unittest

import torch as _torch
import torchvision as _torchvision

import torchutils as _tu


class _TestVisualize(_unittest.TestCase):

    def test_plot_gradients_line(self):
        criterion = _torch.nn.CrossEntropyLoss()
        net = _torchvision.models.alexnet(num_classes=10)
        # forward pass
        out = net(_torch.rand(1, 3, 224, 224))
        # get loss and gradients
        ground_truth = _torch.randint(0, 10, (1, ))
        loss = criterion(out, ground_truth)
        loss.backward()
        # plot gradients
        file_path = './temp/grad_flow.png'
        _tu.plot_gradients(net, file_path, plot_type='line', plot_max=True,
                           ylim=(0.0, 0.02))
        self.assertTrue(_os.path.isfile(file_path))
        # remove temporary directory
        _os.remove(file_path)
        _os.rmdir(_os.path.dirname(file_path))

    def test_plot_gradients_bar(self):
        criterion = _torch.nn.CrossEntropyLoss()
        net = _torchvision.models.alexnet(num_classes=10)
        # forward pass
        out = net(_torch.rand(1, 3, 224, 224))
        # get loss and gradients
        ground_truth = _torch.randint(0, 10, (1, ))
        loss = criterion(out, ground_truth)
        loss.backward()
        # plot gradients
        file_path = './temp/grad_flow.png'
        _tu.plot_gradients(net, file_path, plot_type='bar', plot_max=True,
                           ylim=(0.0, 0.02))
        self.assertTrue(_os.path.isfile(file_path))
        # remove temporary directory
        _os.remove(file_path)
        _os.rmdir(_os.path.dirname(file_path))


if __name__ == '__main__':
    _unittest.main()
