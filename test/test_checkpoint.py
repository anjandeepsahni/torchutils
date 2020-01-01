import os as _os
import random as _random
import unittest as _unittest

import torch.optim as _optim
import torchvision as _torchvision

import torchutils as _tu


class _TestCheckpoint(_unittest.TestCase):

    def test_save_load_ckpt(self):
        model = _torchvision.models.alexnet()
        optimizer = _optim.Adam(model.parameters())
        scheduler = _optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
        # change optimizer lr, just for load_checkpoint test
        new_lr = round(_random.uniform(0.0001, 0.0009), 4)
        optimizer = _tu.set_lr(optimizer, new_lr)
        _tu.save_checkpoint(epoch=0, model_path='.', model=model,
                            optimizer=optimizer, scheduler=scheduler,
                            metric=0.7531)
        ckpt_files = []
        for file in _os.listdir("."):
            if file.endswith("0.7531.pt"):
                ckpt_files.append(file)
        # only one checkpoint file should be present.
        self.assertEqual(len(ckpt_files), 1)
        # recreate objects for load test
        model = _torchvision.models.alexnet()
        optimizer = _optim.Adam(model.parameters())
        scheduler = _optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
        # verify that default learning rate is not same as new_lr
        self.assertNotAlmostEqual(_tu.get_lr(optimizer), new_lr)
        start_epoch = _tu.load_checkpoint(model_path='.',
                                          ckpt_name=ckpt_files[0], model=model,
                                          optimizer=optimizer,
                                          scheduler=scheduler)
        # start_epoch should be previous epoch + 1
        self.assertEqual(start_epoch, 1)
        # validate the ckpt is loaded currectly using learning rate
        self.assertAlmostEqual(_tu.get_lr(optimizer), new_lr)
        _os.remove(ckpt_files[0])


if __name__ == '__main__':
    _unittest.main()
