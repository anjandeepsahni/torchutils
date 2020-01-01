import random as _random
import unittest as _unittest

import numpy as _np
import torch as _torch

import torchutils as _tu


class _TestUtils(_unittest.TestCase):

    def test_set_random_seed(self):
        # Set new seed and verify.
        seed = _random.randint(1, 1000)
        _tu.set_random_seed(seed)
        np_new_seed = _np.random.get_state()[1][0]
        torch_new_seed = _torch.initial_seed()
        self.assertEqual(seed, np_new_seed)
        self.assertEqual(seed, torch_new_seed)
        if _torch.cuda.is_available():
            cuda_new_seed = _torch.cuda.initial_seed()
            self.assertEqual(seed, cuda_new_seed)


if __name__ == '__main__':
    _unittest.main()
