==========
TorchUtils
==========

.. image:: https://badge.fury.io/py/torchutils.svg
    :target: https://badge.fury.io/py/torchutils

.. image:: https://travis-ci.org/anjandeepsahni/torchutils.svg?branch=master
    :target: https://travis-ci.org/anjandeepsahni/torchutils

.. image:: https://codecov.io/gh/anjandeepsahni/torchutils/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/anjandeepsahni/torchutils

.. image:: https://img.shields.io/github/license/anjandeepsahni/torchutils
    :target: https://img.shields.io/github/license/anjandeepsahni/torchutils

.. image:: https://pepy.tech/badge/torchutils
    :target: https://pepy.tech/badge/torchutils

|

**TorchUtils** is a Python package providing helpful utility functions for your
PyTorch projects. TorchUtils helps speed up PyTorch development by implementing
trivial but useful functionality so that you don't have to.

Key Features
------------

* Calculate total model parameters.
* Calculate model FLOPs.
* Print model summary in Keras style.
* Save/load checkpoints.
* Get/set learning rate.
* Set random seed.

Examples
--------

Model Summary::

    import torch
    import torchvision
    import torchutils as tu

    model = torchvision.models.alexnet()
    tu.get_model_summary(model, torch.rand((1, 3, 224, 224)))

    # Output

    =========================================================================================
    Layer                           Kernel             Output          Params           FLOPs
    =========================================================================================
    0_features.Conv2d_0         [3, 64, 11, 11]    [1, 64, 55, 55]       23,296    70,470,400
    1_features.ReLU_1                         -    [1, 64, 55, 55]            0             0
    2_features.MaxPool2d_2                    -    [1, 64, 27, 27]            0             0
    3_features.Conv2d_3         [64, 192, 5, 5]   [1, 192, 27, 27]      307,392   224,088,768
    4_features.ReLU_4                         -   [1, 192, 27, 27]            0             0
    5_features.MaxPool2d_5                    -   [1, 192, 13, 13]            0             0
    6_features.Conv2d_6        [192, 384, 3, 3]   [1, 384, 13, 13]      663,936   112,205,184
    7_features.ReLU_7                         -   [1, 384, 13, 13]            0             0
    8_features.Conv2d_8        [384, 256, 3, 3]   [1, 256, 13, 13]      884,992   149,563,648
    9_features.ReLU_9                         -   [1, 256, 13, 13]            0             0
    10_features.Conv2d_10      [256, 256, 3, 3]   [1, 256, 13, 13]      590,080    99,723,520
    11_features.ReLU_11                       -   [1, 256, 13, 13]            0             0
    12_features.MaxPool2d_12                  -     [1, 256, 6, 6]            0             0
    13_classifier.Dropout_0                   -          [1, 9216]            0             0
    14_classifier.Linear_1         [9216, 4096]          [1, 4096]   37,752,832    75,493,376
    15_classifier.ReLU_2                      -          [1, 4096]            0             0
    16_classifier.Dropout_3                   -          [1, 4096]            0             0
    17_classifier.Linear_4         [4096, 4096]          [1, 4096]   16,781,312    33,550,336
    18_classifier.ReLU_5                      -          [1, 4096]            0             0
    19_classifier.Linear_6         [4096, 1000]          [1, 1000]    4,097,000     8,191,000
    =========================================================================================
    Total params: 61,100,840
    Trainable params: 61,100,840
    Non-trainable params: 0
    Total FLOPs: 773,286,232 / 773.29 MFLOPs
    -----------------------------------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 8.31
    Params size (MB): 233.08
    Estimated Total Size (MB): 241.96
    =========================================================================================

Learning Rate::

    import torchvision
    import torchutils as tu
    import torch.optim as optim

    model = torchvision.models.alexnet()
    optimizer = optim.Adam(model.parameters())
    current_lr = tu.get_lr(optimizer)
    print('Current learning rate:', current_lr)

    optimizer = tu.set_lr(optimizer, current_lr*0.1)
    revised_lr = tu.get_lr(optimizer)
    print('Revised learning rate:', revised_lr)

    # Output

    Current learning rate: 0.001
    Revised learning rate: 0.0001

Checkpoint::

    import torchvision
    import torchutils as tu
    import torch.optim as optim

    model = torchvision.models.alexnet()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.1)

    print('Original learning rate:', tu.get_lr(optimizer))

    # load checkpoint model_20190814-212442_e0_0.7531.pt
    start_epoch = tu.load_checkpoint(model_path='.',
                           ckpt_name='model_20190814-212442_e0_0.7531.pt',
                           model=model, optimizer=optimizer,
                           scheduler=scheduler)

    print('Checkpoint learning rate:', tu.get_lr(optimizer))
    print('Start from epoch:', start_epoch)

    # Output

    Original learning rate: 0.001
    Checkpoint learning rate: 0.1234
    Start epoch: 1

Requirements
------------

* Numpy >= 1.16.2
* PyTorch >= 1.0.0

Installation
------------

::

    $ pip install torchutils

Documentation
-------------
API documentation is available at: https://anjandeepsahni.github.io/torchutils/

License
-------
TorchUtils is distributed under the MIT license, see LICENSE.
