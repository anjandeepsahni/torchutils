==========
TorchUtils
==========

.. image:: https://badge.fury.io/py/torchutils.svg
    :target: https://badge.fury.io/py/torchutils

.. image:: https://travis-ci.org/anjandeepsahni/pytorch_utils.svg?branch=master
    :target: https://travis-ci.org/anjandeepsahni/pytorch_utils

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

    import torchvision
    import torchutils as tu

    model = torchvision.models.alexnet()
    tu.get_model_summary(model, (3, 224, 224))

    # Output

    -----------------------------------------------------------------------
              Layer                  Output          Params           FLOPs
    =======================================================================
           Conv2d-1        [-1, 64, 55, 55]          23,296      70,470,400
             ReLU-2        [-1, 64, 55, 55]               0               0
        MaxPool2d-3        [-1, 64, 27, 27]               0               0
           Conv2d-4       [-1, 192, 27, 27]         307,392     224,088,768
             ReLU-5       [-1, 192, 27, 27]               0               0
        MaxPool2d-6       [-1, 192, 13, 13]               0               0
           Conv2d-7       [-1, 384, 13, 13]         663,936     112,205,184
             ReLU-8       [-1, 384, 13, 13]               0               0
           Conv2d-9       [-1, 256, 13, 13]         884,992     149,563,648
            ReLU-10       [-1, 256, 13, 13]               0               0
          Conv2d-11       [-1, 256, 13, 13]         590,080      99,723,520
            ReLU-12       [-1, 256, 13, 13]               0               0
       MaxPool2d-13         [-1, 256, 6, 6]               0               0
         Dropout-14              [-1, 9216]               0               0
          Linear-15              [-1, 4096]      37,752,832      75,493,376
            ReLU-16              [-1, 4096]               0               0
         Dropout-17              [-1, 4096]               0               0
          Linear-18              [-1, 4096]      16,781,312      33,550,336
            ReLU-19              [-1, 4096]               0               0
          Linear-20              [-1, 1000]       4,097,000       8,191,000
    =======================================================================
    Total params: 61,100,840
    Trainable params: 61,100,840
    Non-trainable params: 0
    Total FLOPs: 773,286,232 / 0.77 GFLOPs
    -----------------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 8.31
    Params size (MB): 233.08
    Estimated Total Size (MB): 241.96
    -----------------------------------------------------------------------

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
API documentation is available at: https://anjandeepsahni.github.io/pytorch_utils/

License
-------
TorchUtils is distributed under the MIT license, see LICENSE.
