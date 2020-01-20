History
=======

0.0.3 (2020-01-21)
---------------------

New features:

* FLOPs support for softmax, embedding, sigmoid, tanh, lstm, rnn, gru.
* Calculate and track running statistics of data.
* Get statistics of dataset (mean, standard deviation and variance).
* Calculate and track accuracy of predictions.
* Calculate and track hamming loss of predictions.
* Track and maintain running average of loss.
* Plot (average) gradients for each layer in model.

Bug fixes:

* Multi-input model support for model summary and FLOPs.

Deprecations:

* get_current_lr() is deprecated, use get_lr() instead.
* set_current_lr() is deprecated, use set_lr() instead.



0.0.2 (2019-08-24)
---------------------

New features:

* Model summary.
* Model FLOPs.
* Model 'trainable' parameter count.

Bug fixes:

* Model parameter count.


0.0.1 (2019-07-19)
---------------------

* Initial package
