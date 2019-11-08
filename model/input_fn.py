"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np


def input_fn(is_training, x, y_c, y_w, y_d, params):
    """Input function for the dataset.
    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        x: (numpy array) size: (m, n_x, 6), input S matrices
        y_c: (numpy array) size: (m, NC, 1), corresponding list of labels for background scattering
        y_w: (numpy array) size: (m, MMODE, 1): labels for the resonant frequency
        y_d: (numpy array) size: (m, 4, MODE): labels for the coupling coefficients
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = x.shape[0]
    assert x.shape[0] == y_w.shape[0], "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y_w), tf.constant(y_d)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y_w), tf.constant(y_d)))
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    smatrix, labels_w, labels_d = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'smatrix': smatrix, 'labels_w': labels_w, 'labels_d': labels_d, 'iterator_init_op': iterator_init_op}
    return inputs
