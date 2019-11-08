"""Define the model."""

import tensorflow as tf
from const_params import *


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    smatrix = inputs['smatrix']

    assert smatrix.get_shape().as_list() == [None, DATA_SIZE, DATA_CHANNEL]

    out = smatrix
    # Define the number of channels of each convolution
    # For each block, we do: 3 conv1D -> batch norm -> relu -> 2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv1d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling1d(out, 2, 2)

    assert out.get_shape().as_list() == [None, 64, num_channels]

    # Split the placeholder to two parts: omega (w) and d (d)
    # omega part:
    with tf.variable_scope('block_w_1'):
        out_w = tf.layers.conv1d(out, MMODE, 3, padding='same')
        if params.use_batch_norm:
            out_w = tf.layers.batch_normalization(out_w, momentum=bn_momentum, training=is_training)
        out_w = tf.nn.relu(out_w)
        out_w = tf.layers.max_pooling1d(out_w, 2, 2)

    out_w = tf.reshape(out_w, [-1, 32 * MMODE])
    with tf.variable_scope('fc_w_1'):
        out_w = tf.layers.dense(out_w, 8 * MMODE)
        if params.use_batch_norm:
            out_w = tf.layers.batch_normalization(out_w, momentum=bn_momentum, training=is_training)
        out_w = tf.nn.relu(out_w)
    with tf.variable_scope('fc_w_2'):
        logits_w = tf.layers.dense(out_w, MMODE)
    logits_w = tf.reshape(logits_w, [-1, MMODE, 1])

    # d part:
    out_d = out
    num_channels_d = 8 * MMODE
    channels_d = [num_channels_d, num_channels_d, num_channels_d]
    for i, c in enumerate(channels_d):
        with tf.variable_scope('block_d_{}'.format(i + 1)):
            out_d = tf.layers.conv1d(out_d, c, 3, padding='same')
            if params.use_batch_norm:
                out_d = tf.layers.batch_normalization(out_d, momentum=bn_momentum, training=is_training)
            out_d = tf.nn.relu(out_d)
            out_d = tf.layers.max_pooling1d(out_d, 2, 2)

    print("Shape of out_d:")
    print(out_d.get_shape().as_list())
    assert out_d.get_shape().as_list() == [None, 8, num_channels_d]
    # assert out.get_shape().as_list() == [None, 64, num_channels]

    out_d = tf.reshape(out_d, [-1, 8 * num_channels_d])
    with tf.variable_scope('fc_d_1'):
        out_d = tf.layers.dense(out_d, 4 * num_channels_d)
        if params.use_batch_norm:
            out_d = tf.layers.batch_normalization(out_d, momentum=bn_momentum, training=is_training)
        out_d = tf.nn.relu(out_d)
    with tf.variable_scope('fc_d_2'):
        logits_d = tf.layers.dense(out_d, 4 * MMODE)
    logits_d = tf.reshape(logits_d, [-1, 4, MMODE])

    return logits_w, logits_d


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels_w = inputs['labels_w']
    labels_w = tf.cast(labels_w, tf.float32)
    labels_d = inputs['labels_d']
    labels_d = tf.cast(labels_d, tf.float32)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits_w, logits_d = build_model(is_training, inputs, params)
        # predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss_w = tf.losses.mean_squared_error(labels=labels_w, predictions=logits_w)
    labels_d_square = tf.square(tf.norm(labels_d, axis=-2, keepdims=True))
    labels_d_square = tf.broadcast_to(input=labels_d_square, shape=tf.shape(labels_d))
    assert labels_d_square.get_shape().as_list() == labels_d.get_shape().as_list()
    print("Label d square after broadbast:")
    print(labels_d_square[0, :, :])
    loss_d = tf.losses.mean_squared_error(labels=labels_d, predictions=logits_d,
                                          weights=tf.divide(1.0, (labels_d_square + params.epsilon)))
    loss = tf.add(params.loss_w_ratio * loss_w, params.loss_d_ratio * loss_d)
    accuracy_w = tf.reduce_mean(tf.cast(tf.norm(labels_w-tf.cast(logits_w, tf.float32),
                                                axis=-1), tf.float32))
    accuracy_d = tf.reduce_mean(tf.cast(tf.norm(labels_d-tf.cast(logits_d, tf.float32),
                                                axis=[-2, -1]), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy_w': tf.metrics.mean(accuracy_w),
            'accuracy_d': tf.metrics.mean(accuracy_d),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy_w', accuracy_w)
    tf.summary.scalar('accuracy_d', accuracy_d)
    tf.summary.histogram('train_smatrix', inputs['smatrix'])

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions_w"] = logits_w
    model_spec["predictions_d"] = logits_d
    model_spec['loss'] = loss
    model_spec['accuracy_w'] = accuracy_w
    model_spec['accuracy_d'] = accuracy_d
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
