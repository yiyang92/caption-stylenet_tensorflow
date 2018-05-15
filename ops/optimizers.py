import tensorflow as tf


def masked_loss(labels_flat, logits, mode):
    """
    Arguments:
        mode: what model to train
    """
    ce_loss_padded = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_flat)
    loss_mask = tf.sign(tf.to_float(labels_flat))
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(ce_loss_padded, loss_mask)),
                        tf.reduce_sum(loss_mask),
                        name="batch_loss".format(mode))
    return batch_loss


def lstm_optimizer(loss, params, mode='capt', num_ex_per_epoch=None):
    gradients = tf.gradients(loss, tf.trainable_variables())
    # clipped_grad = clip_by_value(gradients, -0.1, 0.1)
    clip_norm = params['lstm_clip_norm']
    clipped_grad, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
    grads_vars = zip(clipped_grad, tf.trainable_variables())
    # learning rate decay
    learning_rate = tf.constant(params['learning_rate'])
    global_step = tf.Variable(initial_value=0, name="global_step",
                              trainable=False,
                              collections=[tf.GraphKeys.GLOBAL_STEP,
                                           tf.GraphKeys.GLOBAL_VARIABLES])
    if num_ex_per_epoch is not None:
        num_batches_per_epoch = num_ex_per_epoch / (
            params['batch_size'] + 0.00001)
        decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)
        learning_rate_decay = tf.train.exponential_decay(
            learning_rate, global_step, decay_steps=decay_steps,
            decay_rate=0.5, staircase=True)
    else:
        if params['optimizer'] in ('SGD', 'Momentum'):
            raise ValueError("Must specify num_ex_per_epoch if use SGD/Mom")
    # lstm parameters update
    if params['optimizer'] == 'SGD':
        optimize = tf.train.GradientDescentOptimizer(
            learning_rate_decay).apply_gradients(grads_vars,
                                                 global_step=global_step)
    if params['optimizer'] == 'Adam':
        optimize = tf.train.AdamOptimizer(
            params['learning_rate']).apply_gradients(
                grads_vars, global_step=global_step)
    elif params['optimizer'] == 'Momentum':
        momentum = 0.99
        optimize = tf.train.MomentumOptimizer(learning_rate_decay,
                                              momentum).apply_gradients(
                                                  grads_vars,
                                                  global_step=global_step)
    return optimize, global_step, global_norm
# fine-tuning CNN
# def cnn_optimizer(loss, params):
#     cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'cnn')
#     gradients = tf.gradients(loss, cnn_vars)
#     grads_vars = zip(gradients, cnn_vars)
#     # learning rate decay
#     learning_rate = tf.constant(params.cnn_lr)
#     global_step = tf.Variable(initial_value=0, name="global_step",
#                               trainable=False,
#                               collections=[tf.GraphKeys.GLOBAL_STEP,
#                                            tf.GraphKeys.GLOBAL_VARIABLES])
#     num_batches_per_epoch = params.num_ex_per_epoch / (
#         params.batch_size + 0.001)
#     decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)
#     learning_rate_decay = tf.train.exponential_decay(learning_rate,
#                                                global_step,
#                                                decay_steps=decay_steps,
#                                                decay_rate=0.5,
#                                                staircase=True)
#     # lstm parameters update
#     if params.cnn_optimizer == 'SGD':
#         optimize = tf.train.GradientDescentOptimizer(
#             learning_rate_decay).apply_gradients(grads_vars,
#                                                  global_step=global_step)
#     elif params.cnn_optimizer == 'Adam':
#         optimize = tf.train.AdamOptimizer(
#             params.cnn_lr, beta1=0.8).apply_gradients(grads_vars,
#                                                   global_step=global_step)
#     elif params.cnn_optimizer == 'Momentum':
#         momentum = 0.90
#         optimize = tf.train.MomentumOptimizer(learning_rate_decay,
#                                               momentum).apply_gradients(
#                                                   grads_vars,
#                                                   global_step=global_step)
#     return optimize, global_step
