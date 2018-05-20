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


def lstm_optimizer(loss, params, learning_rate, mode,
                   num_ex_per_epoch=None):
    trainable_variables = []
    # a bit overkill, but let it be
    print(mode)
    for v in tf.trainable_variables():
        name_tokens = v.op.name.split('/')
        if mode == 'train_lmh' or mode == 'train_lmr':
            if name_tokens[1] == 'u_and_v':
                continue
            elif name_tokens[1] == 's_c':
                continue
            elif name_tokens[1] == 's_r' and mode == 'train_lmh':
                continue
            elif name_tokens[1] == 's_h' and mode == 'train_lmr':
                continue
            elif name_tokens[1] == 'emb':
                continue
            elif name_tokens[1] == 'imf_emb':
                continue
            elif name_tokens[1] == 'logits_train_capt':
                continue
            elif name_tokens[1] == 'logits_train_lmh' and mode == 'train_lmr':
                continue
            elif name_tokens[1] == 'logits_train_lmr' and mode == 'train_lmh':
                continue
        if mode == 'train_capt':
            if name_tokens[1] == 's_r' or name_tokens[1] == 's_h':
                continue
            if name_tokens[1] == 'logits_train_lmr' and mode == 'train_capt':
                continue
            if name_tokens[1] == 'logits_train_lmh' and mode == 'train_capt':
                continue
        trainable_variables.append(v)
    clip_norm = params['lstm_clip_norm']
    for v in trainable_variables:
        print(v.op.name.split('/'))
    gradients = tf.gradients(loss, trainable_variables)
    clipped_grad, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
    grads_vars = zip(clipped_grad, trainable_variables)
    # learning rate decay
    learning_rate = tf.constant(learning_rate)
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
        optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(
                grads_vars, global_step=global_step)
    elif params['optimizer'] == 'Momentum':
        momentum = 0.99
        optimize = tf.train.MomentumOptimizer(learning_rate_decay,
                                              momentum).apply_gradients(
                                                  grads_vars,
                                                  global_step=global_step)
    return optimize, global_step, global_norm
