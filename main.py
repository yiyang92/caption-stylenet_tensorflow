import tensorflow as tf
import os
import numpy as np

from model.model import Decoder
from utils.parameters import parse_args
from utils.data import Data
from ops.inference import inference
from ops.optimizers import lstm_optimizer, masked_loss
# from utils.image_embeddings import vgg16

params = parse_args()
IMAGE_DIR = params['image_dir']
PICKLES_DIR = './pickles'


def main():
    # data
    print(params.keys())
    data = Data(IMAGE_DIR, PICKLES_DIR, params['keep_words'])
    data_dict = data.dictionary
    # define placeholders
    capt_inputs = tf.placeholder(tf.int32, [None, None])
    capt_labels = tf.placeholder(tf.int32, [None, None])
    seq_length = tf.placeholder(tf.int32, [None])
    image_embs = tf.placeholder(tf.float32, [None, 4096])  # vgg16
    model = Decoder(capt_inputs, params['lstm_hidden'],
                    params['embed_dim'], seq_length,
                    data_dict, params['lstm_hidden'], image_embs,
                    params=params)
    with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
        x_cmlogits, _ = model.forward(mode='train_capt')
        x_lmhlogits, _ = model.forward(mode='train_lmh', lm_label='humorous')
        x_lmrlogits, _ = model.forward(mode='train_lmr', lm_label='romantic')
    # losses
    labels_flat = tf.reshape(capt_labels, [-1])
    cm_loss = masked_loss(labels_flat, x_cmlogits, mode='train_capt')
    lmh_loss = masked_loss(labels_flat, x_lmhlogits, mode='train_lmh')
    lmr_loss = masked_loss(labels_flat, x_lmrlogits, mode='train_lmr')
    # optimizers
    cm_opt = lstm_optimizer(cm_loss, params, mode='train_capt')
    lmh_opt = lstm_optimizer(lmh_loss, params, mode='train_lmh')
    lmr_opt = lstm_optimizer(lmr_loss, params, mode='train_lmr')
    # train
    saver = tf.train.Saver(tf.trainable_variables(),
                           max_to_keep=params['keep_cp'])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # print(tf.trainable_variables())
        # train 3 networks, save S_act, S_hum, S_rom
        if params['mode'] == 'training':
            for label in ('actual', 'humorous', 'romantic'):
                for e in range(params['epochs_{}'.format(label)]):
                    for captions, lengths, image_f in data.get_batch(
                        params['batch_size'], label=label, set='train'):
                        feed = {capt_inputs: captions[0],
                                capt_labels: captions[1],
                                image_embs: image_f,
                                seq_length: lengths}
                        if label == 'actual':
                            opt_loss, optim = cm_loss, cm_opt
                        elif label == 'humorous':
                            opt_loss, optim = lmh_loss, lmh_opt
                        elif label == 'romantic':
                            opt_loss, optim = lmr_loss, lmr_opt
                        loss_, _ = sess.run([opt_loss, optim], feed)
                    if e % 4 == 0:
                        losses = []
                        for captions, lengths, image_f in data.get_batch(
                            params['batch_size'], label=label, set='val'):
                            feed = {capt_inputs: captions[0],
                                    capt_labels: captions[1],
                                    image_embs: image_f,
                                    seq_length: lengths}
                            if label == 'actual':
                                opt_loss, optim = cm_loss, cm_opt
                            elif label == 'humorous':
                                opt_loss, optim = lmh_loss, lmh_opt
                            elif label == 'romantic':
                                opt_loss, optim = lmr_loss, lmr_opt
                            vloss_ = sess.run([opt_loss], feed)
                            losses.append(vloss_)
                        print("{} Validation Model: Epoch: {} Loss: {}".format(
                            label, e, np.mean(losses)))
                    print("{} Model: Epoch: {} Loss: {}".format(label,
                                                                e, loss_))
            # save model
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
            save_path = saver.save(sess, "./checkpoints/{}.ckpt".format(
                params['checkpoint']))
            print("Model saved in file: %s" % save_path)
        elif params['mode'] == 'inference':
            inference(params, model, data, saver, sess)


if __name__ == '__main__':
    main()
