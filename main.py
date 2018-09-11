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

# TODO: coco-evaluation change
# TODO: inspect train/test split, split according to carpathy split
# TODO: write script, which can launch coco-eval at every some period of time
# TODO: rewrite main file, carefully check the performance

def main():
    # data
    data = Data(IMAGE_DIR, PICKLES_DIR, params['keep_words'], params=params)
    data_dict = data.dictionary
    # define placeholders
    capt_inputs = tf.placeholder(tf.int32, [None, None])
    capt_labels = tf.placeholder(tf.int32, [None, None])
    seq_length = tf.placeholder(tf.int32, [None])
    image_embs = tf.placeholder(tf.float32, [None, 4096])  # vgg16
    # forward pass is expensive, so can use this method to reduce computation
    if params['num_captions'] > 1 and params['mode'] == 'training':
        features_tiled = tf.tile(tf.expand_dims(image_embs, 1),
                                 [1, params['num_captions'], 1])
        features_tiled = tf.reshape(features_tiled,
                                    [tf.shape(image_embs
                                              )[0] * params['num_captions'],
                                     4096])  # [5 * b_s, 4096]
    else:
        features_tiled = image_embs
    model = Decoder(capt_inputs, params['lstm_hidden'],
                    params['embed_dim'], seq_length,
                    data_dict, params['lstm_hidden'], image_embs,
                    params=params, reuse_text_emb=True)
    with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
        x_cmlogits, _ = model.forward(mode='train_capt',
                                      image_embs=features_tiled)
        x_lmrlogits, _ = model.forward(mode='train_lmr', lm_label='romantic')
        x_lmhlogits, _ = model.forward(mode='train_lmh', lm_label='humorous')

    # losses
    labels_flat = tf.reshape(capt_labels, [-1])
    cm_loss = masked_loss(labels_flat, x_cmlogits, mode='train_capt')
    lmh_loss = masked_loss(labels_flat, x_lmhlogits, mode='train_lmh')
    lmr_loss = masked_loss(labels_flat, x_lmrlogits, mode='train_lmr')
    # optimizers
    cm_opt = lstm_optimizer(cm_loss, params, params['learning_rate'],
                            mode='train_capt')
    lmh_opt = lstm_optimizer(lmh_loss, params, 0.0005,
                             mode='train_lmh')
    lmr_opt = lstm_optimizer(lmr_loss, params, 0.0005,
                             mode='train_lmr')
    # train
    saver = tf.train.Saver(tf.trainable_variables(),
                           max_to_keep=params['keep_cp'])
    config = tf.ConfigProto()
    # print(tf.trainable_variables())
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if params['write_summary']:
            summary_writer = tf.summary.FileWriter('./logs', sess.graph)
            summary_writer.add_graph(sess.graph)
        # print(tf.trainable_variables())
        # train 3 networks, save S_act, S_hum, S_rom
        if params['restore']:
            print("Restoring from checkpoint")
            saver.restore(sess,
                          "./checkpoints/{}.ckpt".format(params['checkpoint']))
        # choose labels for the training
        tr_labels = ['actual']
        tr_style = params['tr_style']
        if tr_style == 'both':
            tr_labels.extend(['humorous', 'romantic'])
        else:
            tr_labels.append(tr_style)
        if params['mode'] == 'training':
            for e in range(params['epochs']):
                for label in tr_labels:
                    if label == 'actual':  # folowing paper
                        batch_size = params['batch_size']
                    else:
                        batch_size = params['batch_size_lm']
                    for captions, lengths, image_f in data.get_batch(
                            batch_size, label=label, set='train'):
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
                        # save model
                        if not os.path.exists("./checkpoints"):
                            os.makedirs("./checkpoints")
                    if e % 10 == 0 and e != 0:  # save every 10 epochs
                        save_path = saver.save(sess,
                                               "./checkpoints/{}.ckpt".format(
                                                   params['checkpoint']))
                        print("Model saved in file: %s" % save_path)
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
