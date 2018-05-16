import os
from glob import glob
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm

from random import shuffle
from utils.captions import Dictionary
from utils.image_utils import load_image
from utils.image_embeddings import vgg16


class Data():
    def __init__(self, images_dir, pickles_dir='./pickles',
                 keep_words=3, n_classes=2, params=None):
        self.images_dir = images_dir
        self.pickles_dir = pickles_dir
        # labelled (+ unlabelled)
        self.train_captions = self._load_captions('captions_ltr.pkl')
        self.val_captions = self._load_captions('captions_val.pkl')
        self.test_captions = self._load_captions('captions_test.pkl')
        print("Train data: ", len(self.train_captions.keys()))
        self.dictionary = Dictionary(self.train_captions, keep_words)
        self.weights_path = './utils/vgg16_weights.npz'
        self.im_features = self._extract_features_from_dir()
        # number of classes
        # self.n_classes = n_classes
        self._params = params

    def _load_captions(self, f_name):
        with open(os.path.join(self.pickles_dir, f_name), 'rb') as rf:
            return pickle.load(rf)

    def get_batch(self, batch_size, set='train', im_features=True,
                  get_names=False, label=None):
        """Get batch."""
        # if select inly one caption
        imn_batch = [None] * batch_size
        if set == 'train':
            self._iterable = self.train_captions.copy()
        elif set == 'val':
            self._iterable = self.val_captions.copy()
        else:
            self._iterable = self.test_captions.copy()
        im_names = list(self._iterable.keys())
        shuffle(im_names)
        mult_captions = True if label == 'actual' else False
        for i, item in enumerate(im_names):
            inx = i % batch_size
            imn_batch[inx] = item
            if inx == batch_size - 1:
                # images or features
                images = self._get_images(imn_batch, im_features)
                captions, lengths = self._form_captions_batch(
                    imn_batch, self._iterable, label, mult_captions)
                ret = (captions, lengths, images)
                if get_names:
                    ret += (imn_batch,)
                yield ret
                imn_batch = [None] * batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            images = self._get_images(imn_batch, im_features)
            captions, lengths = self._form_captions_batch(imn_batch,
                                                          self._iterable,
                                                          label,
                                                          mult_captions)
            ret = (captions, lengths, images)
            if get_names:
                ret += (imn_batch,)
            yield ret

    def _form_captions_batch(self, imn_batch, captions, label, mult_captions):
        # randomly choose 2 captions, labelled, one unlabelled
        if mult_captions and label != 'actual':
            print("romantic and humorous captions are unique")
            mult_captions = False
        labelled = []
        lengths = np.zeros((len(imn_batch)))
        if mult_captions:
            lengths = np.zeros((len(imn_batch) * self._params['num_captions']))
        labels = {'actual': 0, 'humorous': 1, 'romantic': 2}
        label = labels[label]
        for i, imn in enumerate(imn_batch):
            cap_dict = captions[imn]
            hum_c, rom_c = cap_dict['humorous'], cap_dict['romantic']
            act_c = cap_dict['actual']
            hum_c = self.dictionary.index_caption(hum_c[0])
            rom_c = self.dictionary.index_caption(rom_c[0])
            # randomly choose one of the 5 actual captions
            if mult_captions and self._params['num_captions'] > 1:
                ctr = i
                for j in range(self._params['num_captions']):
                    labelled.append(self.dictionary.index_caption(act_c[j]))
                    lengths[ctr] = len(act_c[j]) - 1
                    ctr += 1
            else:
                rand_cap = np.random.randint(low=0, high=len(act_c))
                act_c = self.dictionary.index_caption(act_c[rand_cap])
                # important, label-label_index correspondance
                cap_list = [act_c, hum_c, rom_c]
                # what will be labelled, what unlabelled
                labelled.append(cap_list[label])
                lengths[i] = len(labelled[i]) - 1
        pad_l = len(max(labelled, key=len))
        captions_inp = np.array([cap[:-1] + [0] * (
            pad_l - len(cap)) for cap in labelled])
        captions_lbl = np.array([cap[1:] + [0] * (
            pad_l - len(cap)) for cap in labelled])
        captions = (captions_inp, captions_lbl)
        return captions, lengths

    def _get_images(self, imn_batch, im_features=True):
        images = []
        if im_features:
            for name in imn_batch:
                images.append(self.im_features[name])
        else:
            for name in imn_batch:
                img = load_image(os.path.join(self.images_dir, name))
                images.append(img)
        return np.stack(np.squeeze(images))

    def _extract_features_from_dir(self, save_pickle=True, im_shape=(224,
                                                                     224)):
        """
        Args:
            data_dir: image data directory
            save_pickle: bool, will serialize feature_dict and save it into
        ./pickle directory
            im_shape: desired images shape
        Returns:
            feature_dict: dictionary of the form {image_name: feature_vector}
        """
        feature_dict = {}
        data_dir = self.images_dir
        try:
            with open(
                "./pickles/" + data_dir.split('/')[-2] + '.pickle', 'rb') as rf:
                print("Loading prepared feature vector from {}".format(
                    "./pickles/" + data_dir.split('/')[-2] + '.pickle'))
                feature_dict = pickle.load(rf)
        except:
            print("Extracting features")
            if not os.path.exists("./pickles"):
                os.makedirs("./pickles")
            im_embed = tf.Graph()
            with im_embed.as_default():
                input_img = tf.placeholder(tf.float32, [None,
                                                        im_shape[0],
                                                        im_shape[1], 3])
                image_embeddings = vgg16(input_img)
                features = image_embeddings.fc2
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            with tf.Session(graph=im_embed) as sess:
                if len(list(glob(data_dir + '*.jpg'))) == 0:
                    raise FileNotFoundError()
                print("loading imagenet weights")
                image_embeddings.load_weights(self.weights_path, sess)
                for img_path in tqdm(glob(data_dir + '*.jpg')):
                    img = load_image(img_path)
                    img = np.expand_dims(img, axis=0)
                    f_vector = sess.run(features, {input_img: img})
                    # ex. COCO_val2014_0000000XXXXX.jpg
                    feature_dict[img_path.split('/')[-1]] = f_vector
            if save_pickle:
                with open(
                    "./pickles/" + data_dir.split('/')[-2] + '.pickle', 'wb') as wf:
                    pickle.dump(feature_dict, wf)
        return feature_dict
