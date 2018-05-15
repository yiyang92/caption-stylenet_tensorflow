import json
from collections import defaultdict, Counter
import re

# building Vocabulary
class Dictionary(object):
    def __init__(self, captions, keep_words=3):
        """
        Args:
            captions: {file_name: {'caption_type': [['cap1'], ['cap2']]}}
            keep_words: keep every n words
        """
        # sentences - array of sentences
        self._captions = captions
        self._keep_words = keep_words
        self._word2idx = {}
        self._idx2word = {}
        self._words = []
        self._get_words()
        # add tokens
        self._words.append('<UNK>')
        self.build_vocabulary()

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def _get_words(self):
        for im_fn in self._captions:
            for cap_type in self._captions[im_fn]:
                for cap in self._captions[im_fn][cap_type]:
                    for word in cap:
                        word = word if word in ["<EOS>",
                                                "<BOS>",
                                                "<PAD>"] else word.lower()
                        self._words.append(word)

    def build_vocabulary(self):
        counter = Counter(self._words)
        # words, that occur less than 5 times dont include
        sorted_dict = sorted(counter.items(), key= lambda x: (-x[1], x[0]))
        # keep n words to be included in vocabulary
        sorted_dict = [(wd, count) for wd, count in sorted_dict
                       if count >= self._keep_words or wd == '<UNK>']
        # after sorting the dictionary, get ordered words
        words, _ = list(zip(*sorted_dict))
        self._word2idx = dict(zip(words, range(1, len(words) + 1)))
        self._idx2word = dict(zip(range(1, len(words) + 1), words))
        # add <PAD> as zero
        self._idx2word[0] = '<PAD>'
        self._word2idx['<PAD>'] = 0
        import pickle
        # save to ./pickles folder
        with open('./pickles/capt_vocab.pickle', 'wb') as wf:
            pickle.dump(file=wf, obj=self._captions)
        print("Vocabulary size: ", len(self._word2idx))

    def __len__(self):
        return len(self.idx2word)

    def index_caption(self, caption):
        '''
        Args:
            word2idx: word to indices mapping
        '''
        def add_index(word):
            try:
                index = self.word2idx[word]
            except KeyError as e:
                index = self.word2idx['<UNK>']
            return index
        # take caption and tokenize it
        # if word not in vocab, use <UNK>
        return [add_index(word) for word in caption]
