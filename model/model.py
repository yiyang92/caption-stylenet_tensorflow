import tensorflow as tf
import numpy as np
from model.lstm_cell import FactoredLSTMCell, rnn_placeholders
from utils.top_n import TopN, Beam


class Decoder():
    """LSTM Decoder model"""
    def __init__(self, capt_inputs, num_units, embed_size, seq_length,
                 data_dict, factored_dim=None, image_embs=None,
                 reuse_text_emb=True, gen_temperature=1.0, params=None):
        """"""
        self._capt_inputs = capt_inputs
        self._image_embs = image_embs
        self._num_units = num_units
        self._embed_size = embed_size
        self._factored_dim = factored_dim or self._num_units
        self._seq_length = seq_length
        self._data_dict = data_dict
        self._reuse_text_emb = reuse_text_emb
        # during generation
        self._gen_temperature = gen_temperature
        self._params = params

    def _scope_helper(self, scope_name, mode):
        # TODO: make a decorator
        if mode != 'gen':
            scope = (scope_name + '_{}').format(mode)
        else:
            scope = (scope_name + '_{}').format('train_capt')
        return scope

    def forward(self, mode='train_capt', lm_label=None,
                sample_gen=None, image_embs=None):
        """
        Arguments:
            mode: training mode (train_capt, train_lm)
            lm_label: if training language model, set a class (humorous,
            romantic)
            image_embs: optional image embeddings input
        """
        if image_embs is None:
            image_embs = self._image_embs
        emb_size = self._embed_size
        f_dim = self._factored_dim
        with tf.variable_scope('u_and_v', reuse=tf.AUTO_REUSE):
            h_depth = self._num_units
            # U matrix
            self._u = tf.get_variable('u', [emb_size, 4 * f_dim])
            # V matrix
            self._v = tf.get_variable('v', [f_dim, 4 * h_depth])
            # Wh matrix
            self._wh = tf.get_variable('wh', [h_depth, 4 * h_depth])
            # bias
            bias = tf.get_variable('rnn_bias', shape=[4 * self._num_units],
                                   initializer=tf.zeros_initializer(
                                       dtype=tf.float32))
        if mode == 'train_capt':
            # actual (caption model) S matrix
            self._sc = tf.get_variable('s_c', [f_dim, 4 * f_dim])
            s = self._sc
        elif mode == 'train_lmh':
            # humorous (language model) S matrix
            self._sh = tf.get_variable('s_h', [f_dim, 4 * f_dim])
            s = self._sh
        elif mode == 'train_lmr':
            # romantic (language model) S matrix
            self._sr = tf.get_variable('s_r', [f_dim, 4 * f_dim])
            s = self._sr
        elif mode == 'gen':
            if lm_label is None:
                raise ValueError("Need to supply label for caption generation")
            if lm_label == 'actual':
                s = tf.get_variable('s_c', [f_dim, 4 * f_dim])
            elif lm_label == 'humorous':
                s = tf.get_variable('s_h', [f_dim, 4 * f_dim])
            elif lm_label == 'romantic':
                s = tf.get_variable('s_r', [f_dim, 4 * f_dim])
        u = self._u
        v = self._v
        wh = self._wh
        rnn_scope = self._scope_helper('rnn_scope', mode)
        with tf.variable_scope(rnn_scope, reuse=tf.AUTO_REUSE):
            self._lstm = FactoredLSTMCell(self._num_units, s, u, v, wh, bias)
            init_state = self._lstm.zero_state(
                batch_size=tf.shape(image_embs)[0], dtype=tf.float32)
        # embeddings
        if self._reuse_text_emb:
            text_emb_scope = 'emb'
        else:
            text_emb_scope = self._scope_helper('emb', mode)
        with tf.variable_scope(text_emb_scope, reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable("text_embeddings",
                                        [self._data_dict.vocab_size,
                                         self._embed_size], dtype=tf.float32)
        vect_inputs = tf.nn.embedding_lookup(embedding, self._capt_inputs)
        # if not language model, input image embeddings
        if mode == 'train_capt' or mode == 'gen':
            # images_fv = tf.layers.dense(image_embs, self._embed_size,
            #                             name='imf_emb')
            # with tf.variable_scope(rnn_scope, reuse=tf.AUTO_REUSE):
            #     _, first_state = self._lstm(images_fv, init_state)
            c = h = tf.layers.dense(image_embs, self._num_units,
                                    name='imf_lstm_state')
            first_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        elif mode == 'train_lmh' or mode == 'train_lmr':
            first_state = init_state
        initial_state = rnn_placeholders(first_state)
        with tf.variable_scope(rnn_scope, reuse=tf.AUTO_REUSE):
            length = self._seq_length
            outputs, last_state = tf.nn.dynamic_rnn(self._lstm,
                                                    inputs=vect_inputs,
                                                    sequence_length=None,
                                                    initial_state=initial_state,
                                                    swap_memory=True,
                                                    dtype=tf.float32)
        if mode == 'gen':
            # only interested in the last output during generation
            outputs = outputs[:, -1, :]
        outputs_r = tf.reshape(outputs, [-1, self._lstm.output_size])
        # set logits scope
        rnn_logits_scope = self._scope_helper('logits', mode)
        with tf.variable_scope(rnn_logits_scope, reuse=tf.AUTO_REUSE):
            x_logits = tf.layers.dense(outputs_r,
                                       units=self._data_dict.vocab_size,
                                       name='rnn_logits')
        sample = None
        if mode == 'gen':
            if sample_gen == 'sample':
                sample = tf.multinomial(
                    x_logits / self._gen_temperature, 1)[0][0]
            elif sample_gen == 'beam_search':
                sample = tf.nn.softmax(x_logits)
            else:
                sample = tf.nn.softmax(x_logits)
        return x_logits, (initial_state, last_state, sample)

    def online_inference(self, sess, picture_ids, in_pictures, label,
                         stop_word='<EOS>', ground_truth=None):
        """Generate caption, given batch of pictures and their ids (names).
        Args:
            sess: tf.Session() object
            picture_ids: list of picture ids in shape [batch_size]
            in_pictures: input pictures
            stop_word: when stop caption generation
            label: generation caption label ('actual', 'humorous', 'romantic')
        Returns:
            cap_list: list of format [{'image_id', caption: ''}]
            cap_raw: list of generated caption indices
        """
        # get stop word index from dictionary
        stop_word_idx = self._data_dict.word2idx['<EOS>']
        cap_list = [None] * in_pictures.shape[0]
        # initialize caption generator
        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            _, states = self.forward(mode='gen', lm_label=label)
        init_state, out_state, sample = states
        cap_raw = []
        # get label names, if will be more labels can be loaded from pickle
        # labels_names = ['humorous', 'romantic', 'actual']
        for i in range(len(in_pictures)):
            state = None
            if ground_truth is not None:
                b_index = self._data_dict.word2idx['<BOS>']
                e_index = self._data_dict.word2idx['<EOS>']
                g_truth = ' '.join([self._data_dict.idx2word[word]
                                    for word in ground_truth[i]
                                    if word not in [b_index, e_index, 0]])
            cap_list[i] = {'image_id': int(picture_ids[i].split('.')[0]),
                           'caption': ' ',
                           'label': label}
            if ground_truth is not None:
                cap_list[i].update({'ground_truth': g_truth})
            sentence = ['<BOS>']
            cur_it = 0
            gen_word_idx = 0
            cap_raw.append([])
            gen_max = self._params['gen_max']
            while (cur_it < gen_max):
                input_seq = [self._data_dict.word2idx[
                    word] for word in sentence]
                feed = {self._capt_inputs: np.array(
                    input_seq)[-1].reshape([1, 1]),
                        self._seq_length: [len(input_seq)],
                        self._image_embs: np.expand_dims(in_pictures[i], 0)
                        }
                # for the first decoder step, the state is None
                if state is not None:
                    feed.update({init_state: state})
                next_word_probs, state = sess.run([sample, out_state], feed)
                if self._params['sample_gen'] == 'greedy':
                    next_word_probs = next_word_probs.ravel()
                    t = self._params['temperature']
                    next_word_probs = next_word_probs**(
                        1/t) / np.sum(next_word_probs**(1/t))
                    gen_word_idx = np.argmax(next_word_probs)
                elif self._params['sample_gen'] == 'sample':
                    gen_word_idx = next_word_probs
                gen_word = self._data_dict.idx2word[gen_word_idx]
                sentence += [gen_word]
                cap_raw[i].append(gen_word_idx)
                cur_it += 1
                if gen_word_idx == stop_word_idx:
                    break
            cap_list[i]['caption'] = ' '.join([word for word in sentence
                                               if word not in ['<BOS>',
                                                               '<EOS>']])
            print(cap_list[i]['caption'] + ' ' + cap_list[i]['label'])
            print("Ground truth caption: ", cap_list[i]['ground_truth'])
        return cap_list, cap_raw

    def beam_search(self, sess, picture_ids, in_pictures, label,
                    stop_word='<EOS>', ground_truth=None, beam_size=2,
                    ret_beams=False, len_norm_f=0.7):
        """Generate captions using beam search algorithm
        Args:
            sess: tf.Session
            picture_ids: list of picture ids in shape [batch_size]
            in_pictures: input pictures
            beam_size: keep how many beam candidates
            ret_beams: whether to return several beam canditates
            image_f_inputs: image placeholder
            len_norm_f: beam search length normalization parameter
        Returns:
            cap_list: list of format [{'image_id', caption: ''}]
                or (if ret_beams)
            cap_list: list of format [[{'image_id', caption: '' * beam_size}]]
        """
        # get stop word index from dictionary
        start_word_idx = self._data_dict.word2idx['<BOS>']
        stop_word_idx = self._data_dict.word2idx['<EOS>']
        cap_list = [None] * in_pictures.shape[0]
        # initialize caption generator
        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            _, states = self.forward(mode='gen', lm_label=label)
        init_state, out_state, sample = states
        # get label names, if will be more labels can be loaded from pickle
        for im in range(len(in_pictures)):
            state = None
            if ground_truth is not None:
                b_index = self._data_dict.word2idx['<BOS>']
                e_index = self._data_dict.word2idx['<EOS>']
                g_truth = ' '.join([self._data_dict.idx2word[word]
                                    for word in ground_truth[im]
                                    if word not in [b_index, e_index, 0]])

            cap_list[im] = {'image_id': int(picture_ids[im].split('.')[0]),
                           'caption': ' ',
                           'label': label}
            if ground_truth is not None:
                cap_list[im].update({'ground_truth': g_truth})
            # initial feed
            seed = start_word_idx
            feed = {self._capt_inputs: np.array(seed).reshape([1, 1]),
                    self._seq_length: [1],
                    self._image_embs: np.expand_dims(in_pictures[im], 0)
                    }
            # probs are normalized probs
            probs, state = sess.run([sample, out_state], feed)
            # initial Beam, pushed to the heap (TopN class)
            # inspired by tf/models/im2txt
            initial_beam = Beam(sentence=[seed],
                                state=state,
                                logprob=0.0,
                                score=0.0)
            partial_captions = TopN(beam_size)
            partial_captions.push(initial_beam)
            complete_captions = TopN(beam_size)

            # continue to generate, until max_len
            gen_max = self._params['gen_max']
            for _ in range(gen_max - 1):
                partial_captions_list = partial_captions.extract()
                partial_captions.reset()
                # get last word in the sentence
                input_feed = [(c.sentence[-1],
                               len(c.sentence)) for c in partial_captions_list]
                state_feed = [c.state for c in partial_captions_list]
                # get states and probs for every beam
                probs_list, states_list = [], []
                for inp_length, state in zip(input_feed, state_feed):
                    inp, length = inp_length
                    feed = {self._capt_inputs: np.array(inp).reshape([1, 1]),
                            self._seq_length: [length],
                            self._image_embs: np.expand_dims(
                                in_pictures[im], 0),
                            init_state: state}
                    probs, new_state = sess.run([sample, out_state], feed)
                    probs_list.append(probs)
                    states_list.append(new_state)
                # for every beam get candidates and append to list
                for i, partial_caption in enumerate(partial_captions_list):
                    cur_state = states_list[i]
                    cur_probs = probs_list[i]
                    # sort list probs, enumerate to remember indices
                    w_probs = list(enumerate(cur_probs.ravel()))
                    w_probs.sort(key=lambda x: -x[1])
                    # keep n probs
                    w_probs = w_probs[:beam_size]
                    for w, p in w_probs:
                        if p < 1e-12:
                            continue  # Avoid log(0).
                        sentence = partial_caption.sentence + [w]
                        logprob = partial_caption.logprob + np.log(p)
                        score = logprob
                        # complete caption, got <EOS>
                        if w == stop_word_idx:
                            if len_norm_f > 0:
                                score /= len(sentence)**len_norm_f
                            beam = Beam(sentence, cur_state, logprob, score)
                            complete_captions.push(beam)
                        else:
                            beam = Beam(sentence, cur_state, logprob, score)
                            partial_captions.push(beam)
                if partial_captions.size() == 0:
                    # When all captions are complete
                    break
            # If we have no complete captions then fall back to the partial captions.
            # But never output a mixture of complete and partial captions because a
            # partial caption could have a higher score than all the complete captions.
            if not complete_captions.size():
                complete_captions = partial_captions
            # find the best beam
            beams = complete_captions.extract(sort=True)
            if not ret_beams:
                best_beam = beams[0]
                capt = [self._data_dict.idx2word[word] for
                                                   word in best_beam.sentence
                                                   if word not in [seed,
                                                                   stop_word_idx]]
                cap_list[im]['caption'] = ' '.join(capt)
            print(cap_list[im]['caption'] + ' ' + cap_list[im]['label'])
            print("Ground truth caption: ", cap_list[im]['ground_truth'])
            # return list of beam candidates
            if ret_beams:
                c_list = []
                for c in beams:
                    capt = [self._data_dict.idx2word[word] for
                                                       word in c.sentence
                                                       if word not in [seed,
                                                                       stop_word_idx]]
                    c_list.append(' '.join(capt))
                cap_list[im]['caption'] = c_list
        return cap_list
