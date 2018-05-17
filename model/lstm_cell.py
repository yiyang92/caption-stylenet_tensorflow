# main lstm network
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.layers import base as base_layer


class FactoredLSTMCell():
    """Factored LSTM recurrent network cell.
  The implementation is based on: http://ieeexplore.ieee.org/document/8099591/.
    """

    def __init__(self,
                 num_units,
                 s,
                 u,
                 v,
                 wh=None,
                 bias=None,
                 activation=None,
                 forget_bias=1.0,
                 reuse=None,
                 name=None,
                 dtype=None):
        # super(FactoredLSTMCell, self).__init__(_reuse=reuse, name=name,
        #                                        dtype=dtype)
        self._input_seq = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._activation = activation or tf.tanh
        self._forget_bias = forget_bias
        self._s = s  # variable shape [fact_e, fact_e]
        self._u = u  # if not None, variable shape [embed_size, fact_e]
        self._v = v  # if not None, variable shape [fact_e, 4 * n_units]
        self._wh = wh
        self._bias = bias
        # other parameters
        self.built = False

    def __call__(self, inputs, state):
        if not self.built:
            self.build(tf.shape(inputs))
        return self.call(inputs, state)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype)
        h = tf.zeros([batch_size, self._num_units], dtype)
        return LSTMStateTuple(c, h)

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1] is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
        # self._bias = tf.get_variable(
        #     'bias', shape=[4 * self._num_units],
        #     initializer=tf.zeros_initializer(dtype=tf.float32))
        one = tf.constant(1, dtype=tf.int32)

        self.ui, self.uf, self.uo, self.uc = tf.split(
            value=self._u, num_or_size_splits=4, axis=one)

        self.vi, self.vf, self.vo, self.vc = tf.split(
            value=self._v, num_or_size_splits=4, axis=one)

        self.si, self.sf, self.so, self.sc = tf.split(
            value=self._s, num_or_size_splits=4, axis=one)

        self.whi, self.whf, self.who, self.whc = tf.split(
            value=self._wh, num_or_size_splits=4, axis=one)

        self.built = True

    def call(self, inputs, state):
        """Factorized Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, num_units]`
        Returns:
          A pair containing the new hidden state, and the new state (a
            `LSTMStateTuple`).
        """
        sigmoid = tf.sigmoid
        add = tf.add
        multiply = tf.multiply
        matmul = tf.matmul
        # previous state
        c, h = state
        # bias
        bi, bf, bo, bc = tf.split(
            value=self._bias, num_or_size_splits=4, axis=0)
        # wi, wf, wo, wc = tf.split(
        #     value=self._kernel, num_or_size_splits=4, axis=1)
        # input gate
        wi = matmul(self.ui, self.si)
        wi = matmul(wi, self.vi)
        i = add(matmul(inputs, wi), matmul(h, self.whi))
        i = tf.nn.bias_add(i, bi)
        # forget gate
        wf = matmul(self.uf, self.sf)
        wf = matmul(wf, self.vf)
        f = add(matmul(inputs, wf), matmul(h, self.whf))
        f = tf.nn.bias_add(f, bf)
        # output gate
        wo = matmul(self.uo, self.so)
        wo = matmul(wo, self.vo)
        o = add(matmul(inputs, wo), matmul(h, self.who))
        o = tf.nn.bias_add(o, bo)
        # ~c
        wc = matmul(self.uc, self.sc)
        wc = matmul(wc, self.vc)
        _c = add(matmul(inputs, wc), matmul(h, self.whc))
        _c = tf.nn.bias_add(_c, bc)
        # forget bias
        forget_bias_tensor = tf.constant(self._forget_bias, dtype=tf.float32)
        # c
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(_c)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
        # i = input_gate, j = new_input(c~), f = forget_gate, o = output_gate
        # i, j, f, o = tf.split(
        #     value=gate_inputs, num_or_size_splits=4, axis=one)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state


def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    # if using GRU Cells
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)
