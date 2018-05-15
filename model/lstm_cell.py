# main lstm network
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base as base_layer


class FactoredLSTMCell(LayerRNNCell):
    """Factored LSTM recurrent network cell.
  The implementation is based on: http://ieeexplore.ieee.org/document/8099591/.
    """

    def __init__(self,
                 num_units,
                 s,
                 u=None,
                 v=None,
                 activation=None,
                 forget_bias=1.0,
                 reuse=None,
                 name=None,
                 dtype=None):
        super(FactoredLSTMCell, self).__init__(_reuse=reuse, name=name,
                                               dtype=dtype)
        self._input_seq = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._activation = activation or tf.tanh
        self._forget_bias = forget_bias
        self._s = s  # variable shape [fact_e, fact_e]
        self._u = u  # if not None, variable shape [embed_size, fact_e]
        self._v = v  # if not None, variable shape [fact_e, 4 * n_units]

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        h_depth = self._num_units
        # calculate wx = u * s * v
        self._wx = tf.matmul(self._u, self._s)
        self._wx = tf.matmul(self._wx, self._v)  # [embed_dims, 4 * num_units]
        self._wh = self.add_variable('wh', [h_depth, 4 * h_depth])
        # get w (kernel) matrix
        self._kernel = tf.concat([self._wx, self._wh], 0)
        self._bias = self.add_variable(
            'bias', shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, num_units]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size, 2 * num_units]`.
        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        one = tf.constant(1, dtype=tf.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = state

        gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input(c~), f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = tf.add
        multiply = tf.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
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
