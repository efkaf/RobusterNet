import numpy as np
seed = np.random.seed(1337)
from keras import backend as K
from keras.layers import Layer
from keras.utils import conv_utils
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


class QuadraticLayer(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,

                 **kwargs):
        super(QuadraticLayer, self).__init__(**kwargs)

        self.rank = 2
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.data_format = K.common.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank,
                                                        'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_dim):
        self.volterra_kernel = self.add_weight(name='volterra_kernel',
                                               shape=(input_dim[-1], self.kernel_size[0] ** 2, self.kernel_size[1] ** 2,
                                                      self.filters),
                                               initializer=initializers.he_normal(seed=seed),
                                               trainable=True)

        self.linear_kernel = self.add_weight(name='linear_kernel',
                                             shape=(
                                             self.kernel_size[0], self.kernel_size[1], input_dim[-1], self.filters),
                                             initializer=initializers.he_normal(seed=seed),
                                             trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=initializers.Zeros(),
                                        name='bias', )
        else:
            self.bias = None

        super(QuadraticLayer, self).build(input_dim)

    def call(self, inputs):
        input_patches = K.tf.extract_image_patches(inputs, ksizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
                                                  strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding=self.padding)
        batch, out_row, out_col, sizes = input_patches.get_shape().as_list()

        input_patches = K.tf.reshape(input_patches, [-1, out_row, out_col, self.kernel_size[0] * self.kernel_size[1], inputs.shape[-1]])
        V = K.tf.einsum('abcid,abcjd,dijo->abcdo', input_patches, input_patches, self.volterra_kernel)
        S = K.tf.reduce_sum(V, 3)

        return S + K.tf.nn.conv2d(inputs, self.linear_kernel, strides=[1, 1, 1, 1], padding=self.padding) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1],input_shape[2], self.filters


