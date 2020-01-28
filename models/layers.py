import numpy as np
import tensorflow as tf
from helpers import utils, tf_helpers

class ConstrainedConv2D(tf.keras.layers.Layer):

    def __init__(self, filter_strength=100):
        super(ConstrainedConv2D, self).__init__()
        self.filter_strength = filter_strength

    def build(self, input_shape):
        f = np.array([[0, 0, 0, 0, 0], [0, -1, -2, -1, 0], [0, -2, 12, -2, 0], [0, -1, -2, -1, 0], [0, 0, 0, 0, 0]])        
        rf = utils.repeat_2dfilter(f, 3)        
        self.kernel = self.add_weight("kernel", shape=(5, 5, 3, 3), initializer=tf.constant_initializer(rf))

    def call(self, input):
        # Mask for normalizing the residual filter                
        tf_ind = tf.constant(utils.center_mask_2dfilter(5, 3), dtype=tf.float32)

        # Normalize the residual filter
        nf = self.kernel * (1 - tf_ind)
        df = tf.tile(tf.reshape(tf.reduce_sum(nf, axis=(0,1,2)), [1, 1, 1, 3]), [5, 5, 3, 1])
        nf = self.filter_strength * nf / df
        nf = nf - self.filter_strength * tf_ind

        # Convolution with the residual filter
        xp = tf.pad(input, [[0, 0], [2, 2], [2, 2], [0, 0]], 'SYMMETRIC')
        return tf.nn.conv2d(xp, nf, [1, 1, 1, 1], 'VALID')


class Quantization(tf.keras.layers.Layer):

    def __init__(self, rounding='soft', v=50, gamma=25, latent_bpf=4, trainable=False):
        super(Quantization, self).__init__()

        if rounding not in {'round', 'sin', 'soft', 'identity', 'soft-codebook'}:
            raise ValueError('Unsupported quantization: {}'.format(rounding))

        self.rounding = rounding
        self.approx_steps = 2
        self.v = v
        self.gamma = gamma
        self.latent_bpf = latent_bpf
        self.trainable = trainable

        qmin = -2 ** (self.latent_bpf - 1) + 1
        qmax = 2 ** (self.latent_bpf - 1)
                            
        if self.trainable:
            self.codebook = self.add_weight(initializer=tf.constant_initializer(np.arange(qmin, qmax + 1)), shape=(1, 2 ** self.latent_bpf), dtype=tf.float32)
        else:
            self.codebook = tf.constant(np.arange(qmin, qmax + 1), shape=(1, 2 ** self.latent_bpf), dtype=tf.float32)

    # def build(self, input_shape):
    #     # Initialize the quantization codebook

    def call(self, x):

        if self.rounding == 'round':
            x = tf.round(x)

        elif self.rounding == 'sin':
            x = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi))

        elif self.rounding == 'soft':
            x_ = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi))
            x = tf.add(tf.stop_gradient(tf.round(x) - x_), x_)

        elif self.rounding == 'harmonic':
            xa = x - tf.sin(2 * np.pi * x) / np.pi
            for k in range(2, self.approx_steps):
                xa += tf.pow(-1.0, k) * tf.sin(2 * np.pi * k * x) / (k * np.pi)
            x = xa

        elif self.rounding == 'identity':
            x = x

        elif self.rounding == 'soft-codebook':

            prec_dtype = tf.float64
            eps = 1e-72

            assert(self.codebook.shape[0] == 1)
            assert(self.codebook.shape[1] > 1)

            values = tf.reshape(x, (-1, 1))

            if self.v <= 0:
                # Gaussian soft quantization
                weights = tf.exp(-self.gamma * tf.pow(tf.cast(values, dtype=prec_dtype) - tf.cast(self.codebook, dtype=prec_dtype), 2))
            else:
                # t-Student soft quantization
                dff = tf.cast(values, dtype=prec_dtype) - tf.cast(self.codebook, dtype=prec_dtype)
                dff = self.gamma * dff
                weights = tf.pow((1 + tf.pow(dff, 2)/self.v), -(self.v+1)/2)

            weights = (weights + eps) / (tf.reduce_sum(weights + eps, axis=1, keepdims=True))

            assert(weights.shape[1] == np.prod(self.codebook.shape))

            soft = tf.reduce_mean(tf.matmul(weights, tf.transpose(tf.cast(self.codebook, dtype=prec_dtype))), axis=1)
            soft = tf.cast(soft, dtype=tf.float32)
            soft = tf.reshape(soft, tf.shape(x))

            hard = tf.gather(self.codebook, tf.argmax(weights, axis=1), axis=1)
            hard = tf.reshape(hard, tf.shape(x))

            x = tf.stop_gradient(hard - soft) + soft
            x = tf.identity(x)

        return x


class DiscreteLatent(tf.keras.layers.Layer):

    def __init__(self, rounding='soft', v=50, gamma=25, latent_bpf=4, trainable_codebook=False, trainable_scale=True):
        super(DiscreteLatent, self).__init__()
        self.trainable_scale = trainable_scale
        self.rounding = rounding
        self.v = v
        self.gamma = gamma
        self.latent_bpf = latent_bpf
        self.trainable_codebook = trainable_codebook
        if self.trainable_scale:
            self.scaling_factor = self.add_weight(shape=(), dtype=tf.float32, initializer=tf.constant_initializer(1), name='latent_scaling')
        self.quantization = Quantization(self.rounding, self.v, self.gamma, self.latent_bpf, self.trainable_codebook)

    def call(self, inputs):
        """
        Set up quantization of the latent space. The following attributes will be used (see constructor for details):
        - self.use_gdn
        - self.use_batchnorm
        - self.scale_latent
        - self._codebook
        - self._h.rounding

        The following new attributes will be set:
        - self.latent_pre (original real-values)
        - self.latent_post (quantized)

        :param net: the real-valued latent tensor
        :return: the quantized latent tensor
        """
        # If requested, add batch norm to normalize the latent representation
        latent = inputs

        # Learn a scaling factor for the latent features to encourage greater values (facilitates quantization)
        if self.trainable_scale:
            latent = latent * self.scaling_factor

        # Quantize the latent representation and remember tensors before and after the process
        entropy_ = tf_helpers.entropy(latent, self.quantization.codebook, self.v, self.gamma)[0]

        # self.latent_pre = latent
        latent = self.quantization(latent)
        # self.latent_post = latent

        return latent, entropy_

