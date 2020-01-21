import numpy as np
import tensorflow as tf

from compression import jpeg_helpers
from helpers.utils import jpeg_qtable, is_number
from helpers import tf_helpers


class DifferentiableJPEG(tf.keras.Model):

    def __init__(self, quality=None, rounding_approximation='sin', rounding_approximation_steps=5, trainable=False):
        super().__init__(self)

        if not (is_number(quality) or (isinstance(quality, tuple) and len(quality) == 2)):
            raise ValueError('The JPEG quality needs to be either a number or a tuple of two numbers')

        # Sanitize inputs
        if rounding_approximation is not None and rounding_approximation not in ['sin', 'harmonic', 'soft']:
            raise ValueError('Unsupported rounding approximation: {}'.format(rounding_approximation))

        # Quantization tables
        if trainable:
            q_mtx_luma_init = np.ones((8, 8)) if quality is None else jpeg_qtable(quality, 0)
            q_mtx_chroma_init = np.ones((8, 8)) if quality is None else jpeg_qtable(quality, 1)
            self._q_mtx_luma = self.add_weight('Q_mtx_luma', [8, 8], initializer=tf.constant_initializer(q_mtx_luma_init))
            self._q_mtx_chroma = self.add_weight('Q_mtx_chroma', [8, 8], initializer=tf.constant_initializer(q_mtx_chroma_init))

        # Paramaters
        self.quality = quality
        self.trainable = trainable
        self.rounding_approximation = rounding_approximation
        self.rounding_approximation_steps = rounding_approximation_steps

        # Transformations
        # RGB to YCbCr conversion
        self._color_F = np.array([[0, 0.299, 0.587, 0.114], [128, -0.168736, -0.331264, 0.5], [128, 0.5, -0.418688, -0.081312]])
        self._color_I = np.array([[-1.402 * 128, 1, 0, 1.402], [1.058272 * 128, 1, -0.344136, -0.714136], [-1.772 * 128, 1, 1.772, 0]])
        # DCT
        self._dct_F = np.array([[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
                                [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                                [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                                [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
                                [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
                                [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
                                [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
                                [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]])
        self._dct_I = np.transpose(self._dct_F)                    

    def call(self, inputs):
        # Remember settings
        block_size = 8

        with tf.name_scope('jpeg'):

            # Color conversion (RGB -> YCbCr)
            with tf.name_scope('rgb_to_ycbcr'):
                            
                xc = tf.pad(255.0 * inputs, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=1)
                ycbcr = tf.nn.conv2d(xc, tf.reshape(tf.transpose(self._color_F), [1, 1, 4, 3]), [1, 1, 1, 1], 'SAME', name='jpeg_ycbcr')

            with tf.name_scope('blocking'):
                # Re-organize to get non-overlapping blocks in the following form
                # (n_examples * 3, block_size, block_size, n_blocks)
                p = tf.transpose(ycbcr - 127, [0, 3, 1, 2])
                p = tf.reshape(p, [-1, tf.shape(p)[2], tf.shape(p)[3]])
                p = tf.expand_dims(p, axis=3)
                p = tf.nn.space_to_depth(p, block_size)
                p = tf.transpose(p, [0, 3, 1, 2])
                p = tf.reshape(p, [-1, block_size, block_size, tf.shape(p)[2] * tf.shape(p)[3]])

                # Reorganize to move n_blocks to the first dimension
                r = tf.transpose(p, [0, 3, 1, 2])
                r = tf.reshape(r, [-1, r.shape[2], r.shape[3]])

            # Forward DCT transform
            with tf.name_scope('dct'):
                Xi = tf.matmul(tf.tile(tf.expand_dims(self._dct_F, axis=0), [tf.shape(r)[0], 1, 1]), r)
                X = tf.matmul(Xi, tf.tile(tf.expand_dims(self._dct_I, axis=0), [tf.shape(r)[0], 1, 1]), name='jpeg_dct')

            # Approximate quantization
            with tf.name_scope('quantization'):
                # Tile quantization values for successive channels: 
                # image_0 [R .. R G .. G B .. B] ... image_N [R .. R G .. G B .. B]
                Ql = tf.tile(tf.expand_dims(self._q_mtx_luma, axis=0), [1 * (tf.shape(p)[-1]), 1, 1])
                Qc = tf.tile(tf.expand_dims(self._q_mtx_chroma, axis=0), [2 * (tf.shape(p)[-1]), 1, 1])
                Q = tf.concat((Ql, Qc), axis=0)
                Q = tf.tile(Q, [(tf.shape(inputs)[0]), 1, 1])
                X = X / Q
                X = tf_helpers.quantization(X, 'quantization', self.rounding_approximation, self.rounding_approximation_steps)
                X = X * Q

            with tf.name_scope('idct'):
                # Inverse DCT transform
                xi = tf.matmul(tf.tile(tf.expand_dims(self._dct_I, axis=0), [tf.shape(r)[0], 1, 1]), X)
                xi = tf.matmul(xi, tf.tile(tf.expand_dims(self._dct_F, axis=0), [tf.shape(r)[0], 1, 1]))

            with tf.name_scope('rev-blocking'):
                # Reorganize data back to
                xi = tf.reshape(xi, [3 * tf.shape(inputs)[0], -1, xi.shape[1], xi.shape[2]])
                xi = tf.transpose(xi, [0, 2, 3, 1])

                # Backward re-organization from blocks
                # (n_examples * 3, block, block, n_blocks) -> (n_examples, w, h, 3)
                q = tf.reshape(xi, [-1, tf.shape(xi)[1] * tf.shape(xi)[2], tf.shape(inputs)[1] // block_size,
                                    tf.shape(inputs)[2] // block_size])
                q = tf.transpose(q, [0, 2, 3, 1])
                q = tf.nn.depth_to_space(q, block_size)
                q = tf.reshape(q, [-1, 3, tf.shape(q)[1], tf.shape(q)[2]])
                q = tf.transpose(q, [0, 2, 3, 1])

            # Color conversion (YCbCr-> RGB)
            with tf.name_scope('ycbcr_to_rgb'):
                qc = tf.pad(q + 127, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=1)
                y = tf.nn.conv2d(qc, tf.reshape(tf.transpose(self._color_I), [1, 1, 4, 3]), [1, 1, 1, 1], 'SAME', name='jpeg_y')                                                
                y = y / 255.0                    
                y = tf.clip_by_value(y, 0, 1)

        return y

class JPEG:
    """
    TF model for (a differentiable) approximation of JPEG compression.
    """

    def __init__(self, quality=None, codec='soft', trainable=False):
        """
        Creates a JPEG approximation model.

        Sample usage (separately):

            jpg = DJPG()
            batch_y = jpg.process(batch_x, quality=50)
            
        Sample usage (plugged in after a NIP model):
            
            nip = UNet(sess, tf.get_default_graph(), patch_size=patch_size, loss_metric='L2')
            jpg = DJPG(sess, tf.get_default_graph(), nip.y, nip.x, quality=50, rounding_approximation='sin')
            ...
            fan = FAN(sess, tf.get_default_graph(), n_classes=2, x=imb_out, nip_input=model_a.x, n_convolutions=4)

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param x: input to the DJPG module (TF tensor) or None (creates a placeholder)
        :param nip_input: input to the NIP (if part of a larger model) or None
        :param quality: JPEG quality level or None (can be specified later)
        :param rounding_approximation: None (uses normal rounding), 'sin', 'soft', or 'harmonic'
        :param rounding_approximation_steps: number of approximation terms (for 'harmonic' approx. only)
        """

        # Sanitize inputs
        if codec is not None and codec not in ['libjpeg', 'soft', 'sin', 'harmonic']:
            raise ValueError('Unsupported codec version: {}'.format(codec))

        if codec == 'libjpeg':
            self._model = None
        else:
            self._model = DifferentiableJPEG(quality, codec)

        # Remember settings
        self.codec = codec
        self.quality = quality
        # self.rounding_approximation_steps = rounding_approximation_steps
        # self.init_quality = quality

        # self.x = x
        # self.y = y
        # self.nip_input = nip_input
        # self.Q_mtx_lum = Q_mtx_lum
        # self.Q_mtx_chr = Q_mtx_chr

    def process(self, batch_x, quality=None):

        sampled = False

        if quality is not None:
            if isinstance(quality, tuple) and len(quality) > 2:
                quality = np.random.choice(quality)
                sampled = True
            
            elif isinstance(quality, tuple) and len(quality) == 2:
                quality = np.random.randint(quality[0], quality[1])
                sampled = True
            
            elif is_number(quality):
                quality = int(quality)
                sampled = True
            
            else:
                raise ValueError('Invalid quality! {}'.format(quality))

        if self._model is None:
            return jpeg_helpers.compress_batch(batch_x, quality)[0]
        else:
            if sampled:
                old_q_luma, old_q_chroma = self._model._q_mtx_luma, self._model._q_mtx_chroma
                self._model._q_mtx_luma = jpeg_qtable(quality, 0)
                self._model._q_mtx_chroma = jpeg_qtable(quality, 1)
            
            y = self._model(batch_x).numpy()

            if sampled:
                self._model._q_mtx_luma, self._model._q_mtx_chroma = old_q_luma, old_q_chroma

            return y

    def __repr__(self):
        return 'JPEG(codec={}, trainable={})'.format(self.codec, self._model.trainable)

    def summary(self):
        return 'JPEG(codec={})'.format(self.codec)