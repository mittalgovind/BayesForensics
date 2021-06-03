#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import io

# External libraries
import tensorflow as tf
import imageio
from loguru import logger

# Internal libraries
from helpers.utils import is_number
from models.jpeg import JPEG


class TFJPEG(JPEG):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._q_luma = tf.convert_to_tensor([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ], dtype=tf.int32)
        self._q_chroma = tf.convert_to_tensor([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ], dtype=tf.int32)

    @tf.function(experimental_compile=True)
    def jpeg_qtable(self, quality, channel=0):
        """
        Return a DCT quantization matrix for a given quality level.
        :param quality: JPEG quality level (1-100)
        :param channel: 0 for luminance, >0 for chrominance channels
        """

        # Convert to linear quality scale
        if tf.math.less(quality, 50):
            quality = tf.cast(tf.math.divide(5000, quality), tf.int32)
        else:
            quality = tf.cast(tf.math.add(quality * -2, 200), tf.int32)

        if channel == 0:
            # This is table 0 (the luminance table):
            t = self._q_luma
        else:
            # This is table 1 (the chrominance table):
            t = self._q_chroma

        t = tf.math.multiply(t, quality)
        t = tf.math.add(t, 50)
        t = tf.math.floor(tf.math.divide(t, 100))
        t = tf.clip_by_value(t, clip_value_min=1, clip_value_max=255)

        return t

    @staticmethod
    @tf.function(experimental_compile=True)
    def compress_batch(batch, quality, subsampling="4:4:4"):
        batch_j = tf.zeros((0, *batch.shape[1:]))
        for r in range(batch.shape[0]):
            '''
            s = io.BytesIO()
            imageio.imsave(
                s,
                tf.squeeze(tf.cast((255 * batch[r]), tf.uint8)),
                format="jpg",
                quality=quality,
                subsampling=subsampling,
            )
            image_compressed = imageio.imread(s.getvalue())
            '''
            image_compressed = tf.io.encode_jpeg(
                tf.squeeze(tf.cast((255 * batch[r]), tf.uint8)),
                quality=quality,
            )
            batch_j = tf.concat((batch_j, tf.divide(image_compressed, 255)))

        return batch_j

    @staticmethod
    @tf.function(experimental_compile=True)
    def is_valid_quality(quality):
        if is_number(quality) and tf.greater_equal(quality, 1) \
                and tf.less_equal(quality, 100):
            return True
        return False

    @tf.function(experimental_compile=True)
    def process(self, batch, quality=None, return_entropy=False):
        """Compress an image with given quality"""

        quality = self.quality if quality is None else quality

        if not self.is_valid_quality(quality):
            logger.error("Quality: {}, is not valid".format(quality))

        if self._model is None:
            return self.compress_batch(batch, quality)
        else:
            self._model._q_mtx_luma = self.jpeg_qtable(quality, 0)
            self._model._q_mtx_chroma = self.jpeg_qtable(quality, 1)
            y, _ = self._model(batch)

        return y