#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import sys
import os

# External libraries
import tensorflow as tf
from loguru import logger

# Internal libraries
from helpers.tf_dataset import Dataset
from helpers.tf_jpeg import TFJPEG
from helpers.loading import randint

# Hacky fix
sys.path.append("/scratch/jms1595/neural-imaging-dev/")


class ScalingFactorDataset(Dataset):
    def __init__(
            self,
            scales,
            sampling_method,
            n_classes,
            codec=None,
            jpeg_quality=100,
            **kwargs,
    ):
        """
        Subclass of helpers.dataset.Dataset class.

        Attributes
        ----------
        scales : tuple
            Range of values for scaling factor.
        patch_size : int
            Side length in pixels of the square patch.
        sampling_method : str
            Method to be used for sampling. Can be one of 'nearest',
             'bilinear', 'bicubic', 'lanczos3', or 'random'.
        n_classes : int
            Number of classes to split the scales range into.
        codec : str or None
            JPEG Compression type to preprocess batch with. None = no compression.
        jpeg_quality : int
            JPEG quality to compress with.
        """
        super().__init__(**kwargs)
        self.scales = (float(scales.split(",")[0]),
                       float(scales.split(",")[1]))
        self.sampling_method = sampling_method
        self.methods = ["nearest", "bilinear", "bicubic", "lanczos3"]
        self.random_method = self.sampling_method == "random"
        self.classes = tf.linspace(*self.scales, num=n_classes)
        self.class_multiplier = tf.convert_to_tensor(
            n_classes / (self.scales[1] - self.scales[0]))
        if codec:
            self.codec = TFJPEG(quality=jpeg_quality, codec=codec)
            if codec == 'libjpeg':
                logger.info('Using libjpeg will be slowing the computation.')
        else:
            self.codec = None

    def preprocess_batch(self, batch, training=True, **kwargs):
        """
        Resize a batch with the desired scaling factor and sampling method.
        Includes JPEG compression when required.
        Returns the resized batch and their corresponding labels.

        Parameters
        ----------
        batch : np.array
            Batch to be preprocessed.

        Returns
        -------
        rescaled_images : tf.Tensor
            Tensor containing the resized batch.
        sf_labels : tf.Tensor
            Tensor containing the target labels.
        """
        # TODO Ask Pawel if compression and scaling factor are interchangeable

        if 'sf' in kwargs:
            sf = float(kwargs['sf'])
            class_id = tf.math.floor(
                tf.math.multiply(self.class_multiplier, sf - self.classes[0]))
        else:
            # changed to sampling from finite set instead of infinite
            class_id = randint(maxval=len(self.classes), seed=self.seed)
            sf = self.classes[class_id]

        patch_size = self.train_rgb_patch_size \
            if training else self.val_rgb_patch_size

        resized_size = tf.cast(tf.math.multiply(sf, patch_size), tf.int32)
        resized_size = tf.broadcast_to(resized_size, shape=(2,))
        # Choose sampling method.
        if self.random_method:
            m = self.methods[randint(maxval=len(self.methods), seed=self.seed)]
        else:
            m = self.sampling_method

        # Resize batch.
        rescaled_images = tf.image.resize(batch, resized_size, method=m)

        # Convert to JPEG if a codec is passed.
        if self.codec:
            rescaled_images = self.pad(rescaled_images, resized_size)
            rescaled_images = self.codec.process(rescaled_images)
            rescaled_images = self.unpad(rescaled_images, resized_size)

        sf_labels = tf.repeat(class_id, batch.shape[0])

        return rescaled_images, sf_labels

    @staticmethod
    def pad(images, resized_size):
        """pad with zeros for multiple of 8"""
        pad_size = 8 - resized_size[0] % 8
        if pad_size % 2 == 1:
            pad_size //= 2
            paddings = tf.constant([[pad_size + 1, pad_size],
                                    [pad_size + 1, pad_size]])
        else:
            pad_size //= 2
            paddings = tf.constant([[pad_size, pad_size],
                                    [pad_size, pad_size]])

        return tf.pad(images, paddings)

    @staticmethod
    def unpad(images, resized_size):
        """pad with zeros for multiple of 8"""
        return None#tf.slice(images, )

    def get_training_pipeline(self, discard="flat"):

        return tf.data.Dataset.from_generator(
            self.get_training_generator,
            args=(discard,),
            output_signature=(tf.TensorSpec((self.batch_size, None, None, 3),
                                            tf.float32),
                              tf.TensorSpec((self.batch_size,), tf.float32)),
        )

    def get_validation_pipeline(self):
        return tf.data.Dataset.from_generator(
            self.get_validation_generator,
            output_signature=(tf.TensorSpec((self.batch_size, None, None, 3),
                                            tf.float32),
                              tf.TensorSpec((self.batch_size,), tf.float32)),
        )

    def get_calibration_pipeline(self):
        return tf.data.Dataset.from_generator(
            self.get_calibration_generator,
            output_signature=(tf.TensorSpec((self.batch_size, None, None, 3),
                                            tf.float32),
                              tf.TensorSpec((self.batch_size,), tf.float32)),
        )
