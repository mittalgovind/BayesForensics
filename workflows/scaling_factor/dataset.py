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

# Internal libraries
from helpers.tf_dataset import Dataset
from helpers.tf_jpeg import TFJPEG

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
        # Convert to JPEG if a codec is passed.
        if self.codec:
            batch = self.codec.process(batch)

        if 'sf' in kwargs:
            sf = float(kwargs['sf'])
        else:
            sf = tf.random.uniform((1,), *self.scales)[0].numpy() - 1e-10

        patch_size = self.train_rgb_patch_size \
            if training else self.val_rgb_patch_size

        resized_size = tf.cast(tf.math.multiply(sf, patch_size), tf.int32)
        resized_size = tf.broadcast_to(resized_size, shape=(2,))
        # Choose sampling method.
        if self.random_method:
            m = self.methods[
                tf.random.uniform(shape=(), minval=0, maxval=len(self.methods),
                                  dtype=tf.int32)]
        else:
            m = self.sampling_method

        # Resize batch.
        rescaled_images = tf.image.resize(batch, resized_size, method=m)
        class_id = tf.math.floor(
            tf.math.multiply(self.class_multiplier, sf - self.classes[0]))
        sf_labels = tf.repeat(class_id, batch.shape[0])
        '''
        # Convert to JPEG if a codec is passed.
        if self.codec:
            rescaled_images = self.codec.process(rescaled_images)
        '''
        return rescaled_images, sf_labels

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
