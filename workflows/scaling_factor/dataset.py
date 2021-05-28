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
from models.jpeg import JPEG

# Hacky fix
sys.path.append("/scratch/jms1595/neural-imaging-dev/")


class ScalingFactorDataset(Dataset):
    def __init__(
            self,
            scales,
            patch_size,
            sampling_method,
            n_classes,
            codec=None,
            jpeg_quality=100,
            batch_size=64,
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
        super().__init__(val_rgb_patch_size=patch_size, batch_size=batch_size,
                         **kwargs)
        self.scales = (float(scales.split(",")[0]),
                       float(scales.split(",")[1]))
        self.patch_size = patch_size
        self.sampling_method = sampling_method
        self.methods = ["nearest", "bilinear", "bicubic", "lanczos3"]
        self.random_method = self.sampling_method == "random"
        self.classes = tf.linspace(*self.scales, num=n_classes)
        self.class_multiplier = tf.convert_to_tensor(
            n_classes / (self.scales[1] - self.scales[0]))
        if codec:
            self.codec = JPEG(quality=jpeg_quality, codec=codec)
        else:
            self.codec = None

    def preprocess_batch(self, batch, **kwargs):
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

        if 'sf' in kwargs:
            sf = float(kwargs['sf'])
        else:
            sf = tf.random.uniform((1,), *self.scales)[0].numpy() - 1e-10

        resized_size = tf.cast(tf.math.multiply(sf, self.patch_size), tf.int32)
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

        # Convert to JPEG if a codec is passed.
        if self.codec:
            rescaled_images = self.codec.process(rescaled_images)

        return rescaled_images, sf_labels

    def get_training_generator(self, batch_size, patch_size, discard="flat",
                               **kwargs):
        """
        Get a generator for training data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_training_generator(batch_size, rgb_patch_size, discard),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch in self.batched_data_train:
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_validation_generator(self, batch_size, **kwargs):
        """
        Get a generator for validation data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_validation_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        for batch in self.batched_data_val:
            images, labels = self.preprocess_batch(batch, **kwargs)
            yield images, labels

    def get_training_pipeline(self, batch_size, rgb_patch_size,
                              discard="flat"):

        return tf.data.Dataset.from_generator(
            self.get_training_generator,
            args=(batch_size, rgb_patch_size, discard),
            output_signature=(tf.TensorSpec((batch_size, None, None, 3),
                                            tf.float32),
                              tf.TensorSpec((batch_size,), tf.float32)),
        )

    def get_validation_pipeline(self, batch_size):
        return tf.data.Dataset.from_generator(
            self.get_validation_generator,
            args=(batch_size,),
            output_signature=(tf.TensorSpec((batch_size, None, None, 3),
                                            tf.float32),
                              tf.TensorSpec((batch_size,), tf.float32)),
        )

    def get_calibration_pipeline(self, batch_size):
        return tf.data.Dataset.from_generator(
            self.get_calibration_generator,
            args=(batch_size,),
            output_signature=(tf.TensorSpec((batch_size, None, None, 3),
                                            tf.float32),
                              tf.TensorSpec((batch_size,), tf.float32)),
        )
