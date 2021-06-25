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
        self.test_methods = self.methods \
            if self.random_method else [self.sampling_method]

        self.classes = tf.linspace(*self.scales, num=n_classes)
        self.class_multiplier = tf.convert_to_tensor(
            n_classes / (self.scales[1] - self.scales[0]))
        self.n_classes = n_classes
        self.val_batch = tf.convert_to_tensor(list(
            self.data["validation"]["y"].unbatch())[:self.batch_size])
        if codec:
            self.codec = TFJPEG(quality=jpeg_quality, codec=codec)
            if codec == 'libjpeg':
                logger.info('Using libjpeg will be slowing the computation.')
        else:
            self.codec = None
        self.seen_sfs = []

    def preprocess_batch(self, batch, training=False, **kwargs):
        """
        Resize a batch with the desired scaling factor and sampling method.
        Includes JPEG compression when required.
        Returns the resized batch and their corresponding labels.

        Parameters
        ----------
        batch : np.array
            Batch to be preprocessed.
        training : bool
            Flag for training
        Returns
        -------
        rescaled_images : tf.Tensor
            Tensor containing the resized batch.
        sf_labels : tf.Tensor
            Tensor containing the target labels.
        """
        if 'sf' in kwargs:
            sf = float(kwargs['sf'])
            class_id = tf.math.floor(
                tf.math.multiply(self.class_multiplier, sf - self.classes[0]))
        elif training:
            # changed to sampling from finite set instead of infinite
            class_id = randint(maxval=self.n_classes, seed=self.seed)
            sf = self.classes[class_id]
            self.seen_sfs.append(sf.numpy())
        else:
            raise RuntimeError("Pass an sf value when not training")

        patch_size = self.train_rgb_patch_size \
            if training else self.val_rgb_patch_size

        resized_size = tf.cast(tf.math.multiply(sf, patch_size), tf.int32)
        # Choose sampling method.
        if training and self.random_method:
            m = self.methods[randint(maxval=len(self.methods), seed=self.seed)]
        else:
            m = self.sampling_method

        # Do data augmentation
        if 'rotate' in kwargs and kwargs['rotate']:
            batch = tf.image.rot90(batch, k=randint(maxval=3, seed=self.seed))
        if 'brighten' in kwargs and kwargs['brighten']:
            batch = tf.image.random_brightness(batch, 0.2, seed=self.seed)
        if 'gamma' in kwargs and kwargs['gamma']:
            batch = tf.image.adjust_gamma(batch,
                                          randint(minval=6, maxval=10,
                                                  seed=self.seed) * 0.1)

        # Resize batch.
        rescaled_images = tf.image.resize(batch, [resized_size, resized_size],
                                          method=m)

        # Convert to JPEG if a codec is passed.
        if self.codec:
            rescaled_images, pad_before = self.pad(
                rescaled_images, resized_size)
            rescaled_images = self.codec.process(rescaled_images)
            rescaled_images = self.unpad(rescaled_images, pad_before,
                                         resized_size)

        rescaled_images = tf.math.divide(rescaled_images, 255)
        sf_labels = tf.repeat(class_id, batch.shape[0])

        return rescaled_images, sf_labels

    @staticmethod
    def pad(images, resized_size):
        """pad with zeros for multiple of 8"""
        pad_size = 8 - resized_size.numpy() % 8
        if pad_size % 2 == 1:
            pad_size //= 2
            pad_before = pad_size + 1
        else:
            pad_size //= 2
            pad_before = pad_size

        paddings = [[0, 0], [pad_before, pad_size],
                    [pad_before, pad_size], [0, 0]]

        return tf.pad(images, paddings), pad_before

    def unpad(self, images, pad_before, resized_size):
        """pad with zeros for multiple of 8"""
        return tf.slice(
            images, begin=[0, pad_before, pad_before, 0],
            size=[len(images), resized_size, resized_size, self.channels]
        )

    def get_validation_generator(self, **kwargs):
        if 'sf' in kwargs:
            for batch in self.data["validation"]["y"]:
                yield self.preprocess_batch(batch, training=False, **kwargs)
        else:
            for m, method in enumerate(self.test_methods):
                if self.random_method:
                    self.sampling_method = method
                for s, sf in enumerate(self.classes[:-1]):
                    yield self.preprocess_batch(self.val_batch, training=False, sf=sf)

    def get_calibration_generator(self, **kwargs):
        for m, method in enumerate(self.test_methods):
            if self.random_method:
                self.sampling_method = method
            for s, sf in enumerate(self.classes[:-1]):
                for batch in self.data["calibration"]["y"]:
                    yield self.preprocess_batch(batch, training=False, sf=sf)

    def get_training_generator(self, discard="flat", gamma=False,
                               brighten=False, rotate=False, **kwargs):
        """
        Get a generator for training data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_training_generator(batch_size, rgb_patch_size, discard),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        kwargs['gamma'] = gamma
        kwargs['brighten'] = brighten
        kwargs['rotate'] = rotate
        for batch in self.data["training"]["y"]:
            if not self.preloading_train:
                batch = self.sample_patches(batch, discard, **kwargs)
            images, labels = self.preprocess_batch(batch, training=True,
                                                   **kwargs)
            yield images, labels

    def get_training_pipeline(self, discard="flat", gamma=False,
                              brighten=False, rotate=False):
        return tf.data.Dataset.from_generator(
            self.get_training_generator,
            args=(discard, gamma, brighten, rotate),
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
