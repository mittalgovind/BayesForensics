#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Internal libraries
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel


class JPEGDoubleCompression(BayesBaseModel, ABC):
    def __init__(
        self,
        method,
        c_filters,
        d_filters,
        kernel,
        trainable_residual,
        drop,
        append_rgb,
        tensorboard,
        patch_size,
        **kwargs
    ):
        """
        c_filters: int
            a tuple with the numbers of filters for initial conv layers
        d_filters: int
            a tuple with the numbers of filters for final dense layers
        kernel: int
            kernel size for the initial convolutions
        activation: int
            activation function (string, see tf_helpers.activation_mapping)
        trainable_residual: bool
            flag to make the residual trainable (see layers.ConstrainedConv2D)
        """
        super().__init__(method=method, **kwargs)

        self.c_filters = c_filters
        self.d_filters = d_filters
        self.kernel = kernel
        self.trainable_residual = trainable_residual
        self.drop_rate = drop
        self.append_rgb = append_rgb
        self.residual = ConstrainedConv2D(trainable=self.trainable_residual)
        self.tensorboard = tensorboard
        self.patch_size = patch_size

        # Needs to be called as the last line in the subclass.
        self.create_model()
    def _create_model(self):
        """Need to override to specify model architecture."""

        layers = []

        # Setup conv layers
        for n_filters in self.c_filters:
            layers.append(
                self.conv2d(n_filters, self.kernel, activation=self.activation)
            )
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        # Setup dense layers
        for n, n_filters in enumerate(self.d_filters):
            act = None if n == len(self.d_filters) - 1 else self.activation
            layers.append(self.dense(n_filters, activation=act))
            if self.drop_rate > 0 and n < len(self.d_filters) - 1:
                layers.append(self.dropout(self.drop_rate))

        # make a custom keras model
        inputs = Input(shape=(self.patch_size, self.patch_size, 3))
        if self.append_rgb:
            # concatenate residual if append_rgb is true
            outputs = tf.keras.layers.concatenate(
                [inputs, self.residual(inputs)])
        else:
            outputs = inputs

        for layer in layers:
            outputs = layer(outputs)

        self._model = tf.keras.models.Model(inputs, outputs)
        if self.tensorboard:
            self.tensorboard.set_model(model=self._model)
