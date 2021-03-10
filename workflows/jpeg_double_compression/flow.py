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
from tensorflow.keras.layers import Input, MaxPool2D
from tensorflow.keras.models import Model

# Internal libraries
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel


class JPEGDoubleCompression(BayesBaseModel, ABC):
    def __init__(
            self,
            conv_layers,
            dense_layers,
            method,
            filters,
            dense_units,
            pool_size,
            kernel,
            activation='leaky_relu',
            trainable_residual=True,
            drop=0.1,
            append_rgb=True,
            tensorboard=None,
            patch_size=64,
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
        super().__init__(method=method, activation=activation, **kwargs)

        self.filters = filters
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.kernel = kernel
        self.pool_size = pool_size
        self.conv_layers = conv_layers
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
        for _ in range(self.conv_layers):
            layers.append(
                self.conv2d(self.filters, self.kernel,
                            activation=self.activation)
            )
            layers.append(
                MaxPool2D(pool_size=(self.pool_size, self.pool_size)))

        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        # Setup dense layers
        for i in range(self.dense_layers):
            last_layer = i == self.dense_layers - 1
            act = self.activation if not last_layer else None
            dense_units = self.dense_units // (1.5 ** i) if not last_layer else 2
            layers.append(self.dense(dense_units, activation=act))
            if self.drop_rate > 0 and not last_layer:
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
