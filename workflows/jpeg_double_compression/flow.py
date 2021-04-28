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
            dense_multiplier,
            filter_multiplier,
            activation='leaky_relu',
            trainable_residual=True,
            drop=0.1,
            residual_type='trainable',
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
        if residual_type == 'trainable':
            self.residual = ConstrainedConv2D(
                trainable=self.trainable_residual)
        elif residual_type == 'pywt':
            # TODO easy fix - push it to "main" dataset class.
            self.residual = self.extract_pywt_residual
            self._color_F = np.array(
                [[0, 0.299, 0.587, 0.114], [128, -0.168736, -0.331264, 0.5],
                 [128, 0.5, -0.418688, -0.081312]], dtype=np.float32)

        self.tensorboard = tensorboard
        self.patch_size = patch_size
        self.filter_multiplier = filter_multiplier
        self.dense_multiplier = dense_multiplier

        # Needs to be called as the last line in the subclass.
        self.create_model()

    def _create_model(self):
        """Need to override to specify model architecture."""
        layers = []

        # Setup conv layers
        for i in range(self.conv_layers):
            filters = int(self.filters * self.filter_multiplier ** i)
            layers.append(
                self.conv2d(filters, self.kernel,
                            activation=self.activation)
            )
            layers.append(
                MaxPool2D(pool_size=(self.pool_size, self.pool_size)))

        layers.append(tf.keras.layers.Flatten())

        # Setup dense layers
        for i in range(self.dense_layers):
            dense_units = int(self.dense_units * self.dense_multiplier ** i)
            layers.append(self.dense(dense_units, activation=self.activation))
            if self.drop_rate > 0:
                layers.append(self.dropout(self.drop_rate))
        layers.append(self.dense(2, activation=None))

        # make a custom keras model
        inputs = Input(shape=(self.patch_size, self.patch_size, 3))
        if self.residual:
            # concatenate residual if append_rgb is true
            outputs = tf.keras.layers.concatenate(
                [inputs, self.residual(inputs)])
        else:
            outputs = inputs

        for layer in layers:
            outputs = layer(outputs)

        self._model = tf.keras.models.Model(inputs, outputs)
