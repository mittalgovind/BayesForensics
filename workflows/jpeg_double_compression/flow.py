#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from abc import ABC
import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPool2D

# Internal libraries
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel
from helpers.paramspec import ParamSpec
from helpers.tf_helpers import activation_mapping


class JPEGDoubleCompression(BayesBaseModel, ABC):
    def __init__(
            self,
            conv_layers,
            dense_layers,
            filters,
            dense_units,
            pool_size,
            kernel,
            dense_multiplier,
            filter_multiplier,
            use_bn=True,
            activation="leaky_relu",
            trainable_residual=True,
            dense_dropout=0.0,
            conv_dropout=0.0,
            conv_dropout_after=2,
            residual_type="trainable",
            patch_size=64,
            num_models=1,
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
        super().__init__(activation=activation, drop_rate=dense_dropout,
                         **kwargs)

        # Set-up and validate hyper-parameters
        self._h = ParamSpec(
            {
                "filters": (32, int, (4, 128)),
                "filter_multiplier": (2, float, (0.25, 4)),
                "dense_multiplier": (2, float, (0.25, 4)),
                "conv_layers": (4, int, (1, 32)),
                "kernel": (5, int, (3, 11)),
                "dense_dropout": (0, float, (0, 1)),
                "conv_dropout": (0, float, (0, 1)),
                "conv_dropout_after": (3, int, (1, 32)),
                "use_bn": (False, bool, None),
                "dense_layers": (2, int, (0, 4)),
                "activation": (
                    "prelu", str, set(activation_mapping.keys())),
                "pool_size": (2, int, (1, 4)),
                "dense_units": (200, int, (100, 400))
            }
        )
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})

        # initialize channels and residual type
        self.channels = 3
        if residual_type == "trainable":
            self.residual = ConstrainedConv2D(trainable=True)
        elif residual_type == "pywt":
            self.residual = None
            # as input already contains the filter.
            self.channels = 6
        else:
            self.residual = None

        self.patch_size = patch_size
        self.n_models = num_models

        self.create_model()

    def _create_model(self):
        """Need to override to specify model architecture."""
        layers = []

        for j in range(self.n_models):
            layers.append([])
            # Setup conv layers
            filters = self._h.filters
            for i in range(self._h.conv_layers):
                layers[j].append(
                    self.conv2d(filters,
                                kernel_size=self._h.kernel,
                                padding='same',
                                activation=self.activation))
                if self._h.use_bn:
                    layers[j].append(tf.keras.layers.BatchNormalization())
                if self._h.conv_dropout > 0 and i + 1 >= self._h.conv_dropout_after:
                    layers[j].append(
                        tf.keras.layers.SpatialDropout2D(self._h.conv_dropout))
                layers[j].append(tf.keras.layers.MaxPool2D(self._h.pool_size))
                filters = int(filters * self._h.filter_multiplier)

            layers[j].append(
                self.conv2d(filters // self._h.filter_multiplier, 1,
                            padding='same', activation=self.activation))
            layers[j].append(tf.keras.layers.Flatten())

            # Setup dense layers
            dense_units = self._h.dense_units
            for i in range(self._h.dense_layers):
                layers[j].append(self.dense(dense_units,
                                            activation=self.activation))
                if self._h.dense_dropout > 0:
                    layers[j].append(self.dropout(self._h.dense_dropout))
                dense_units = int(dense_units * self._h.dense_multiplier)

            # Final classification head
            layers[j].append(self.dense(2, activation=None))

        inputs = Input(shape=(self.patch_size, self.patch_size, self.channels))
        outputs_list = []

        for i in range(self.n_models):
            if self.residual:
                # concatenate residual if append_rgb is true
                outputs = tf.keras.layers.concatenate([inputs,
                                                       self.residual(inputs)])
            else:
                outputs = inputs

            for layer in layers[i]:
                outputs = layer(outputs)

            outputs_list.append(outputs)

        if self.n_models == 1:
            outputs_list = outputs_list[0]

        self._model = tf.keras.models.Model(inputs, outputs_list)
