#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from abc import ABC

import tensorflow as tf

# Internal libraries
from helpers.paramspec import ParamSpec
from helpers.tf_helpers import activation_mapping
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel
from tensorflow.keras.layers import Input


class ScalingFactor(BayesBaseModel):
    """Model taken from Bayar & Stamm.
    Ref: Constrained convolutional neural networks: A new approach towards
     general purpose image manipulation detection.
     IEEE Transactions on Information Forensics and Security, 13 (11), 2018."""

    def __init__(
            self,
            uncertainty_method,
            n_classes,
            patch_size=None,
            filters=32,
            filter_multiplier=2,
            conv_layers=4,
            kernel=5,
            dense_dropout=0.0,
            conv_dropout=0.0,
            use_gap=True,
            use_bn=False,
            conv_dropout_after=2,
            dense_layers=0,
            dense_units=200,
            activation="leaky_relu",
            pool_size=3,
            channels=3,
            num_models=1,
            **kwargs
    ):
        """
        Creates a forensic analysis network (see class docstring for details).

        Attributes
        ==========
        n_classes : int
            the number of output classes.
        patch_size : int
            input patch size.
        filters : int
            number of output features for the first conv layer.
        filter_multiplier : int
            multiplier for number of output features in successive conv layers.
        conv_layers: int
            the number of standard conv layers.
        kernel : int
            conv kernel size.
        dense_layers : int
            number of dense layers.
        dropout : float
            dropout rate for fully connected layers.
        use_gap : bool
            whether to use a GAP or to reshape the final conv tensor.
        activation : str
            activation function.
            (see helpers.tf_helpers.activation_mapping for more activations).
        """
        super().__init__(method=uncertainty_method, activation=activation,
                         **kwargs)

        # Set-up and validate hyper-parameters
        self._h = ParamSpec(
            {
                "n_classes": (7, int, (2, 256)),
                "filters": (32, int, (4, 128)),
                "filter_multiplier": (2, float, (0.25, 4)),
                "conv_layers": (4, int, (1, 32)),
                "kernel": (5, int, (3, 11)),
                "dense_dropout": (0, float, (0, 1)),
                "conv_dropout": (0, float, (0, 1)),
                "conv_dropout_after": (3, int, (1, 32)),
                "use_gap": (True, bool, None),
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
        self.activation = activation_mapping[self._h.activation]
        self._layers = list()

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.performance = dict()
        self.patch_size = patch_size
        self.channels = channels
        self.n_models = num_models
        self.create_model()

    def _create_model(self):
        layers = []
        for i in range(self.n_models):
            layers.append([])
            # Constrained convolution with a learned residual filter
            layers[i] = [ConstrainedConv2D()]
            # Standard convolutional layers
            filters = self._h.filters
            for j in range(self._h.conv_layers):
                layers[i].append(
                    tf.keras.layers.Conv2D(filters,
                                            kernel_size=self._h.kernel,
                                            padding='same',
                                            ctivation=self.activation))
                if self._h.use_bn:
                    layers[i].append(tf.keras.layers.BatchNormalization())
                if self._h.conv_dropout > 0 and j + 1 >= self._h.conv_dropout_after:
                    layers[i].append(
                        tf.keras.layers.SpatialDropout2D(self._h.conv_dropout))
                layers[i].append(tf.keras.layers.MaxPool2D(self._h.pool_size))
                filters = int(filters * self._h.filter_multiplier)

            # Final 1 x 1 convolution
            layers[i].extend([
                tf.keras.layers.Conv2D(filters // self._h.filter_multiplier,
                                       kernel_size=1, padding='same',
                                       activation=self.activation),
                # tf.keras.layers.SpatialDropout2D(self._h.conv_dropout),
            ])

            # GAP / Feature formation
            if self._h.use_gap:
                layers[i].append(tf.keras.layers.GlobalAveragePooling2D())
            else:
                layers[i].append(tf.keras.layers.Flatten())

            # Fully-connected classifier
            for _ in range(self._h.dense_layers):
                layers[i].append(
                    self.dense(self._h.dense_units, activation=self.activation)
                )
                if self._h.dense_dropout > 0:
                    layers[i].append(self.dropout(self._h.dense_dropout))

            # final classification head
            layers[i].append(
                tf.keras.layers.Dense(self._h.n_classes, activation=None)
            )

        inputs = Input(shape=(None, None, self.channels))
        outputs_list = []

        for i in range(self.n_models):
            outputs = inputs

            for layer in layers[i]:
                outputs = layer(outputs)

            outputs_list.append(outputs)

        if self.n_models == 1:
            outputs_list = outputs_list[0]

        self._model = tf.keras.models.Model(inputs, outputs_list)

    def reset_performance_stats(self):
        self.performance = {
            "loss": {"training": [], "validation": []},
            "accuracy": {"validation": []},
            "confusion": [],
        }

    def summary(self):
        return (
            "{kernel}x{kernel} CNN: 1+{conv}+1 conv layers {gap}+ {fc} "
            "fc layers [{params:,} parameters]".format(
                kernel=self._h.kernel,
                conv=self._h.n_convolutions,
                fc=self._h.n_dense,
                gap="+ (GAP) " if self._h.use_gap else "",
                params=self.count_parameters(),
            )
        )

    @property
    def model_code(self):
        return "{c}_{k}x{k}_{l}x{f}f".format(
            c=self.class_name,
            k=self._h.kernel,
            f=self._h.n_features,
            l=self._h.n_layers,
        )
