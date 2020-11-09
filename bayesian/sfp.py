#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf

# Internal libraries
import helpers.tf_helpers as tfh
from models import layers
from models.tfmodel import TFModel


class MCDropoutLayer(tf.keras.layers.Dropout):
    """Dropout layer with dropout always ."""

    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(dropout_rate, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout_rate, **kwargs)

    def call(self, inputs):
        return self.dropout(inputs, training=True)


class SFP(tf.keras.Model):
    """
    A simple generic TF Model for detection of sensor fingerprints. The model processes
    a combination of image residual and the target fingerprint through a series of conv
    and dense layers. The model uses a GAP to allow for fully-convolutional operation.
    """

    def __init__(self, c_filters, d_filters, kernel, activation,
                 trainable_residual, drop, append_rgb, **kwargs):
        """
        :param c_filters: a tuple with the numbers of filters for initial conv layers
        :param d_filters: a tuple with the numbers of filters for final dense layers
        :param kernel: kernel size for the initial convolutions
        :param activation: activation function (string, see tf_helpers.activation_mapping)
        :param trainable_residual: flag to make the residual trainable (see layers.ConstrainedConv2D)
        """
        super().__init__(**kwargs)
        activation = tfh.activation_mapping[activation]
        self._residual = layers.ConstrainedConv2D(trainable=trainable_residual)
        self._layers = []
        self.append_rgb = append_rgb
        # Setup conv layers
        for n_filters in c_filters:
            self._layers.append(
                layers.PaddedConv2D(n_filters, kernel, activation=activation))
        self._layers.append(tf.keras.layers.GlobalAvgPool2D())
        # Setup dense layers
        for n, n_filters in enumerate(d_filters):
            act = None if n == len(d_filters) - 1 else activation
            self._layers.append(
                tf.keras.layers.Dense(n_filters, activation=act))
            if drop > 0 and n < len(d_filters) - 1:
                # self._layers.append(tf.keras.layers.Dropout(drop))
                self._layers.append(MCDropoutLayer(dropout_rate=drop))
        # self.create_model()
        # TODO - what would be the input shape in this case?
        # self.x = tf.keras.Input(dtype=tf.uint8, shape=())

    def call(self, inputs, training=False):
        x = inputs
        r = self._residual(x)

        if self.append_rgb:
            f = tf.keras.layers.concatenate([x, r])
        else:
            f = r

        for l in self._layers:
            # if 'dropout' in l.name:
            #     f = l(f, training=True)
            # else:
            f = l(f, training=training)

        return f

