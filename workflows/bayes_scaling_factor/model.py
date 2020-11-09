#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf

# Internal libraries
from models.layers import ConstrainedConv2D
from bayes import BayesBaseModel


class SFP(BayesBaseModel):
    def __init__(self, method, c_filters, d_filters, kernel,
                 trainable_residual, drop, append_rgb, **kwargs):

        super().__init__(method=method, **kwargs)

        self._layers = []
        self.c_filters = c_filters
        self.d_filters = d_filters
        self.kernel = kernel
        self.trainable_residual = trainable_residual
        self.drop_rate = drop
        self.append_rgb = append_rgb
        self._residual = ConstrainedConv2D(trainable=self.trainable_residual)

    def _create_model(self):
        """Need to override to specify model architecture."""
        # Setup conv layers
        for n_filters in self.c_filters:
            self._layers.append(
                self.conv2d(n_filters, self.kernel,
                            activation=self.activation))

        self._layers.append(tf.keras.layers.GlobalAvgPool2D())

        # Setup dense layers
        for n, n_filters in enumerate(self.d_filters):
            act = None if n == len(self.d_filters) - 1 else self.activation
            self._layers.append(self.dense(n_filters, activation=act))
            if self.drop_rate > 0 and n < len(self.d_filters) - 1:
                self._layers.append(self.dropout(self.drop_rate))

        self._model = tf.keras.Sequential(self._layers)

    def _call(self, inputs, training=False):
        """Vanilla part of the forward pass for the model."""
        x = inputs
        r = self._residual(x)

        if self.append_rgb:
            f = tf.keras.layers.concatenate([x, r])
        else:
            f = r

        return self._model(f, training=training)
