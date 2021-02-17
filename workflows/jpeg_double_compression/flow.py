#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import numpy as np
import tensorflow as tf

# Internal libraries
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel


class JPEGDoubleCompression(BayesBaseModel):
    def __init__(
        self,
        method,
        c_filters,
        d_filters,
        kernel,
        trainable_residual,
        drop,
        append_rgb,
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
                self.conv2d(n_filters, self.kernel, activation=self.activation)
            )

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

    def _mc_dropout(self, inputs):
        x = inputs
        r = self._residual(x)

        if self.append_rgb:
            f = tf.keras.layers.concatenate([x, r])
        else:
            f = r

        for l in self._layers:
            f = l(f, training=isinstance(l, tf.keras.layers.Dropout))

        return f

    def mc_dropout(self, x, n_samples=10):

        y_pred = self._mc_dropout(x)
        p_pred = np.zeros(tuple(y_pred.shape) + (n_samples,))

        for n in range(n_samples):
            y_pred = self._mc_dropout(x)
            p_pred[..., n] = tf.nn.softmax(y_pred)

        # Clip to eliminate zeroes
        p_pred = tf.clip_by_value(p_pred, 1e-12, 1)

        p_avg = p_pred.numpy().mean(axis=-1)
        p_entr = np.sum(-p_avg * np.log2(p_avg), axis=-1)

        # Compute uncertainty as mutual information
        p_mi = p_entr - np.mean(p_pred * np.log2(p_pred), axis=(1, 2))

        return p_avg, p_mi
