#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
from abc import abstractmethod

# External libraries
import tensorflow as tf
import tensorflow_probability as tfp

# Internal libraries
from models.tfmodel import TFModel
from models import layers


class MCDropoutLayer(tf.keras.layers.Dropout):
    """Dropout layer with dropout always ."""

    def __init__(self, rate=0.5, **kwargs):
        super().__init__(rate, **kwargs)
        self.dropout = tf.keras.layers.Dropout(rate, **kwargs)

    def call(self, inputs, training=None):
        return self.dropout(inputs, training=True)


class BayesBaseModel(TFModel):
    """Defines a Tensorflow model (keras or not)."""

    def __init__(self, method):
        """
        method : str
            Choice between 'mc-dropout', 'bootstrap', 'combined', 'ensemble', 'flipout'
        """
        super().__init__()
        self._model
        self._deep_ensemble_model = create_ensemble_from_model()
        self.
        # TODO Things to have
        # could group 1-5
        # 1. classic model - only uncertainty is predictive entropy
        # 2. temperature scaling - for softmax (need function for calibration)
        # 3. MC-dropout
        # 4. Flipout
        # 5. Re-parameterization trick + Bayes-by-Backprop

        # abstract method with example and instructions.
        # 6. Deep Ensemble
        # 7. Energy-based model
        # 8. Deep uncertainty Quantification
        # put helper - calculating uncertainities, running inference.

        # TODO 2 - get quantified results with larger experiments.
        # 1. scaling algorithms (interpolation methods)
        # 2. image is post-processed or not (compressed or not, and how strongly).
        # 3. scaling factor is in the training range.
        # 4. different content of images (e.g., native vs resized).
        #   ON drive : native = native, clic/kodak = resized, raw = native,

        # Put all the layer instances used in the forward (call) pass
        # Depending on your choice of method, Conv2D, Dense and Dropout are chosen accordingly.

        # Remark : Check if dropout implementation is encapsulated for conv vs dense.
        # sampling in conv is done for each channels.
        # makes more sense to enable/disable for each channel.

        method = method.lower()
        if method == 'mc-dropout':
            self.conv2d = layers.PaddedConv2D
            self.dropout = MCDropoutLayer
            self.dense = tf.keras.layers.Dense

        elif method == 'flipout':
            self.conv2d = tfp.layers.Convolution2DFlipout
            self.dropout = tf.keras.layers.Dropout
            self.dense = tfp.layers.DenseFlipout

        else:
            self.conv2d = layers.PaddedConv2D
            self.dropout = tf.keras.layers.Dropout
            self.dense = tf.keras.layers.Dense

        self.create_model()

    def create_model(self):
        """Construct the self._model variable with un-compiled keras model.
        Same definition works both for mc-dropout and flipout
        Example :
        self._model = tf.keras.models.Sequential([
            self.conv2d(
                6, kernel=5, activation=tf.nn.relu),
            self.dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            self.conv2d(
                16, kernel=5, activation=tf.nn.relu),
            self.dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            self.conv2d(
                120, kernel=5, activation=tf.nn.relu),
            self.dropout(rate=0.2),
            tf.keras.layers.Flatten(),
            self.dense(
                84, activation=tf.nn.relu),
            self.dropout(rate=0.2),
            self.dense(
                10, activation=tf.nn.softmax)
        ])
        """
        raise NotImplementedError

    # def evaluate_
    # def get_uncertainties

    def inference(self, data_batch, n_passes=50):
        data_batch_multipass = tf.tile(data_batch, n_passes)

    def bootstrap_model(self, num_heads=10):
        """Model appended with n-output heads and
        predictions are done simultaneously."""
        # TODO - add paper references.
        outputs = []
        x = self._model.get_layer(-2)
        for i in range(num_heads):
            logits = tf.layers.dense(inputs=dropout3, units=10)
            class_prob = tf.nn.softmax(logits, name="softmax_tensor")
            outputs.append([logits, class_prob])

    def combined_model(self):
        # Discuss addition of the 'combined' method and '
        if not self._model:
            raise ValueError('Model is not created.')
        try:
            # TODO update the uncertainty estimation layer.
            self._uncertainty_layer = self.dense(inputs=self._model(), units=10)
        except RuntimeError:
            raise RuntimeError('Wrong parameters passed to uncertainty layer.')

