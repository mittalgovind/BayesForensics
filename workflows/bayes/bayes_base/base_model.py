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

    def __init__(self, dropout_rate=0.5, **kwargs):
        super().__init__(dropout_rate, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout_rate, **kwargs)

    def call(self, inputs):
        return self.dropout(inputs, training=True)


class BayesBaseModel(TFModel):
    """Defines a Tensorflow model (keras or not)."""

    def __init__(self, method):
        """
        method : str
            Choice between 'mc-dropout', 'bootstrap', 'combined', 'ensemble', 'flipout'
        """
        super().__init__()

        # Put all the layer instances used in the forward (call) pass
        method = method.lower()
        if   == 'mc-dropout':
            print('Please make sure you use MCDropoutLayer as a replacement'
                  'of actual dropout.')
            self.conv2d = layers.PaddedConv2D
            self.dropout = MCDropoutLayer
            self.dense = tf.keras.layers.Dense
        elif method.lower()

    def create_mcdropout_model(self):
        """Construct the self._model variable with un-compiled keras model."""
        self._model = tf.keras.models.Sequential([
            layers.PaddedConv2D(
                6, kernel=5, activation=tf.nn.relu),
            MCDropoutLayer(dropout_rate=0.2),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            layers.PaddedConv2D(
                16, kernel=5, activation=tf.nn.relu),
            MCDropoutLayer(dropout_rate=0.2),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            layers.PaddedConv2D(
                120, kernel=5, activation=tf.nn.relu),
            MCDropoutLayer(dropout_rate=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                84, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(
                10, activation=tf.nn.softmax)
        ])

        return None

    def create_flipout_model(self):
        """Construct the self._model variable with un-compiled keras model."""
        self._model = tf.keras.models.Sequential([
            tfp.layers.Convolution2DFlipout(
                6, kernel_size=5, padding='SAME',
                activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                16, kernel_size=5, padding='SAME',
                activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            tfp.layers.Convolution2DFlipout(
                120, kernel_size=5, padding='SAME',
                activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                84, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(
                10, activation=tf.nn.softmax)
        ])

        return None

    # TODO - add all the ways of getting different measures of uncertainty + MC handling
    # TODO -

    def inference

    def process(self, add_n_samples):

    def type_of_uncertainty_1(self, ):

    def specific_ways_uncertainty

    def mutual information_uncertainty()
