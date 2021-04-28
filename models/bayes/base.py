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
from loguru import logger

# Internal libraries
from models.tfmodel import TFModel
from models.layers import PaddedConv2D
import helpers.tf_helpers as tfh
from .temp_scaling import TemperatureScaling


class MCDropoutLayer(tf.keras.layers.Dropout):
    """Dropout layer which always drops some connections."""

    def __init__(self, rate=0.5, **kwargs):
        super().__init__(rate, **kwargs)
        self.dropout = tf.keras.layers.Dropout(rate, **kwargs)

    def call(self, inputs, training=None):
        return self.dropout(inputs, training=True)


class BayesBaseModel(TFModel, TemperatureScaling):
    """Defines a Tensorflow model (keras or not)."""

    def __init__(
            self,
            method: str,
            activation: str,
            drop_rate=0.5,
            mc_num_samples=50,
    ):
        """
        method : str
            Choice between 'vanilla', 'mc-dropout', 'temp-scaling', 'mc-temp',
                                'flipout', 'variational', 'reparameterization'.
        activation : str
            Name of the activation method to be used for model creation.
        drop_rate : float
            Dropout rate used for MC-dropout.
        mc_num_samples : int
            Number of forward passes in MC-dropout.
        """
        super().__init__()

        self.mc_num_samples = mc_num_samples
        self.drop_rate = min(max(drop_rate, 0.0), 1.0)
        self.method = method.lower()
        self.activation = tfh.activation_mapping[activation]

        # hard-coded to False, because model will need to be created.
        self.model_created = False

        # Depending on method, Conv2D, Dense and Dropout layers are chosen.
        if "mc" in method:
            self.conv2d = PaddedConv2D
            self.dropout = MCDropoutLayer
            self.dense = tf.keras.layers.Dense

        elif method == "flipout":
            self.conv2d = tfp.layers.Convolution2DFlipout
            self.dropout = tf.keras.layers.Dropout
            self.dense = tfp.layers.DenseFlipout

        elif method == "reparameterization":
            self.conv2d = tfp.layers.Convolution2DReparameterization
            self.dropout = tf.keras.layers.Dropout
            self.dense = tfp.layers.DenseReparameterization

        elif method == "variational":
            self.conv2d = tfp.layers.Convolution2DVariational
            self.dropout = tf.keras.layers.Dropout
            self.dense = tfp.layers.DenseVariational

        else:
            self.conv2d = PaddedConv2D
            self.dropout = tf.keras.layers.Dropout
            self.dense = tf.keras.layers.Dense

        if "temp" not in method:
            self.temperature = None

    def create_model(self):
        """Top-level model creator and corresponding modifier."""
        # store the vanilla model definition in self._model
        try:
            self._create_model()
        except RuntimeError:
            logger.error("Model creation FAILED.")
        self.model_created = True
        logger.info("Model created successfully.")
        logger.info("Number of parameters in the model = {}".format(
            self.count_parameters()))
        if self.tensorboard:
            self.tensorboard.set_model(model=self._model)
        logger.info("Layers : {}".format(self._model.layers))

    @abstractmethod
    def _create_model(self):
        """Construct the self._model variable with un-compiled keras model.
        Note the forward self._model should return logits only, i.e., output before the
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

    def __call__(self, inputs, training):
        """Internal call method for model forward pass"""
        return self._model(inputs, training=training)
