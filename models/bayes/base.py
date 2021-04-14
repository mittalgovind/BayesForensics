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
from temp_scaling import TemperatureScaling


class MCDropoutLayer(tf.keras.layers.Dropout):
    """Dropout layer with dropout always ."""

    def __init__(self, rate=0.5, **kwargs):
        super().__init__(rate, **kwargs)
        self.dropout = tf.keras.layers.Dropout(rate, **kwargs)

    def call(self, inputs, training=None):
        return self.dropout(inputs, training=True)


class IdentityLayer(tf.keras.layers.Layer):
    """Identity layer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        return inputs


class BayesBaseModel(TFModel, TemperatureScaling):
    """Defines a Tensorflow model (keras or not)."""

    def __init__(
            self,
            method: str,
            activation: str,
            drop_rate=0.5,
            mc_num_samples=50,
            bayesian=True,
            use_own_dropout=False,
    ):
        """
        method : str
            Choice between 'mc-dropout', 'flipout'.
        activation : str
            Name of the activation method to be used for model creation.
        drop_rate : float
            Dropout rate used for MC-dropout.
        mc_num_samples : int
            Number of forward passes in MC-dropout.
        bayesian : bool
            Flag to keep the model bayesian. False makes it a vanilla model.
        """
        super().__init__()

        self.mc_num_samples = mc_num_samples
        self.drop_rate = min(max(drop_rate, 0.0), 1.0)
        self.method = method.lower()
        self.activation = tfh.activation_mapping[activation]
        self.bayesian = bayesian
        self.use_own_dropout = use_own_dropout

        # hard-coded to False, because model will need to be created.
        self.model_created = False

        # Put all the layer instances used in the forward (call) pass
        # Depending on your choice of method, Conv2D, Dense and Dropout are chosen accordingly.
        if bayesian:
            if "mc" in method:
                # captures temperature scaling too
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
        else:
            self.conv2d = PaddedConv2D
            self.dropout = tf.keras.layers.Dropout
            self.dense = tf.keras.layers.Dense

    def mc_dropout(self, x):
        """MC dropout forward pass"""
        x_list = []
        for i in range(self.mc_num_samples):
            if self.use_own_dropout:
                x_tmp = self.dropout(x)
            else:
                x_tmp = tf.identity(x)
            x_tmp = self._model.last_layer(x_tmp)
            x_list.append(x_tmp)

        return tf.convert_to_tensor(x_list)

    def create_model(self):
        """Top-level model creator and corresponding modifier."""
        # store the vanilla model definition in self._model
        self._create_model()

        # configuring model for MC inference
        if "mc" in self.method:
            # make output layer as Identity and copy it to a variable
            if "dense" in self._model.layers[-1].name.lower():
                self._model.last_layer = self._model.layers[-1]
                self._model.layers[-1] = IdentityLayer()
            else:
                logger.error("Model not ending with a dense layer.")

            # if second last layer is not dropout then attach MCDropoutLayer
            if "dropout" not in self._model.layers[-2].name.lower():
                self.use_own_dropout = True
        else:
            self._model.last_layer = IdentityLayer()

        self.model_created = True
        logger.info("Model created successfully.")
        logger.info("{}".format(self._model.layers))

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
        if not self.model_created:
            raise RuntimeError("The model needs to be created in the subclass constructor.")
        logits = self._model(inputs, training=training)
        return self._model.last_layer(logits, training=training)
