#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
from abc import abstractmethod, ABC

# External libraries
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from loguru import logger

# Internal libraries
from models.tfmodel import TFModel
from models import layers
import helpers.tf_helpers as tfh


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


class TemperatureScaling(tf.keras.Model):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = tf.Variable(1.0)

    def forward(self, inputs, training):
        logits = self.model(inputs, training=training)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """Perform temp scaling on logits"""
        temperature = self.temperature.expand_as(logits)
        return logits / temperature

    @staticmethod
    def nll_loss(logits, labels):
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))

    @abstractmethod
    def

    def set_temp(self, data):
        """Use validation dataset to calibrate the model."""
        logits_list = []
        labels_list = []
        # TODO remove hard coding
        patch_size = 128
        batch_size = 64
        n_batches = 2

        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, batch_size,
                                               patch_size)
            labels = tf.zeros(len(batch))
            logits_list.append(self.model(batch, training=False))
            labels_list.append(labels)

        logits = tf.convert_to_tensor(logits_list)
        labels = tf.convert_to_tensor(labels_list)

        optimizer_kernel = tfa.optimizers.AdamW(
            learning_rate=0.01, betas=(0.9, 0.999))
        nll_loss = self.nll_loss(self.temperature_scale(logits), labels)
        ece_loss = self.ece_loss(logits, labels)
        step_count = optimizer_kernel.minimize(loss, [self.temperature])

        return loss

    def ece_loss(self, logits, labels, n_bins=15):
        bin_boundaries = tf.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[: -1]
        bin_uppers = bin_boundaries[1:]
        softmaxes = tf.nn.softmax(logits, axis=1)
        confidences, predictions = tf.maximum(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = tf.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class BayesBaseModel(TFModel):
    """Defines a Tensorflow model (keras or not)."""

    def __init__(self, method: str, activation: str, drop_rate=0.5,
                 mc_num_samples=50, bayesian=True):
        """
        method : str
            Choice between 'mc-dropout', 'temp-scaling', 'flipout'.
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
        # self._deep_ensemble_model = create_ensemble_from_model()

        self.mc_num_samples = mc_num_samples
        self.drop_rate = min(max(drop_rate, 0.0), 1.0)
        self.method = method.lower()
        self.activation = tfh.activation_mapping[activation]
        self.bayesian = bayesian
        self.model_created = False
        self.use_own_dropout = False

        # Put all the layer instances used in the forward (call) pass
        # Depending on your choice of method, Conv2D, Dense and Dropout are chosen accordingly.
        # TODO Add more layers below depending how the use-cases expand
        if bayesian:
            if 'mc' in method:
                # captures temperature scaling too
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
        else:
            self.conv2d = layers.PaddedConv2D
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
            x_tmp = self._model._fc(x_tmp)
            x_list.append(x_tmp)

        return tf.convert_to_tensor(x_list)

    # TODO temperature scaling is slow. why?
    def temp_scaling(self, x):
        """temperature scaling forward pass"""
        if self.bayesian:
            fx = self.mc_dropout(x)
        else:
            fx = self._model._fc(x)
        return fx / tf.keras.activations.relu(self._model.temperature)

    def create_model(self):
        self._create_model()

        # configuring model for MC inference
        if 'mc' in self.method:
            # make output layer as Identity and copy it to a variable
            if 'dense' in self._model._layers[-1].name.lower():
                self._model._fc = self._model._layers[-1]
                self._model._layers[-1] = IdentityLayer()
            else:
                logger.error("Model not ending with a dense layer.")

            # if second last layer is not dropout then attach MCDropoutLayer
            if 'dropout' not in self._model._layers[-2].name.lower():
                self.use_own_dropout = True
        else:
            self._model._fc = IdentityLayer()

        if self.method == 'temp-scaling':
            self._model.temperature = tf.Variable(1.0)

    def __call__(self, inputs, training):
        if not self.model_created:
            self.create_model()
            self.model_created = True
            logger.info("Model created successfully.")

        logits = self._call(inputs, training)

        if self.method == 'mc-dropout':
            return self.mc_dropout(logits)

        elif self.method == 'mc-temp-scaling':
            return self.temp_scaling(logits)

        else:
            return self._model._fc(logits, training)

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

    def _call(self, inputs, training=None):
        """Forward pass through model. Extend this and not the call method."""
        return self._model(inputs, training=training)


class SFP(BayesBaseModel):
    def __init__(self, c_filters, d_filters, kernel,
                 trainable_residual, drop, append_rgb, **kwargs):
        super().__init__(**kwargs)

        self._layers = []
        self.c_filters = c_filters
        self.d_filters = d_filters
        self.kernel = kernel
        self.trainable_residual = trainable_residual
        self.drop_rate = drop
        self.append_rgb = append_rgb
        self._residual = layers.ConstrainedConv2D(
            trainable=self.trainable_residual)

    def _create_model(self):
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
        x = inputs
        r = self._residual(x)

        if self.append_rgb:
            f = tf.keras.layers.concatenate([x, r])
        else:
            f = r

        return self._model(f, training=training)


# model = SFP(method='mc-temp-scaling', c_filters=(32, 32, 32, 32),
#             d_filters=(32, 16, 31), kernel=5,
#             activation='leaky_relu', trainable_residual=True,
#             drop=0.1, append_rgb=False)
pass
