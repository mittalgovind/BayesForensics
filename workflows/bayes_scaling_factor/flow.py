#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf

# Internal libraries
from helpers.paramspec import ParamSpec
from helpers.tf_helpers import activation_mapping
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel
from models.bayes import DeepEnsemble


class ScalingFactor(BayesBaseModel):
    """Model taken from Bayar & Stamm.
    Ref: Constrained convolutional neural networks: A new approach towards
     general purpose image manipulation detection.
     IEEE Transactions on Information Forensics and Security, 13 (11), 2018."""

    def __init__(
        self,
        method,
        n_classes,
        patch_size=None,
        filters=32,
        filter_multiplier=2,
        conv_layers=4,
        kernel=5,
        dropout=0.0,
        use_gap=True,
        dense_layers=0,
        activation="leaky_relu",
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
        super().__init__(method=method, activation=activation, **kwargs)

        # Set-up and validate hyper-parameters
        self._h = ParamSpec(
            {
                "n_classes": (7, int, (2, 256)),
                "filters": (32, int, (4, 128)),
                "filter_multiplier": (2, float, (0.25, 4)),
                "conv_layers": (4, int, (1, 32)),
                "kernel": (5, int, (3, 11)),
                "dropout": (0, float, (0, 1)),
                "use_gap": (False, bool, None),
                "dense_layers": (2, int, (0, 16)),
                "activation": ("leaky_relu", str, set(activation_mapping.keys())),
            }
        )
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})
        self.activation = activation_mapping[self._h.activation]
        self._layers = list()

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.performance = dict()

        self.create_model()

    def _create_model(self):
        # Constrained convolution with a learned residual filter
        self._layers.append(ConstrainedConv2D())

        # Standard convolutional layers
        for _ in range(self._h.conv_layers):
            self._layers.append(
                self.conv2D(
                    self._h.filters,
                    [self._h.kernel, self._h.kernel],
                    padding="same",
                    activation=self.activation,
                )
            )
            self._layers.append(tf.keras.layers.BatchNormalization())
            self._layers.append(tf.keras.layers.MaxPool2D([2, 2]))
            filters = int(self._h.n_filters * self._h.n_fscale)

        n_filters = self._h.n_filters // self._h.n_fscale

        # Final 1 x 1 convolution
        self._layers.append(
            self.conv2D(int(self._h.n_filters), [1, 1], activation=self.activation)
        )

        # GAP / Feature formation
        if self._h.use_gap:
            self._layers.append(tf.keras.layers.GlobalAveragePooling2D())
        else:
            self._layers.append(tf.keras.layers.Flatten())

        # Fully-connected classifier
        for _ in range(self._h.n_dense):
            self._layers.append(
                tf.keras.layers.Dense(n_filters, activation=self.activation)
            )
            if self._h.dropout > 0:
                self._layers.append(tf.keras.layers.Dropout(self._h.dropout))

        self._layers.append(
            tf.keras.layers.Dense(
                self._h.n_classes, activation=tf.keras.activations.softmax
            )
        )

        self._model = tf.keras.Sequential(self._layers)

    def reset_performance_stats(self):
        self.performance = {
            "loss": {"training": [], "validation": []},
            "accuracy": {"validation": []},
            "confusion": [],
        }

    def _call(self, batch_x, training=False):
        """Returns class probabilities for an image batch (NHWC:rgb)."""
        return self._model(batch_x, training)

    def process_and_decide(self, batch_x, with_confidence=False):
        """Returns the predicted class (and optionally its confidence)
        for an image batch (NHWC:rgb)."""
        probs = self._model(batch_x)

        if with_confidence:
            return probs.numpy().argmax(axis=1), probs.numpy().max(axis=1)
        else:
            return probs.numpy().argmax(axis=1)

    def training_step(self, batch_x, target_labels, learning_rate=None):
        """Make a single training step and return the current loss
        (Use class numbers for target labels)."""
        with tf.GradientTape() as tape:
            class_probabilities = self._model(batch_x)
            loss = self.loss(target_labels, class_probabilities)

        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return loss

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

    def __getattr__(self, name):
        raise AttributeError(name)
