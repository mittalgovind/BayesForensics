#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from abc import ABC
import numpy as np
import tensorflow as tf

# Internal libraries
from helpers.stats import quantize
from helpers.paramspec import ParamSpec
from helpers.tf_helpers import activation_mapping
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel
from models.bayes.temp_scaling import TemperatureScaling
from models.bayes import DeepEnsemble
from helpers.utils import progress_bar
from helpers.stats import quantize


class SFP(BayesBaseModel):
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

        super().__init__(method=method, **kwargs)

        self._layers = []
        self.c_filters = c_filters
        self.d_filters = d_filters
        self.kernel = kernel
        self.trainable_residual = trainable_residual
        self.drop_rate = drop
        self.append_rgb = append_rgb
        self._residual = ConstrainedConv2D(trainable=self.trainable_residual)

        self._create_model()

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
        self.model_created = True

    def _call(self, inputs, training=False):
        """Vanilla part of the forward pass for the model."""
        x = inputs
        r = self._residual(x)

        if self.append_rgb:
            f = tf.keras.layers.concatenate([x, r])
        else:
            f = r

        return self._model(f, training=training)


class BayarStammSFP(BayesBaseModel):
    """Model taken from Bayar & Stamm.
    Ref: Constrained convolutional neural networks: A new approach towards
     general purpose image manipulation detection.
     IEEE Transactions on Information Forensics and Security, 13 (11), 2018."""

    def __init__(
        self,
        method,
        n_classes,
        patch_size=None,
        n_filters=32,
        n_fscale=2,
        n_convolutions=4,
        kernel=5,
        dropout=0.0,
        use_gap=True,
        n_dense=0,
        activation="leaky_relu",
        **kwargs
    ):
        """
        Creates a forensic analysis network (see class docstring for details).

        :param n_classes: the number of output classes
        :param patch_size: input patch size
        :param n_filters: number of output features for the first conv layer
        :param n_fscale: multiplier for the number of output features in successive conv layers
        :param n_convolutions: the number of standard conv layers
        :param kernel: conv kernel size
        :param dropout: dropout rate for fully connected layers
        :param use_gap: whether to use a GAP or to reshape the final conv tensor
        :param activation: activation function (see helpers.tf_helpers.activation_mapping for available activations)
        """
        super().__init__(method=method, activation=activation, **kwargs)

        # Set-up and validate hyper-parameters
        self._h = ParamSpec(
            {
                "n_classes": (7, int, (2, 256)),
                "n_filters": (32, int, (4, 128)),
                "n_fscale": (2, float, (0.25, 4)),
                "n_convolutions": (4, int, (1, 32)),
                "kernel": (5, int, (3, 11)),
                "dropout": (0, float, (0, 1)),
                "use_gap": (False, bool, None),
                "n_dense": (2, int, (0, 16)),
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

        self._create_model()

    def _create_model(self):
        # Constrained convolution with a learned residual filter
        self._layers.append(ConstrainedConv2D())

        # Standard convolutional layers
        for _ in range(self._h.n_convolutions):
            self._layers.append(
                tf.keras.layers.Conv2D(
                    self._h.n_filters,
                    [self._h.kernel, self._h.kernel],
                    padding="same",
                    activation=self.activation,
                )
            )
            self._layers.append(tf.keras.layers.BatchNormalization())
            self._layers.append(tf.keras.layers.MaxPool2D([2, 2]))
            n_filters = int(self._h.n_filters * self._h.n_fscale)

        n_filters = self._h.n_filters // self._h.n_fscale

        # Final 1 x 1 convolution
        self._layers.append(
            tf.keras.layers.Conv2D(
                int(self._h.n_filters), [1, 1], activation=self.activation
            )
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
        self.model_created = True

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
        """Returns the predicted class (and optionally its confidence) for an image batch (NHWC:rgb)."""
        probs = self._model(batch_x)

        if with_confidence:
            return probs.numpy().argmax(axis=1), probs.numpy().max(axis=1)
        else:
            return probs.numpy().argmax(axis=1)

    def training_step(self, batch_x, target_labels, learning_rate=None):
        """Make a single training step and return the current loss (Use class numbers for target labels)."""
        with tf.GradientTape() as tape:
            class_probabilities = self._model(batch_x)
            loss = self.loss(target_labels, class_probabilities)

        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return loss

    def summary(self):
        return "{kernel}x{kernel} CNN: 1+{conv}+1 conv layers {gap}+ {fc} fc layers [{params:,} parameters]".format(
            kernel=self._h.kernel,
            conv=self._h.n_convolutions,
            fc=self._h.n_dense,
            gap="+ (GAP) " if self._h.use_gap else "",
            params=self.count_parameters(),
        )
    
    def __getattr__(self, name):
        raise AttributeError(name)


class BayarStammCalibrated(BayarStammSFP, ABC):
    """Temperature Scaling subclass for Bayar Stamm model."""

    def __init__(
        self,
        model,
        batch_size,
        patch_size=128,
        scales=(0.25, 1),
        num_classes=31,
    ):
        super().__init__(model, batch_size)
        self.patch_size = patch_size
        self.scales = scales
        self.num_classes = num_classes

    def preprocess(
        self,
        batch,
        return_labels=False,
    ):
        """Resizes a batch of images and returns resized images and their labels."""
        sf = np.random.uniform((1,), *self.scales)
        classes = np.linspace(*self.scales, num=self.num_classes)
        batch_resized = tf.image.resize(
            batch, [int(sf * self.patch_size), int(sf * self.patch_size)]
        )
        class_id = quantize(sf, classes, return_indices=True)
        labels = np.repeat(class_id, len(batch)).reshape((-1, 1))
        labels = tf.convert_to_tensor(labels)
        return batch_resized, labels

    def __call__(self, batch, training, *args, **kwargs):
        super(BayarStammCalibrated, self)._call(batch, training)
