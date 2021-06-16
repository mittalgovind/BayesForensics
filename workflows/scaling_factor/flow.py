#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from abc import ABC

import tensorflow as tf

# Internal libraries
from helpers.paramspec import ParamSpec
from helpers.tf_helpers import activation_mapping
from models.layers import ConstrainedConv2D
from models.bayes import BayesBaseModel
from tensorflow.keras.layers import Input
import tensorflow_probability as tfp


class ScalingFactor(BayesBaseModel):
    """Model taken from Bayar & Stamm.
    Ref: Constrained convolutional neural networks: A new approach towards
     general purpose image manipulation detection.
     IEEE Transactions on Information Forensics and Security, 13 (11), 2018."""

    def __init__(
            self,
            n_classes,
            patch_size=None,
            filters=32,
            filter_multiplier=2,
            conv_layers=4,
            kernel=5,
            dense_dropout=0.0,
            conv_dropout=0.0,
            use_gap=True,
            use_bn=False,
            conv_dropout_after=2,
            dense_layers=0,
            dense_units=200,
            activation="leaky_relu",
            pool_size=3,
            channels=3,
            num_models=1,
            hyperoptimize=False,
            n_train_images=0,
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
        super().__init__(activation=activation, **kwargs)

        # Set-up and validate hyper-parameters
        self._h = ParamSpec(
            {
                "n_classes": (7, int, (2, 256)),
                "filters": (32, int, (4, 128)),
                "filter_multiplier": (2, float, (0.25, 4)),
                "conv_layers": (4, int, (1, 32)),
                "kernel": (5, int, (3, 11)),
                "dense_dropout": (0, float, (0, 1)),
                "conv_dropout": (0, float, (0, 1)),
                "conv_dropout_after": (3, int, (1, 32)),
                "use_gap": (True, bool, None),
                "use_bn": (False, bool, None),
                "dense_layers": (2, int, (0, 4)),
                "activation": (
                    "prelu", str, set(activation_mapping.keys())),
                "pool_size": (2, int, (1, 4)),
                "dense_units": (200, int, (100, 400))
            }
        )
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})
        self.performance = dict()
        self.channels = channels
        self.n_models = num_models
        # Needs to be called as the last line in the subclass.
        self.n_train_images = n_train_images
        # if hyperoptimize:
        # self._seq_create_model()
        # else:
        self.create_model()
        # self.bayesian_vgg((None, None, self.channels))

    def _seq_create_model(self):
        """Made for keras hyperopt."""
        # TODO Fix ensembling thing and delete this.
        # Constrained convolution with a learned residual filter
        # layers.append(ConstrainedConv2D())
        # Standard convolutional layers
        filters = self._h.filters
        kl_divergence_function = (
            lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(
                self.n_train_images, dtype=tf.float32)
        )
        layers = list()
        for j in range(self._h.conv_layers):
            layers.append(
                tfp.layers.Convolution2DFlipout(filters,
                                                kernel_size=self._h.kernel,
                                                padding='same',
                                                activation=self.activation,
                                                # kernel_divergence_fn=kl_divergence_function,
                                                # kernel_posterior_fn=kernel_posterior_fn
                                                )
            )
            if self._h.use_bn:
                layers.append(tf.keras.layers.BatchNormalization())
            if self._h.conv_dropout > 0 and j + 1 >= self._h.conv_dropout_after:
                layers.append(
                    tf.keras.layers.SpatialDropout2D(self._h.conv_dropout))
            layers.append(tf.keras.layers.MaxPool2D(self._h.pool_size))
            filters = int(filters * self._h.filter_multiplier)
        # Final 1 x 1 convolution

        layers.append(
            tfp.layers.Convolution2DFlipout(
                int(filters // self._h.filter_multiplier),
                kernel_size=1, padding='same',
                activation=self.activation,
                # kernel_divergence_fn=kl_divergence_function,
            )
        )
        # GAP / Feature formation
        layers.append(tf.keras.layers.GlobalAveragePooling2D())

        # Fully-connected classifier
        for _ in range(self._h.dense_layers):
            layers.append(
                self.dense(self._h.dense_units, activation=self.activation)
            )
            if self._h.dense_dropout > 0:
                layers.append(self.dropout(self._h.dense_dropout))

        # final classification head
        layers.append(self.dense(self._h.n_classes,
                                 activation=None))

        self._model = tf.keras.models.Sequential(layers)

    def _create_model(self):

        layers = []
        for i in range(self.n_models):
            layers.append([])
            # Constrained convolution with a learned residual filter
            layers[i].append(ConstrainedConv2D())
            # Standard convolutional layers
            filters = self._h.filters
            for j in range(self._h.conv_layers):
                layers[i].append(
                    self.conv2d(filters,
                                kernel_size=self._h.kernel,
                                padding='same',
                                activation=self.activation))
                if self._h.use_bn:
                    layers[i].append(tf.keras.layers.BatchNormalization())
                if self._h.conv_dropout > 0 and j + 1 >= self._h.conv_dropout_after:
                    layers[i].append(
                        tf.keras.layers.SpatialDropout2D(self._h.conv_dropout))
                layers[i].append(tf.keras.layers.MaxPool2D(self._h.pool_size))
                filters = int(filters * self._h.filter_multiplier)

            # Final 1 x 1 convolution
            layers[i].extend([
                self.conv2d(int(filters // self._h.filter_multiplier),
                            kernel_size=1, padding='same',
                            activation=self.activation),
                # tf.keras.layers.SpatialDropout2D(self._h.conv_dropout),
            ])

            # GAP / Feature formation
            if self._h.use_gap:
                layers[i].append(tf.keras.layers.GlobalAveragePooling2D())
            else:
                layers[i].append(tf.keras.layers.Flatten())

            # Fully-connected classifier
            for _ in range(self._h.dense_layers):
                layers[i].append(
                    self.dense(self._h.dense_units, activation=self.activation)
                )
                if self._h.dense_dropout > 0:
                    layers[i].append(self.dropout(self._h.dense_dropout))

            # final classification head
            layers[i].append(self.dense(self._h.n_classes, activation=None))

        inputs = Input(shape=(None, None, self.channels))
        outputs_list = []

        for i in range(self.n_models):
            outputs = inputs

            for layer in layers[i]:
                outputs = layer(outputs)

            outputs_list.append(outputs)

        if self.n_models == 1:
            outputs_list = outputs_list[0]

        self._model = tf.keras.models.Model(inputs, outputs_list)

    def reset_performance_stats(self):
        self.performance = {
            "loss": {"training": [], "validation": []},
            "accuracy": {"validation": []},
            "confusion": [],
        }

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

    @property
    def model_code(self):
        return "{c}_{k}x{k}_{l}x{f}f".format(
            c=self.class_name,
            k=self._h.kernel,
            f=self._h.n_features,
            l=self._h.n_layers,
        )

    def _vggconv_block(self, x, filters, kernel, stride, kernel_posterior_fn):
        """Network block for VGG."""
        out = tfp.layers.Convolution2DFlipout(
            filters,
            kernel,
            padding='same',
            kernel_posterior_fn=kernel_posterior_fn)(x)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('relu')(out)

        out = tfp.layers.Convolution2DFlipout(
            filters,
            kernel,
            padding='same',
            kernel_posterior_fn=kernel_posterior_fn)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('relu')(out)

        out = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=stride)(out)
        return out

    def bayesian_vgg(self, input_shape,
                     num_classes=10,
                     kernel_posterior_scale_mean=-9.0,
                     kernel_posterior_scale_stddev=0.1,
                     kernel_posterior_scale_constraint=0.2):

        """Constructs a VGG16 model.

        Args:
          input_shape: A `tuple` indicating the Tensor shape.
          num_classes: `int` representing the number of class labels.
          kernel_posterior_scale_mean: Python `int` number for the kernel
            posterior's scale (log variance) mean. The smaller the mean the closer
            is the initialization to a deterministic network.
          kernel_posterior_scale_stddev: Python `float` number for the initial kernel
            posterior's scale stddev.
            ```
            q(W|x) ~ N(mu, var),
            log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
            ````
          kernel_posterior_scale_constraint: Python `float` number for the log value
            to constrain the log variance throughout training.
            i.e. log_var <= log(kernel_posterior_scale_constraint).

        Returns:
          tf.keras.Model.
        """

        filters = [64, 128, 128, 256]
        kernels = [3, 3, 3, 3]
        strides = [2, 2, 2, 2]

        def _untransformed_scale_constraint(t):
            return tf.clip_by_value(t, -1000,
                                    tf.math.log(
                                        kernel_posterior_scale_constraint))

        kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
            untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
                mean=kernel_posterior_scale_mean,
                stddev=kernel_posterior_scale_stddev),
            untransformed_scale_constraint=_untransformed_scale_constraint)

        image = tf.keras.layers.Input(shape=input_shape, dtype='float32')

        x = image
        for i in range(len(kernels)):
            x = self._vggconv_block(
                x,
                filters[i],
                kernels[i],
                strides[i],
                kernel_posterior_fn)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tfp.layers.DenseFlipout(
            11, activation=None,
            kernel_posterior_fn=kernel_posterior_fn)(x)
        self._model = tf.keras.Model(inputs=image, outputs=x, name='vgg16')
        # return model