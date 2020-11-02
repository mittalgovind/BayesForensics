#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
from abc import abstractmethod, ABC

# External libraries
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from loguru import logger


# Internal libraries


class TemperatureScaling(tf.keras.Model, ABC):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.temperature = tf.Variable(1.5, trainable=True, dtype=tf.float32)
        self.batch_size = batch_size

    def forward(self, inputs, training):
        logits = self.model(inputs, training=training)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """Perform temp scaling on logits"""
        return logits / self.temperature

    def preprocess(self, batch, return_labels=False):
        """Override method to preprocess batch before forward pass."""
        if return_labels:
            labels = batch.labels  # this wont work!
            return batch, labels
        else:
            return batch

    def set_temp(self, data):
        """Use validation dataset to calibrate the model."""
        logits_list = []
        labels_list = []
        # TODO remove hard coding
        n_batches = 2
        nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        opt = tf.optimizers.Adam(learning_rate=0.01)

        # Before training
        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, self.batch_size)
            batch, labels = self.preprocess(batch, return_labels=True)
            logits_list.append(self.model(batch, training=False))
            labels_list.append(labels)

        logits = np.stack(logits_list).reshape(
            (n_batches * self.batch_size, -1))
        labels = np.stack(labels_list).reshape(
            (n_batches * self.batch_size, -1))

        init_nll_loss = nll_loss(labels, logits)
        init_ece_loss = self.ece_loss(labels, logits)

        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, self.batch_size)
            batch, labels = self.preprocess(batch, return_labels=True)
            logits = self.model(batch, training=False)

            with tf.GradientTape() as tape:
                tape.watch(self.temperature)
                loss = nll_loss(labels, self.temperature_scale(logits))

            grads = [tape.gradient(loss, self.temperature)]
            opt.apply_gradients(zip(grads, [self.temperature]))

        final_nll_loss = nll_loss(labels, self.temperature_scale(logits))
        final_ece_loss = self.ece_loss(labels, self.temperature_scale(logits))
        logger.info(
            'NLL Loss diff = {:.6f}'.format(final_nll_loss - init_nll_loss))
        logger.info('ECE Loss diff = {:.6f}'.format(
            (final_ece_loss - init_ece_loss)[0]))
        return

    @staticmethod
    def ece_loss(labels, logits, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[: -1]
        bin_uppers = bin_boundaries[1:]
        softmaxes = np.array(tf.nn.softmax(logits, axis=1))
        confidences = np.max(softmaxes, axis=1)
        predictions = np.argmax(softmaxes, axis=1).astype(np.int32)
        accuracies = np.array(tf.metrics.binary_accuracy(labels, predictions))

        ece = tf.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            lower_bounded = np.where(confidences > bin_lower, True, False)
            upper_bounded = np.where(confidences <= bin_upper, True, False)
            in_bin = lower_bounded * upper_bounded
            proportion_in_bin = in_bin.mean()
            if proportion_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += proportion_in_bin * np.abs(
                    avg_confidence_in_bin - accuracy_in_bin)

        return ece
