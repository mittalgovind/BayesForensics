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
from loguru import logger
import matplotlib.pyplot as plt


# Internal libraries


class TemperatureScaling(tf.keras.Model, ABC):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.temperature = tf.Variable(1, trainable=True, dtype=tf.float32)
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
            try:
                labels = batch.labels
            except:
                raise NotImplementedError(
                    "The dataset has no labels attribute. "
                    "Override this method to supply them."
                )
            return batch, labels
        else:
            return batch

    @staticmethod
    def plot_conf(ece, acc, conf, title="init"):
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
        ax.plot([0, 1], [0, 1], "k--")
        ax.plot(conf, acc, marker=".")
        ax.set_xlabel(r"confidence")
        ax.set_ylabel(r"accuracy")
        ax.set_xticks((np.arange(0, 1.1, step=0.2)))
        ax.set_yticks((np.arange(0, 1.1, step=0.2)))

        textstr_freq_ts = "ECE={:.2f}".format(ece * 100)
        props = dict(boxstyle="round", facecolor="white", alpha=0.75)
        ax.text(
            0.075,
            0.925,
            textstr_freq_ts,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=props,
        )
        ax.set_title(r" TS - {}".format(title))
        fig.tight_layout()
        fig.show()
        return fig, ax

    def set_temp(self, data):
        """Use validation dataset to calibrate the model."""
        logits_list = []
        labels_list = []
        # TODO remove hard coding
        n_batches = data.count_validation // self.batch_size
        epochs = 35
        nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.optimizers.Adam(learning_rate=0.01)

        # Before training
        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, self.batch_size)
            batch, labels = self.preprocess(batch, return_labels=True)
            logits_list.append(self.model(batch, training=False))
            labels_list.append(labels)

        logits = np.stack(logits_list).reshape((n_batches * self.batch_size, -1))
        labels = np.stack(labels_list).reshape((n_batches * self.batch_size,))

        init_nll_loss = nll_loss(labels, logits)
        init_ece_loss, init_acc_list, init_conf_list = self.ece_loss(labels, logits)
        self.plot_conf(init_ece_loss, init_acc_list, init_conf_list, "init")
        for epoch in range(epochs):
            print(self.temperature)
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
        final_ece_loss, final_acc_list, final_conf_list = self.ece_loss(
            labels, self.temperature_scale(logits)
        )
        self.plot_conf(final_ece_loss, final_acc_list, final_conf_list, "final")
        logger.info("NLL Loss diff = {:.6f}".format(final_nll_loss - init_nll_loss))
        logger.info("ECE Loss diff = {:.6f}".format((final_ece_loss - init_ece_loss)))
        return

    @staticmethod
    def ece_loss(labels, logits, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        softmaxes = np.array(tf.nn.softmax(logits, axis=1))
        confidences = np.max(softmaxes, axis=1)
        predictions = np.argmax(softmaxes, axis=1).astype(np.int32)
        accuracies = np.where(labels == predictions, 1, 0)

        ece = 0
        acc_bin_list = list()
        avg_conf_list = list()
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            lower_bounded = np.where(confidences > bin_lower, True, False)
            upper_bounded = np.where(confidences <= bin_upper, True, False)
            in_bin = lower_bounded * upper_bounded
            proportion_in_bin = in_bin.mean()
            if proportion_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += proportion_in_bin * np.abs(
                    avg_confidence_in_bin - accuracy_in_bin
                )
                acc_bin_list.append(accuracy_in_bin)
                avg_conf_list.append(avg_confidence_in_bin)

        return ece, acc_bin_list, avg_conf_list
