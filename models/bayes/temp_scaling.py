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
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

# Internal libraries


class TemperatureScaling(ABC):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def set_temp(self, data, epochs=100, lr=1e-4):
        """Use validation dataset to calibrate the model."""
        self.temperature = tf.Variable(1, trainable=True, dtype=tf.float32)
        logits_list = []
        labels_list = []
        nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.optimizers.Adam(learning_rate=lr)

        # Before training
        for images, labels in data.get_calibration_generator():
            logits_list.append(self._model(images, training=False))
            labels_list.append(labels)
        num_classes = len(logits_list[0][0])

        init_logits = tf.stack(logits_list)
        init_logits = tf.cast(tf.reshape(init_logits, (-1, num_classes)), dtype=tf.float32)

        init_labels = tf.stack(labels_list)
        init_labels = tf.cast(tf.reshape(init_labels, -1), dtype=tf.int32)

        init_nll_loss = nll_loss(init_labels, init_logits)
        init_ece_loss = tfp.stats.expected_calibration_error(
            num_classes, init_logits, init_labels)

        # train to find temperature
        loss = init_ece_loss
        for epoch in range(epochs):
            if loss < 0.01 or self.temperature < 0.1:
                break
            for images, labels in data.get_calibration_generator():
                logits = tf.cast(self._model(images, training=False), tf.float32)
                labels = tf.cast(labels, tf.int32)
                with tf.GradientTape() as tape:
                    tape.watch(self.temperature)
                    loss = tfp.stats.expected_calibration_error(num_classes, logits / self.temperature, labels)
                grads = [tape.gradient(loss, self.temperature)]
                opt.apply_gradients(zip(grads, [self.temperature]))

        final_nll_loss = nll_loss(init_labels, init_logits / self.temperature)
        final_ece_loss = tfp.stats.expected_calibration_error(
            num_classes, init_logits / self.temperature, init_labels
        )

        logger.info('Calibrated! Temperature set to {}'.format(self.temperature))
        # self.plot_conf(final_ece_loss, final_acc_list, final_conf_list, "final")
        logger.info("NLL Loss diff = {:.6f}".format(final_nll_loss - init_nll_loss))
        logger.info("ECE Loss diff = {:.6f}".format((final_ece_loss - init_ece_loss)))

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
