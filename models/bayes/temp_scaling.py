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


class TemperatureScaling(ABC):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def set_temp(self, data):
        """Use validation dataset to calibrate the model."""
        self.temperature = tf.Variable(1, trainable=True, dtype=tf.float32)
        logits_list = []
        labels_list = []
        nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.optimizers.Adam(learning_rate=0.001)
        epochs = 100

        # Before training
        for images, labels in data.get_validation_generator():
            logits_list.append(self._model(images, training=False))
            labels_list.append(labels)

        logits = tf.stack(logits_list)
        logits = tf.reshape(logits, (data.count_validation, -1))

        labels = tf.stack(labels_list)
        labels = tf.reshape(labels, (data.count_validation, -1))
        # labels = tf.cast(labels, tf.int64)

        init_nll_loss = nll_loss(labels, logits)
        init_ece_loss, init_acc_list, init_conf_list = self.ece_loss(labels, logits)
        # self.plot_conf(init_ece_loss, init_acc_list, init_conf_list, "init")

        # train to find temperature
        for epoch in range(epochs):
            print(self.temperature)
            for images, labels in data.get_calibration_generator():
                logits = self._model(images, training=False)

                with tf.GradientTape() as tape:
                    tape.watch(self.temperature)
                    loss, _, _ = self.ece_loss(labels, logits / self.temperature)

                grads = [tape.gradient(loss, self.temperature)]
                opt.apply_gradients(zip(grads, [self.temperature]))

        final_nll_loss = nll_loss(labels, logits / self.temperature)
        final_ece_loss, final_acc_list, final_conf_list = self.ece_loss(
            labels, logits / self.temperature
        )

        # self.plot_conf(final_ece_loss, final_acc_list, final_conf_list, "final")
        logger.info("NLL Loss diff = {:.6f}".format(final_nll_loss - init_nll_loss))
        logger.info("ECE Loss diff = {:.6f}".format((final_ece_loss - init_ece_loss)))

    @staticmethod
    def ece_loss(labels, logits, n_bins=5):
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

        return tf.convert_to_tensor(ece), acc_bin_list, avg_conf_list


    # # @tf.function(experimental_compile=True)
    # @staticmethod
    # def ece_loss(labels, logits, n_bins=5):
    #     bin_boundaries = tf.cast(tf.linspace(0, 1, n_bins + 1), tf.float32)
    #     bin_lowers = bin_boundaries[:-1]
    #     bin_uppers = bin_boundaries[1:]
    #     softmaxes = tf.nn.softmax(logits, axis=1)
    #     confidences = tf.cast(
    #         tf.experimental.numpy.max(softmaxes, axis=1), tf.float32)
    #     predictions = tf.cast(tf.math.argmax(softmaxes, axis=1), tf.float32)
    #     accuracies = tf.where(labels == predictions, 1, 0)
    #
    #     ece = tf.zeros(1, dtype=tf.float32)
    #     acc_bin_list = list()
    #     avg_conf_list = list()
    #     for i in range(bin_lowers.shape[0]):
    #         lower_bounded = tf.where(confidences > bin_lowers[i], 1, 0)
    #         upper_bounded = tf.where(confidences <= bin_uppers[i], 1, 0)
    #         in_bin = lower_bounded * upper_bounded
    #         proportion_in_bin = tf.cast(tf.math.reduce_mean(in_bin), tf.float32)
    #         indices_bin = tf.where(in_bin)
    #         if tf.greater(proportion_in_bin, 0):
    #             accuracy_in_bin = tf.cast(tf.math.reduce_mean(
    #                 tf.gather(accuracies, indices_bin)), tf.float32)
    #             avg_confidence_in_bin = tf.cast(tf.math.reduce_mean(
    #                 tf.gather(confidences, indices_bin)), tf.float32)
    #             ece = tf.math.add(ece, proportion_in_bin * tf.math.abs(
    #                 avg_confidence_in_bin - accuracy_in_bin
    #             ))
    #             acc_bin_list.append(accuracy_in_bin)
    #             avg_conf_list.append(avg_confidence_in_bin)
    #
    #     return ece, acc_bin_list, avg_conf_list

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
