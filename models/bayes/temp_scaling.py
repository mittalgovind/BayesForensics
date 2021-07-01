#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
from abc import ABC
import os

# External libraries
import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import dtype_util
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import progressbar

# Internal libraries


class TemperatureScaling(ABC):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def calibrate(self, epochs, data, num_classes, opt, loss, strategy):
        if strategy:
            strategy.run(self.single_calibrate,
                         args=(data, num_classes, opt, loss))
        else:
            self.single_calibrate(epochs, data, num_classes, opt, loss)

    def single_calibrate(self, epochs, data, num_classes, opt, loss):
        self.temperature = tf.Variable(1, trainable=True, dtype=tf.float32)
        with progressbar.ProgressBar(widgets=[progressbar.Variable('ECE_Loss')]) as bar:
            for i in bar(range(epochs)):
                if loss < 0.01 or self.temperature < 0.1:
                    break
                total_loss = 0.0
                runs = 0
                for images, labels in data.get_calibration_generator():
                    logits = tf.cast(self._model(images, training=False),
                                     tf.float32)
                    labels = tf.cast(labels, tf.int32)
                    with tf.GradientTape() as tape:
                        tape.watch(self.temperature)
                        calibrated_logits = logits / self.temperature
                        loss = self.expected_calibration_error(num_classes,
                                                               calibrated_logits,
                                                               labels)
                    grads = [tape.gradient(loss, self.temperature)]
                    opt.apply_gradients(zip(grads, [self.temperature]))
                    total_loss += loss
                    runs += 1
                bar.update(i, ECE_Loss=total_loss / runs)

    def set_temp(self, data, save_dir=None, strategy=None, epochs=100, lr=1e-3):
        """Use validation dataset to calibrate the model."""
        logits_list = []
        labels_list = []
        nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        # Before training
        for images, labels in data.get_calibration_generator():
            logits_list.append(self._model(images, training=False))
            labels_list.append(labels)
        num_classes = len(logits_list[0][0])

        init_logits = tf.stack(logits_list)
        init_logits = tf.cast(tf.reshape(init_logits, (-1, num_classes)),
                              dtype=tf.float32)

        init_labels = tf.stack(labels_list)
        init_labels = tf.cast(tf.reshape(init_labels, -1), dtype=tf.int32)

        init_nll_loss = nll_loss(init_labels, init_logits)
        init_ece_loss, init_acc, init_conf = self.expected_calibration_error(
            num_classes, init_logits, init_labels, return_conf=True)
        self.plot_conf(init_ece_loss, init_acc, init_conf, save_dir,
                       title="init")

        # train to find temperature
        opt = tf.optimizers.Adam(learning_rate=lr)
        self.calibrate(epochs, data, num_classes, opt, init_ece_loss, strategy)

        final_nll_loss = nll_loss(init_labels, init_logits / self.temperature)
        final_ece_loss, final_acc, final_conf = self.expected_calibration_error(
            num_classes, init_logits / self.temperature, init_labels,
            return_conf=True
        )
        logger.info(
            'Calibrated! Temperature set to {}'.format(self.temperature))
        logger.info(
            "NLL Loss diff = {:.6f}".format(final_nll_loss - init_nll_loss))
        logger.info(
            "ECE Loss diff = {:.6f}".format((final_ece_loss - init_ece_loss)))
        self.plot_conf(final_ece_loss, final_acc, final_conf, save_dir,
                       "final")

    def expected_calibration_error(self, num_bins, logits=None,
                                   labels_true=None,
                                   labels_predicted=None, name=None,
                                   return_conf=False):
        """Compute the Expected Calibration Error (ECE).
        This method implements equation (3) in [1].  In this equation the probability
        of the decided label being correct is used to estimate the calibration
        property of the predictor.
        Note: a trade-off exist between using a small number of `num_bins` and the
        estimation reliability of the ECE.  In particular, this method may produce
        unreliable ECE estimates in case there are few samples available in some bins.
        As an alternative to this method, consider also using
        `bayesian_expected_calibration_error`.
        #### References
        [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
             On Calibration of Modern Neural Networks.
             Proceedings of the 34th International Conference on Machine Learning
             (ICML 2017).
             arXiv:1706.04599
             https://arxiv.org/pdf/1706.04599.pdf
        Args:
          num_bins: int, number of probability bins, e.g. 10.
          logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
          labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
            ground truth class labels in the range [0,nlabels].
          labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
            containing decisions of the predictive system.  If `None`, we will use
            the argmax decision using the `logits`.
          name: Python `str` name prefixed to Ops created by this function.
        Returns:
          ece: Tensor, scalar, tf.float32.
        """
        with tf.name_scope(name or 'expected_calibration_error'):
            logits = tf.convert_to_tensor(logits)
            labels_true = tf.convert_to_tensor(labels_true)
            if labels_predicted is not None:
                labels_predicted = tf.convert_to_tensor(labels_predicted)

            # Compute empirical counts over the events defined by the sets
            # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
            # of predicted probabilities in each probability bin.
            event_bin_counts, pmean_observed = self._compute_calibration_bin_statistics(
                num_bins, logits=logits, labels_true=labels_true,
                labels_predicted=labels_predicted)

            # Compute the marginal probability of observing a probability bin.
            event_bin_counts = tf.cast(event_bin_counts, tf.float32)
            bin_n = tf.reduce_sum(event_bin_counts, axis=0)
            pbins = bin_n / tf.reduce_sum(
                bin_n)  # Compute the marginal bin probability

            # Compute the marginal probability of making a correct decision given an
            # observed probability bin.
            tiny = np.finfo(np.float32).tiny
            pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

            # Compute the ECE statistic as defined in reference [1].
            ece = tf.reduce_sum(pbins * tf.abs(pcorrect - pmean_observed))

            if return_conf:
                return ece, pcorrect, pmean_observed
            else:
                return ece

    @staticmethod
    def _compute_calibration_bin_statistics(
            num_bins, logits=None, labels_true=None, labels_predicted=None):
        """Compute binning statistics required for calibration measures.
        Args:
          num_bins: int, number of probability bins, e.g. 10.
          logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
          labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
            ground truth class labels in the range [0,nlabels].
          labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
            containing decisions of the predictive system.  If `None`, we will use
            the argmax decision using the `logits`.
        Returns:
          bz: Tensor, shape (2,num_bins), tf.int32, counts of incorrect (row 0) and
            correct (row 1) predictions in each of the `num_bins` probability bins.
          pmean_observed: Tensor, shape (num_bins,), tf.float32, the mean predictive
            probabilities in each probability bin.
        """

        if labels_predicted is None:
            # If no labels are provided, we take the label with the maximum probability
            # decision.  This corresponds to the optimal expected minimum loss decision
            # under 0/1 loss.
            pred_y = tf.argmax(logits, axis=1, output_type=labels_true.dtype)
        else:
            pred_y = labels_predicted

        correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)

        # Collect predicted probabilities of decisions
        pred = tf.nn.softmax(logits, axis=1)
        prob_y = tf.gather(
            pred, pred_y[:, tf.newaxis], batch_dims=1)  # p(pred_y | x)
        prob_y = tf.reshape(prob_y, (ps.size(prob_y),))

        # Compute b/z histogram statistics:
        # bz[0,bin] contains counts of incorrect predictions in the probability bin.
        # bz[1,bin] contains counts of correct predictions in the probability bin.
        bins = tf.histogram_fixed_width_bins(prob_y, [0.0, 1.0],
                                             nbins=num_bins)
        event_bin_counts = tf.math.bincount(
            correct * num_bins + bins,
            minlength=2 * num_bins,
            maxlength=2 * num_bins)
        event_bin_counts = tf.reshape(event_bin_counts, (2, num_bins))

        # Compute mean predicted probability value in each of the `num_bins` bins
        pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, num_bins)
        tiny = np.finfo(dtype_util.as_numpy_dtype(logits.dtype)).tiny
        pmean_observed = pmean_observed / (
                tf.cast(tf.reduce_sum(event_bin_counts, axis=0),
                        logits.dtype) + tiny)

        return event_bin_counts, pmean_observed

    @staticmethod
    def plot_conf(ece, acc, conf, save_dir=None, title="init"):
        acc = np.array(acc)
        conf = np.array(conf)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot([0, 1], [0, 1], "k--")
        plt.bar(conf, acc, 1 / (len(conf) * 1.1))
        ax.set_xlabel(r"confidence")
        ax.set_ylabel(r"accuracy")
        ax.set_xticks((np.arange(0, 1.1, step=0.2)))
        ax.set_yticks((np.arange(0, 1.1, step=0.2)))

        textstr_freq_ts = "ECE={:.6f}".format(ece)
        props = dict(boxstyle="round", facecolor="white", alpha=0.75)
        ax.text(
            0.075,
            0.925,
            textstr_freq_ts,
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=props,
        )
        ax.set_title(r" TS - {}".format(title))
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'ts-{}.png'.format(title)))
        else:
            plt.show()
