#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf
import numpy as np
import os
from loguru import logger
import pywt

# Internal libraries
from helpers.utils import progress_bar
from helpers.plots import perf
from helpers.dataset import Dataset


class DoubleCompressionDataset(Dataset):
    def __init__(self, codec, qf_train, qf_test, calc_pywt_residual=False, **kwargs):
        """
        Subclass of helpers.dataset.Dataset class.

        codec : object (models.jpeg)
            Code for compressing image patches
        qf_train : tuple
            Range of quality factor to choose for compressing during training.
        qf_test : tuple
            Range of quality factor to choose for compressing during testing.
        calc_pywt_residual : bool
            Flag for calculate PyWavalet residual
        """
        super().__init__(presample_epochs=calc_pywt_residual, **kwargs)
        self.qf_train = qf_train
        self.qf_test = qf_test
        self.codec = codec
        if calc_pywt_residual:
            self.extract_pywt_residual()

    def preprocess_batch(self, batch, **kwargs):

        batch_size = len(batch)

        # sample quality factors
        QF1 = np.random.randint(low=self.qf_train[0], high=self.qf_train[1])
        QF2 = np.random.randint(low=self.qf_train[0], high=self.qf_train[1])
        while QF1 == QF2:
            QF2 = np.random.randint(low=self.qf_train[0], high=self.qf_train[1])
        batch_single_compressed = self.codec.process(batch, QF2)

        # compressing with QF1 before QF2, to give compression history.
        batch_double_compressed = self.codec.process(
            self.codec.process(batch, QF1), QF2)
        images = tf.concat((batch_single_compressed, batch_double_compressed),
                           axis=0)
        labels = tf.concat((tf.zeros(batch_size), tf.ones(batch_size)), axis=0)

        return images, labels

    @staticmethod
    def _noise_extract(im: np.ndarray, levels: int = 4,
                       sigma: float = 4) -> np.ndarray:
        """
        NoiseExtract as from Binghamton toolbox.
        :param im: grayscale or color image, np.uint8
        :param levels: number of wavelet decomposition levels
        :param sigma: estimated noise power
        :return: noise residual
        """

        im = im.astype(np.float32)

        noise_var = sigma ** 2

        if im.ndim == 2:
            im.shape += (1,)

        if im.shape[0] < 128 or im.shape[1] < 128:
            levels = 3

        W = np.zeros(im.shape, np.float32)

        for ch in range(im.shape[2]):

            wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            wlet_details = wlet[1:]

            wlet_details_filter = [None] * len(wlet_details)
            # Cycle over Wavelet levels 1:levels-1
            for wlet_level_idx, wlet_level in enumerate(wlet_details):
                # Cycle over H,V,D components
                level_coeff_filt = [None] * 3
                for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                    level_coeff_filt[
                        wlet_coeff_idx] = commons._wiener_adaptive(wlet_coeff,
                                                                   noise_var)
                wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

            # Set filtered detail coefficients for Levels > 0 ---
            wlet[1:] = wlet_details_filter

            # Set to 0 all Level 0 approximation coefficients ---
            wlet[0][...] = 0

            # Invert wavelet transform ---
            wrec = pywt.waverec2(wlet, 'db4')
            W[:, :, ch] = wrec

        W = W.squeeze()
        W = W[:im.shape[0], :im.shape[1]]

        return W

    def extract_pywt_residual(self):
        """Calculate and append an external filter to all the patches."""
        logger.info("Calculating PyWavelet residuals ...")
        for split in ["training", "validation", "calibration"]:
            xc = tf.pad(255.0 * self.data[split],
                        [[0, 0], [0, 0], [0, 0], [1, 0]],
                        'CONSTANT',
                        constant_values=1)
            ycbcrs = tf.nn.conv2d(xc,
                                  tf.reshape(tf.transpose(self._color_F),
                                             [1, 1, 4, 3]),
                                  [1, 1, 1, 1], 'SAME')
            ycbcrs = tf.cast(ycbcrs, dtype=tf.uint8)
            residuals = tf.tensor([self._noise_extract(ycbcr) for ycbcr in ycbcrs])
            self.data[split] = tf.concat(self.data[split], residuals, axis=-1)
        logger.info("Residuals appended to each patch.")


def train(
        model,
        epochs,
        data,
        batch_size,
        cache,
        qf,
        patch_size,
        lr,
        codec,
        save_every,
        save_dir,
        patience=150,
        min_delta=0.01,
        **kwargs
):
    performance = {"loss": {"training": []}, "accuracy": {"training": []}}
    # using half the batch size as it will be doubled after pre-processing batch
    batch_size //= 2
    n_batches = data.count_training // batch_size
    optimizer = tf.keras.optimizers.Adam(lr)
    # TODO make loss a argument
    loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    with progress_bar(epochs, "Training") as pbar:
        for epoch in range(epochs):
            losses = 0.0
            accuracies = 0.0

            for batch_id in range(n_batches):
                batch = data.next_training_batch(batch_id, batch_size,
                                                 patch_size)
                batch, labels = preprocess_batch(batch, codec, qf)
                with tf.GradientTape() as tape:
                    predictions = model(batch, training=True)
                    loss = loss_criterion(labels, predictions)

                grads = tape.gradient(loss, model._model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model._model.trainable_variables))

                losses += loss.numpy()
                accuracies += np.mean(
                    predictions.numpy().argmax(axis=-1) == labels)

            performance["loss"]["training"].append(losses / n_batches)
            performance["accuracy"]["training"].append(accuracies / n_batches)

            pbar.set_postfix(loss=losses / n_batches)
            pbar.update(1)

            if (epoch + 1) % save_every == 0:
                model.save_model(dirname=save_dir)
                fig = perf(performance, results="training")
                fig.savefig(
                    os.path.join(save_dir,
                                 "training_progress".format(epoch + 1))
                )

            if (epoch + 1) % patience == 0 and (
                    performance["loss"]["training"][-patience]
                    - performance["loss"]["training"][-1]
            ) < min_delta:
                logger.log(
                    1,
                    "Loss did not decrease by {} in {} epochs. Stopping training.".format(
                        min_delta, patience
                    ),
                )
                break

    if cache:
        cache.save(performance, step="performance")

    return performance
