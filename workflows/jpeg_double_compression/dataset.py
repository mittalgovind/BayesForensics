#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
from itertools import product

# External libraries
import tensorflow as tf
import numpy as np
from loguru import logger
import pywt
from tqdm import tqdm

# Internal libraries
from helpers.tf_dataset import Dataset
from helpers.loading import randint
from helpers.tf_jpeg import TFJPEG

try:
    from prnulib import commons
except RuntimeError:
    raise RuntimeError(
        "Could NOT load prnulib. Run git submodule init & git submodule update ")


class DoubleCompressionDataset(Dataset):
    def __init__(self, codec, qf_train, qf_test, calc_pywt_residual=False,
                 per_batch_sub=128, **kwargs):
        """
        Subclass of helpers.dataset.Dataset class.

        codec : object (models.jpeg)
            Code for compressing image patches
        qf_train : tuple
            Range of quality factor to choose for compressing during training.
        qf_test : tuple
            Range of quality factor to choose for compressing during testing.
        calc_pywt_residual : bool
            Flag for calculate PyWavelet residual
        """
        super().__init__(**kwargs)

        qf_train = qf_train.split(",")
        if len(qf_train) == 2:
            qf_train = (int(qf_train[0]), int(qf_train[1]))
        elif len(qf_train) == 3:
            qf_train = (int(qf_train[0]), int(qf_train[1]), int(qf_train[2]))
        else:
            logger.error("Invalid training quality factor range")
        self.qf_train = tf.convert_to_tensor(np.arange(*qf_train))
        self.len_qf_train = len(self.qf_train)
        self.calc_pywt_residual = calc_pywt_residual
        qf_test = qf_test.split(",")
        if len(qf_test) == 2:
            qf_test = (int(qf_test[0]), int(qf_test[1]))
        elif len(qf_test) == 3:
            qf_test = (int(qf_test[0]), int(qf_test[1]), int(qf_test[2]))
        else:
            logger.error("Invalid testing quality factor range")
        self.qf_test = tf.convert_to_tensor(np.arange(*qf_test))
        self.len_qf_test = len(self.qf_test)

        self.codec = TFJPEG(codec=codec)
        if codec == 'libjpeg':
            logger.info('Using libjpeg will be slowing the computation.')

        self.eval_mode = False
        self.qf_val_pairs = [(q1, q2) for q1 in self.qf_train
                             for q2 in self.qf_train if q1 > q2]
        self.per_batch_sub = per_batch_sub

    def preprocess_batch(self, batch, **kwargs):
        if 'QF1' not in kwargs or 'QF2' not in kwargs:
            # sample quality factors
            QF1 = self.qf_train[
                randint(maxval=self.len_qf_train, seed=self.seed)]
            QF2 = self.qf_train[
                randint(maxval=self.len_qf_train, seed=self.seed)]
            while QF1 == QF2:
                QF2 = self.qf_train[randint(maxval=self.len_qf_train,
                                            seed=self.seed)]
        else:
            QF1 = kwargs["QF1"]
            QF2 = kwargs["QF2"]

        batch_single_compressed = self.codec.process(batch, QF2)
        # compressing with QF1 before QF2, to give compression history.
        batch_double_compressed = self.codec.process(
            self.codec.process(batch, QF1), QF2)
        images = tf.concat((batch_single_compressed, batch_double_compressed),
                           axis=0)
        if self.calc_pywt_residual:
            images = self.extract_pywt_residual(images)
        if tf.reduce_max(images) > 255:
            images = tf.math.divide(images, 255)
        labels = tf.concat((tf.zeros(self.batch_size),
                            tf.ones(self.batch_size)), axis=0)

        return images, labels

    @staticmethod
    def _noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 4):
        """
        NoiseExtract as from Binghamton toolbox.
        :param im: grayscale or color image, np.uint8 (shape =
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

            wlet = pywt.wavedec2(im[:, :, ch], "db4", level=levels)
            wlet_details = wlet[1:]

            wlet_details_filter = [None] * len(wlet_details)
            # Cycle over Wavelet levels 1:levels-1
            for wlet_level_idx, wlet_level in enumerate(wlet_details):
                # Cycle over H,V,D components
                level_coeff_filt = [None] * 3
                for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                    level_coeff_filt[
                        wlet_coeff_idx] = commons._wiener_adaptive(
                        wlet_coeff, noise_var
                    )
                wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

            # Set filtered detail coefficients for Levels > 0 ---
            wlet[1:] = wlet_details_filter

            # Set to 0 all Level 0 approximation coefficients ---
            wlet[0][...] = 0

            # Invert wavelet transform ---
            wrec = pywt.waverec2(wlet, "db4")
            W[:, :, ch] = wrec

        W = W.squeeze()
        W = W[: im.shape[0], : im.shape[1]]

        return W

    def extract_pywt_residual(self, batch):
        """Calculate and append an external filter to all the patches."""
        logger.info("Calculating PyWavelet residuals ...")
        color_F = np.array(
            [[0, 0.299, 0.587, 0.114], [128, -0.168736, -0.331264, 0.5],
             [128, 0.5, -0.418688, -0.081312]], dtype=np.float32)
        color_F = tf.reshape(tf.transpose(color_F), [1, 1, 4, 3])

        xc = tf.pad(batch, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT',
                    constant_values=1)
        ycbcrs = tf.nn.conv2d(xc, color_F, [1, 1, 1, 1], 'SAME').numpy()

        residuals = list()
        for ycbcr in tqdm(ycbcrs):
            residuals.append(self._noise_extract(ycbcr))
        return tf.convert_to_tensor(residuals)

    #
    # def extract_pywt_residual(self):
    #     """Calculate and append an external filter to all the patches."""
    #     logger.info("Calculating PyWavelet residuals ...")
    #     color_F = np.array(
    #         [[0, 0.299, 0.587, 0.114], [128, -0.168736, -0.331264, 0.5],
    #          [128, 0.5, -0.418688, -0.081312]], dtype=np.float32)
    #     color_F = tf.reshape(tf.transpose(color_F), [1, 1, 4, 3])
    #     for split in ["training", "validation", "calibration"]:
    #         if 'y' not in self.data[split].keys():
    #             continue
    #
    #         data = tf.cast(self.data[split]['y'], tf.float32)
    #         xc = tf.pad(data, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT',
    #                     constant_values=1)
    #         ycbcrs = tf.nn.conv2d(xc, color_F, [1, 1, 1, 1], 'SAME').numpy()
    #
    #         residuals = list()
    #         for ycbcr in tqdm(ycbcrs):
    #             residuals.append(self._noise_extract(ycbcr))
    #         residuals = np.array(residuals)
    #         self.data[split]["y"] = residuals
    #     logger.info("Residuals replaced each patch.")

    def get_training_pipeline(self, discard="flat"):
        return tf.data.Dataset.from_generator(
            self.get_training_generator,
            args=(discard,),
            output_signature=(
                tf.TensorSpec(
                    shape=(self.batch_size * 2, self.train_rgb_patch_size,
                           self.train_rgb_patch_size, self.channels),
                    dtype=tf.float32),
                tf.TensorSpec(shape=self.batch_size * 2, dtype=tf.float32)
            )
        )

    def get_validation_generator(self, **kwargs):
        batch = self.data["validation"]["y"][:self.per_batch_sub]
        for QF1, QF2 in self.qf_val_pairs:
            yield self.preprocess_batch(batch, QF1=QF1, QF2=QF2)

    def get_validation_pipeline(self):
        return tf.data.Dataset.from_generator(
            self.get_validation_generator,
            output_signature=(
                tf.TensorSpec(shape=(
                    self.batch_size * 2, self.valid_patch_size_rgb,
                    self.valid_patch_size_rgb, self.channels),
                    dtype=tf.float32),
                tf.TensorSpec(shape=self.batch_size * 2, dtype=tf.float32)
            )
        )

    def get_calibration_pipeline(self):
        return tf.data.Dataset.from_generator(
            self.get_calibration_generator,
            output_signature=(
                tf.TensorSpec(shape=(
                    self.batch_size * 2, self.calib_patch_size_rgb,
                    self.calib_patch_size_rgb, self.channels),
                    dtype=tf.float32),
                tf.TensorSpec(shape=self.batch_size * 2, dtype=tf.float32)
            )
        )
