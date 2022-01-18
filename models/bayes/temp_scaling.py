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
import tensorflow_probability as tfp
from loguru import logger
from tqdm import tqdm
import functools


# Internal libraries


def _make_val_and_grad_fn(value_fn):
    @functools.wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)

    return val_and_grad


class TemperatureScaling(ABC):
    """Decorator for wrapping a TensorFlow model with temperature scaling."""

    def set_temperature(self, data, save_dir=None):
        logits_list = []
        labels_list = []
        if data.n_classes == 2:
            nll_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)

        # Before training
        for batch_x, batch_y in tqdm(data.get_calibration_generator()):
            probabilities = self._model(batch_x, training=False)

            logits_list.append(probabilities)
            labels_list.append(batch_y)

        num_classes = data.n_classes

        logits_list = tf.stack(logits_list)
        labels_list = tf.stack(labels_list)

        logits_list = tf.cast(tf.reshape(logits_list, (-1, num_classes)),
                              dtype=tf.float32)
        labels_list = tf.cast(tf.reshape(labels_list, -1), dtype=tf.int32)

        if data.n_classes == 2:
            base_nll = nll_loss(labels_list, logits_list[..., 1])
        else:
            base_nll = nll_loss(labels_list, logits_list[..., 1])
        base_ece = tfp.stats.expected_calibration_error(10, logits=logits_list,
                                                        labels_true=labels_list)

        logger.info(
            f'Original NLL: {base_nll.numpy():.3f}, Original ECE: {base_ece.numpy():.3f}')

        @_make_val_and_grad_fn
        def temp_scale_loss(x):
            if data.n_classes == 2:
                return nll_loss(labels_list, logits_list[..., 1] / x)
            else:
                return nll_loss(labels_list, logits_list / x)

        xinit = tf.constant([self.temperature])
        results = tfp.optimizer.lbfgs_minimize(temp_scale_loss,
                                               initial_position=xinit)
        temp = tf.squeeze(results.position)

        if data.n_classes == 2:
            new_nll = nll_loss(labels_list, logits_list[..., 1] / temp)
        else:
            new_nll = nll_loss(labels_list, logits_list / temp)

        new_ece = tfp.stats.expected_calibration_error(15,
                                                       logits=logits_list / temp,
                                                       labels_true=labels_list)

        logger.info(
            f'Resulting NLL: {new_nll.numpy():.3f}, Resulting ECE: {new_ece.numpy():.3f}')
        logger.info(f'Resulting temperature: {temp.numpy():.3f}')

        self.temperature = temp

        if save_dir:
            f = open(os.path.join(save_dir, "temperature.txt"), "w")
            f.write(f"{temp.numpy():.4f}")
            f.close()
