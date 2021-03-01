#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
from itertools import product
import os

# External libraries
import tensorflow as tf
import numpy as np

# Internal libraries
from helpers.utils import progress_bar


def run_tests(model, qf, data, batch_size, save_dir,
              patch_size, codec, temperature, cache):
    """

    Parameters
    ----------
    model : BayesModel()
        Bayes model for running tests
    qf
    data
    batch_size
    patch_size
    num_runs
    cache
    temperature
    """
    q_factors = np.arange(*qf)
    n_factors = len(q_factors)
    n_batches = data.count_validation // batch_size
    counters = np.zeros((4, n_factors, n_factors))

    if cache:
        try:
            performance = cache.load(os.path.join(save_dir, "performance.npz"))
            performance["loss"]["validation"] = []
            performance["accuracy"]["validation"] = []
        except RuntimeWarning:
            performance = {"loss": {"validation": []},
                           "accuracy": {"validation": []}}
            raise RuntimeWarning(
                "performance cache from training could not be loaded. Making a new one.")

    for QF1, QF2 in progress_bar(product(q_factors, repeat=2)):
        QF1, QF2 = int(QF1), int(QF2)
        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, batch_size)
            batch_single_compressed = codec.process(batch, QF2)
            batch_double_compressed = codec.process(codec.process(batch, QF1),
                                                    QF2)

            images = tf.concat(
                (batch_single_compressed, batch_double_compressed), axis=0
            )
            del batch, batch_single_compressed, batch_double_compressed
            predictions = model(images, training=False).numpy().argmax(axis=1)

            qf1 = np.argmax(q_factors == QF1)
            qf2 = np.argmax(q_factors == QF2)

            # Counter for True negatives, negatives, True positives, positives.
            counters[0, qf2, qf1] += np.sum(predictions[:batch_size] == 0)
            counters[1, qf2, qf1] += batch_size
            counters[2, qf2, qf1] += np.sum(predictions[batch_size:] == 1)
            counters[3, qf2, qf1] += batch_size
        tnr = counters[0] / counters[1]
        tpr = counters[2] / counters[3]
        accuracies = (tnr + tpr) / 2

        performance["accuracy"]["validation"].append(accuracies / n_batches)

    if cache:
        cache.save(performance, step="performance")

    return (
        tnr,
        tpr,
        accuracies,
    )
