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
from loguru import logger

# Internal libraries
from helpers.utils import progress_bar


def validate(model, data, batch_size, cache):
    """

    Parameters
    ----------
    model : BayesModel()
        Bayes model for running tests
    data
    batch_size
    num_runs
    cache
    """
    q_factors = np.arange(*data.qf_test)
    n_factors = len(q_factors)
    batch_size //= 2
    n_batches = data.count_validation // batch_size
    counters = np.zeros((4, n_factors, n_factors))

    if cache:
        try:
            performance = cache.load()
            performance["loss"]["validation"] = []
            performance["accuracy"]["validation"] = []
        except:
            performance = {"loss": {"validation": []}, "accuracy": {"validation": []}}
            logger.warning(
                "performance cache from training could not be loaded. Making a new one."
            )

    for QF1, QF2 in progress_bar(product(q_factors, repeat=2)):

        # index of corresponding QF in the counters array.
        qf1 = np.argmax(q_factors == QF1)
        qf2 = np.argmax(q_factors == QF2)

        # skipping if an image has a history of stronger or equal compression

        QF1, QF2 = int(QF1), int(QF2)
        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, batch_size)
            images, labels = data.preprocess_batch(batch, QF1=QF1, QF2=QF2)

            # TODO (Marcelo) Add validation for MC dropout
            # get logits, calibrate and calculate predictions.
            logits = model(images, training=False)
            calibrated_logits = logits / model.temperature
            predictions = calibrated_logits.numpy().argmax(axis=1)

            # Counter for True negatives, negatives, True positives, positives.
            counters[0, qf1, qf2] += np.sum(predictions[:batch_size] == 0)
            counters[1, qf1, qf2] += batch_size
            counters[2, qf1, qf2] += np.sum(predictions[batch_size:] == 1)
            counters[3, qf1, qf2] += batch_size

    tnr = counters[0] / counters[1]
    tpr = counters[2] / counters[3]
    accuracies = (tnr + tpr) / 2

    if cache:
        performance["accuracy"]["validation"].append(accuracies / n_batches)
        cache.save(performance, step="performance")

    return accuracies
