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
    data.set_eval_mode()
    q_factors = np.arange(*data.qf_test)
    n_batches = data.count_validation // batch_size
    accuracies = 0

    if cache:
        try:
            performance = cache.load()
            performance["loss"]["validation"] = []
            performance["accuracy"]["validation"] = []
        except:
            performance = {"loss": {"validation": []},
                           "accuracy": {"validation": []}}
            logger.warning(
                "performance cache from training could not be loaded. Making a new one."
            )

    for QF1, QF2 in progress_bar(product(q_factors, repeat=2)):
        QF1, QF2 = int(QF1), int(QF2)
        for batch in data.get_validation_generator(batch_size=batch_size):
            images, labels = data.preprocess_batch(batch, QF1=QF1, QF2=QF2)

            # TODO (Marcelo) Add validation for MC dropout
            # get logits, calibrate and calculate predictions.
            logits = model(images, training=False)
            calibrated_logits = logits / model.temperature
            predictions = calibrated_logits.numpy().argmax(axis=1)

            accuracies += np.sum(predictions == labels)

    accuracies /= data.count_validation

    if cache:
        performance["accuracy"]["validation"].append(accuracies / n_batches)
        cache.save(performance, step="performance")

    return accuracies
