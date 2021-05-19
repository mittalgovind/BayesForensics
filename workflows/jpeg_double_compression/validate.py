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
from helpers.uncertainty import get_pred, variation_ratio, predictive_entropy, mutual_information


def validate(model, data, batch_size, cache, num_runs):
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
    accuracies = np.zeros((len(q_factors), len(q_factors)))

    performance = None
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
        qf1_ind, qf2_ind = np.where(np.isclose(q_factors, QF1))[0][0], np.where(np.isclose(q_factors, QF2))[0][0]
        QF1, QF2 = int(QF1), int(QF2)
        for batch_id in range(n_batches):
            batch = data.next_validation_batch(batch_id, batch_size)
            images, labels = data.preprocess_batch(batch, QF1=QF1, QF2=QF2)

            if model.method == "vanilla":
                logits = model(images, training=False) / model.temperature
                predictions = logits.numpy().argmax(axis=1)

            else:
                if model.method == "ensemble":
                    logits = model(images, training=False) / model.temperature

                else:
                    logits = tf.convert_to_tensor([model(images, training=False) for _ in range(len(num_runs))])

                predictions = get_pred(logits)
            accuracies[qf1_ind, qf2_ind] += np.sum(predictions == labels)

    accuracies /= data.count_validation

    print(accuracies)

    if cache:
        performance["accuracy"]["validation"].append(accuracies / n_batches)
        cache.save(performance, step="performance")

    return accuracies
