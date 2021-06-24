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

# Internal libraries
from helpers.utils import progress_bar
from helpers.uncertainty import get_pred, variation_ratio, predictive_entropy, \
    mutual_information


def validate(model, data, cache, num_runs):
    """

    Parameters
    ----------
    model : BayesModel()
        Bayes model for running tests
    data
    num_runs
    cache
    """
    q_factors = data.qf_test
    accuracies = np.zeros((len(q_factors), len(q_factors)))
    sizes = np.zeros((len(q_factors), len(q_factors)))

    performance = None
    if cache:
        try:
            performance = cache.load()
            performance["accuracy"]["testing"] = []
        except:
            performance = {"loss": {"validation": []},
                           "accuracy": {"validation": []}}
            logger.warning(
                "performance cache from training could not be loaded."
                " Making a new one."
            )

    for QF1, QF2 in progress_bar(product(q_factors, repeat=2)):
        qf1_ind, qf2_ind = np.where(np.isclose(q_factors, QF1))[0][0], \
                           np.where(np.isclose(q_factors, QF2))[0][0]
        for images, labels in data.get_validation_generator(QF1=QF1, QF2=QF2):
            labels = labels.numpy()
            if model.uncertainty_method == "vanilla":
                logits = model(images, training=False) / model.temperature
                predictions = logits.numpy().argmax(axis=1)

            else:
                if model.uncertainty_method == "ensemble":
                    logits = (model(images, training=False) /
                              model.temperature).numpy()

                else:
                    logits = np.array(
                        [model(images, training=False) / model.temperature
                         for _ in range(num_runs)])

                predictions = tf.squeeze(get_pred(logits))
                print(predictions, labels)

            accuracies[qf1_ind, qf2_ind] += np.sum(predictions == labels)
            sizes[qf1_ind, qf2_ind] += len(labels)

    accuracies = np.divide(accuracies, sizes)

    if cache:
        performance["accuracy"]["validation"].append(accuracies)
        cache.save(performance, step="performance")

    return accuracies
