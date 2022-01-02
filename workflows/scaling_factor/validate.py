#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from loguru import logger
import numpy as np

# Internal libraries
from helpers.utils import progress_bar
from helpers.uncertainty import get_pred


def validate(model, data, batch_size, cache, uncertainty_method,
             test_classes, num_runs=50, prefix=None,
             disable_temp_scaling=True):
    performance = None
    len_test_classes = len(test_classes)
    len_test_methods = len(data.methods)
    if cache:
        try:
            performance = cache.load()
            performance["conf_matrices"] = []
            performance["tests_summary"] = []
        except:
            performance = {"test_accuracy": [], "tests_summary": []}
            logger.warning(
                "performance cache from training could not be loaded."
                " Making a new one."
            )

    conf_matrix = np.zeros((len_test_methods, len_test_classes,
                            data.n_classes))

    # not testing on random method
    data.random_method = False
    if uncertainty_method == "vanilla":
        logits = np.zeros((len_test_methods, len_test_classes,
                           data.count_validation, data.n_classes))

    elif uncertainty_method == "ensemble":
        logits = np.zeros((len_test_methods, len_test_classes,
                           model.n_models, data.count_validation,
                           data.n_classes))

    else:
        logits = np.zeros((len_test_methods, len_test_classes,
                           num_runs, data.count_validation,
                           data.n_classes))

    if disable_temp_scaling:
        temperature = 1.0
    else:
        if uncertainty_method != "vanilla":
            logger.warning("Doing temperature scaling on non-vanilla models.")
        temperature = model.temperature

    with progress_bar(len_test_methods * len_test_classes,
                      "Evaluation") as pbar:
        for m, method in enumerate(data.methods):
            data.sampling_method = method
            for s, sf in enumerate(test_classes):
                i = 0
                for images, labels in data.get_validation_generator(sf=sf):
                    if uncertainty_method == "vanilla":
                        logits[m][s][i: i + batch_size] = model(
                            images, training=False).numpy()

                    elif uncertainty_method == "ensemble":
                        logits[m][s][:, i: i + batch_size] = model(
                            images, training=False).numpy()

                    else:
                        logits[m][s][:, i: i + batch_size] = [
                            model(images, training=False).numpy()
                            for _ in range(num_runs)]
                    i += batch_size

                logits /= temperature

                if uncertainty_method == "vanilla":
                    predictions = np.argmax(logits[m][s], axis=-1)
                else:
                    predictions = np.squeeze(get_pred(logits[m][s]))

                labels, counts = np.unique(predictions, return_counts=True)
                for label, count in zip(labels, counts):
                    conf_matrix[m][s][label] += count

                pbar.update(1)

    conf_matrix /= data.count_validation
    if cache:
        performance["conf_matrices"] = conf_matrix
        performance["logits"] = logits
        if prefix:
            cache.save(performance, step=f"{prefix}_performance")
        else:
            cache.save(performance, step="performance")

    return logits, conf_matrix
