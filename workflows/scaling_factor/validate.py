#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf
from loguru import logger
import numpy as np

# Internal libraries
from helpers.utils import progress_bar
from helpers.uncertainty import get_pred, variation_ratio, predictive_entropy, \
    mutual_information


def validate(model, data, batch_size, cache, uncertainty_method,
             test_classes, num_runs=50):
    tests_summary = {}
    performance = None
    len_test_classes = test_classes.shape[0]
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

    with progress_bar(len(data.methods) * len(test_classes),
                      "Evaluation") as pbar:
        for m, method in enumerate(data.methods):
            tests_summary[method] = {}
            data.sampling_method = method
            for s, sf in enumerate(test_classes):
                if uncertainty_method == "vanilla":
                    logits = np.zeros(
                        (data.count_validation, len(data.classes)))

                elif uncertainty_method == "ensemble":
                    logits = np.zeros((data.count_validation, model.n_models,
                                       len(data.classes)))

                else:
                    logits = np.zeros((num_runs, data.count_validation,
                                       len(data.classes)))

                i = 0
                for images, labels in data.get_validation_generator(sf=sf):
                    if uncertainty_method in ["vanilla", "ensemble"]:
                        logits[i: i + batch_size] = (model(
                            images,
                            training=False) / model.temperature).numpy()

                    else:
                        logits[:, i: i + batch_size] = np.array(
                            [model(images, training=False) / model.temperature
                             for _ in range(num_runs)])

                    i += batch_size

                if uncertainty_method == "vanilla":
                    predictions = logits.argmax(axis=-1)
                else:
                    predictions = np.squeeze(get_pred(logits))

                labels, counts = np.unique(predictions, return_counts=True)
                for label, count in zip(labels, counts):
                    conf_matrix[m][s][label] += count
                conf_matrix[m][s] /= data.count_validation

                tests_summary[method] = logits
                pbar.update(1)

    if cache:
        performance["conf_matrices"] = conf_matrix
        performance["tests_summary"] = tests_summary
        cache.save(performance, step="performance")

    return tests_summary, conf_matrix


@tf.function
def distributed_validate(model, data, batch_size, cache, uncertainty_method,
                         test_classes, strategy=None, num_runs=50):
    if strategy:
        return strategy.run(validate, args=(model, data, batch_size, cache,
                                            uncertainty_method, test_classes,
                                            num_runs))
    else:
        return validate(model, data, batch_size, cache,
                        uncertainty_method, test_classes, num_runs)
