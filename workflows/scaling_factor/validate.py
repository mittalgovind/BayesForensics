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
from helpers.uncertainty import get_pred


def get_indices(m, s, batch_size):
    indices = []
    for i in range(batch_size):
        indices.append([m, s, i])


def validate(model, data, batch_size, cache, uncertainty_method,
             test_classes, num_runs=50):
    tests_summary = {}
    performance = None
    len_test_classes = test_classes.shape[0]
    temp = tf.arange(batch_size)
    # making tensor iterable
    test_classes = tf.data.Dataset.from_tensor_slices(test_classes)
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

    conf_matrix = tf.zeros((len_test_methods, len_test_classes,
                            data.n_classes))

    # not testing on random method
    data.random_method = False
    if uncertainty_method == "vanilla":
        logits = tf.zeros((len_test_methods, len_test_classes,
                           data.count_validation, data.n_classes))

    elif uncertainty_method == "ensemble":
        logits = tf.zeros((len_test_methods, len_test_classes,
                           data.count_validation, model.n_models,
                           data.n_classes))

    else:
        logits = tf.zeros((len_test_methods, len_test_classes,
                           num_runs, data.count_validation,
                           data.n_classes))

    with progress_bar(len_test_methods * len_test_classes, "Evaluation") as pbar:
        for m, method in enumerate(data.methods):
            data.sampling_method = method
            for s, sf in enumerate(test_classes):
                i = 0
                for images, labels in data.get_validation_generator(sf=sf):
                    if uncertainty_method in ["vanilla", "ensemble"]:
                        logit = model(
                            images, training=False) / model.temperature
                        logits = tf.tensor_scatter_nd_update(
                            logits, [[m, s, i + k] for k in range(batch_size)],
                            logit)
                    else:
                        logit = [model(images, training=False) / model.temperature
                                 for _ in range(num_runs)]
                        logits = tf.tensor_scatter_nd_update(logits,
                                    [[m, s, r, i + k] for r in range(num_runs)
                                     for k in range(batch_size)], logit)

                    i += batch_size

                if uncertainty_method == "vanilla":
                    predictions = logits.argmax(axis=-1)
                else:
                    predictions = np.squeeze(get_pred(logits))

                labels, counts = np.unique(predictions, return_counts=True)
                for label, count in zip(labels, counts):
                    conf_matrix = tf.tensor_scatter_nd_add(
                        conf_matrix, [[m, s, label]], [count])

                pbar.update(1)

    conf_matrix = tf.math.divide(conf_matrix, data.count_validation)
    if cache:
        performance["conf_matrices"] = conf_matrix
        performance["tests_summary"] = logits
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
