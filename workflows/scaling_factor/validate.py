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


def validate(model, data, batch_size, cache, uncertainty_method,
             test_classes, num_runs=50, prefix=None):
    tests_summary = {}
    performance = None
    len_test_classes = test_classes.shape[0]
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
                    if uncertainty_method == "vanilla":
                        logit = model(images, training=False) / model.temperature
                        logits = tf.tensor_scatter_nd_update(
                            logits, [[m, s, i + k] for k in range(batch_size)],
                            logit)
                    
                    elif uncertainty_method == "ensemble":
                        logit = tf.convert_to_tensor(
                            model(images, training=False)) / model.temperature
                        logit = tf.reshape(logit, (-1, data.n_classes))
                        logits = tf.tensor_scatter_nd_update(
                            logits,
                            [[m, s, r, i + k] for r in range(model.n_models)
                             for k in range(batch_size)], logit)
                    
                    else:
                        logit = [model(images, training=False) / model.temperature
                                 for _ in range(num_runs)]
                        logit = tf.reshape(logit, (-1, data.n_classes))
                        logits = tf.tensor_scatter_nd_update(logits,
                                    [[m, s, r, i + k] for r in range(num_runs)
                                     for k in range(batch_size)], logit)

                    i += batch_size

                if uncertainty_method == "vanilla":
                    predictions = tf.math.argmax(logits[m][s], axis=-1)
                else:
                    predictions = np.squeeze(get_pred(logits[m][s]))

                labels, counts = np.unique(predictions, return_counts=True)
                conf_matrix = tf.tensor_scatter_nd_add(
                        conf_matrix, [[m, s, label] for label in labels], counts)

                pbar.update(1)

    conf_matrix = tf.math.divide(conf_matrix, data.count_validation)
    if cache:
        performance["conf_matrices"] = conf_matrix
        performance["logits"] = logits
        if prefix:
            cache.save(performance, step=f"{prefix}_performance")
        else:
            cache.save(performance, step="performance")

    return tests_summary, conf_matrix


# @tf.function
def distributed_validate(model, data, batch_size, cache, uncertainty_method,
                         test_classes, strategy=None, num_runs=50, prefix=None):
    if strategy:
        return strategy.run(validate, args=(model, data, batch_size, cache,
                                            uncertainty_method, test_classes,
                                            num_runs, prefix))
    else:
        return validate(model, data, batch_size, cache,
                        uncertainty_method, test_classes, num_runs, prefix)
