#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf

# Internal libraries
from helpers.utils import progress_bar
from .train import preprocess_batch


def run_tests(
    model,
    uncertainty_method,
    sampling_method,
    data,
    methods,
    classes,
    batch_size,
    patch_size,
    num_runs,
    cache,
    temperature,
    codec,
):
    """

    Parameters
    ----------
    model : BayesModel()
        Bayes model for running tests
    sampling_method : str

    data
    methods
    classes
    batch_size
    patch_size
    num_runs
    cache
    temperature
    codec

    Returns
    -------

    """
    tests_summary = {"runs": []}

    n_val_batches = data.count_validation // batch_size
    num_eval = n_val_batches * len(classes) * len(methods)
    sfs = tf.convert_to_tensor((classes * patch_size).astype(int))

    random_method = sampling_method == "random"

    with progress_bar(num_eval, "Evaluation") as pbar:
        for batch_id in range(n_val_batches):
            test_batch = data.next_validation_batch(batch_id, batch_size)

            for m, method in enumerate(methods[:-1]):
                for s, sf in enumerate(sfs):
                    rescaled = preprocess_batch(test_batch, classes, patch_size, sampling_method,
                                                random_method, methods, classes, codec)

                    if uncertainty_method == 'ensemble':
                        logits = model(rescaled, training=False) / temperature
                    else:
                        logits = tf.convert_to_tensor([model(rescaled, training=False) for _ in range(num_runs)])

                    tests_summary["runs"].append(
                        {"sf": sf, "method": method, "logits": logits}
                    )
                    pbar.update(1)

    cache.save(tests_summary, step="tests", sampling_method=sampling_method)

    return tests_summary
