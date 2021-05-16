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


def validate(model, data, batch_size, cache, uncertainty_method, num_runs=50):
    tests_summary = {}
    if cache:
        try:
            performance = cache.load()
            performance["accuracy"]["validation"] = []
        except:
            performance = {"accuracy": {"validation": []}}
            logger.warning(
                "performance cache from training could not be loaded. Making a new one."
            )

    n_batches = data.count_validation // batch_size
    conf_matrix = np.zeros((len(data.methods), len(data.classes),
                            len(data.classes)))

    # not testing on random method
    data.random_method = False

    with progress_bar(len(data.methods) * len(data.classes),
                      "Evaluation") as pbar:
        for m, method in enumerate(data.methods):
            tests_summary[method] = {}
            data.sampling_method = method
            for s, sf in enumerate(data.classes):
                if "mc" in uncertainty_method:
                    logits = np.zeros((data.count_validation, num_runs,
                                       len(data.classes)))
                else:
                    logits = np.zeros(
                        (data.count_validation, len(data.classes)))
                for batch_id in range(n_batches):
                    batch = data.next_validation_batch(batch_id, batch_size)
                    images, labels = data.preprocess_batch(batch, sf=sf)
                    bindex = batch_id * batch_size

                    if "mc" in uncertainty_method:
                        # even if you set training=False, if the model is
                        # created for MC dropout, it should still work.
                        logits[
                        bindex: bindex + batch_size] = tf.convert_to_tensor(
                            [model(images, training=False) for _ in
                             range(num_runs)]) / model.temperature
                    else:
                        logits[bindex: bindex + batch_size] = (model(
                            images, training=False) / model.temperature).numpy()

                predictions = logits.argmax(axis=-1)

                if 'mc' in uncertainty_method:
                    # TODO finish calculating accuracy for mc
                    pass
                else:

                    conf_matrix[m][s] += np.unique(
                        predictions, return_counts=True)[
                                             1] / data.count_validation
                tests_summary[method][sf] = logits
                pbar.update(1)
    if cache:
        performance["accuracy"]["validation"] = conf_matrix
        cache.save(tests_summary, step="tests")
        cache.save(performance, step="performance")

    return tests_summary, conf_matrix
