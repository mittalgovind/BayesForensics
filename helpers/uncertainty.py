#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import numpy as np
from scipy.special import softmax
from scipy import stats

# Internal libraries

MAX = int(2e16)
MIN = -int(2e16)


def get_pred(logits):
    # logits.shape = (None, ..., num_runs, batch_size, num_classes)
    pred_per_run = np.argmax(logits, axis=-1)
    return stats.mode(pred_per_run, axis=-2)[0].squeeze()


def get_probs(logits):
    soft_logits = np.clip(softmax(logits, axis=-1), 1e-16, None)
    return soft_logits


def variation_ratio(logits):
    probs = get_probs(logits)

    # (..., batch_size, num_runs)
    preds = probs.argmax(axis=-1)
    # mode = (..., batch_size, 1), count = (..., batch_size, 1)
    mode = stats.mode(preds, axis=-2)
    # (..., batch_size)
    var_ratio = 1 - mode[1].squeeze() / preds.shape[-1]
    return var_ratio


def predictive_entropy(logits, mean_across_data=False):
    probs = get_probs(logits)
    # mean across runs. ensure you have (..., num_runs, num_samples, num_classes)
    mean_probs = probs.mean(axis=-3)

    # sum across classes
    pred_ent = -np.sum(np.multiply(mean_probs, np.log2(mean_probs)), axis=-1)

    if mean_across_data:
        pred_ent = pred_ent.mean(axis=-1)

    return pred_ent


def mutual_information(logits, entropy=None, mean_across_data=False):
    if entropy is None:
        entropy = predictive_entropy(logits, mean_across_data=mean_across_data)

    # (None, ..., batch_size)
    probs = get_probs(logits)
    # last sum is over classes
    expectation_across_classes = -np.multiply(probs, np.log2(probs)).sum(axis=-1)

    # this one is over runs.
    expectation_across_runs = expectation_across_classes.mean(axis=-2)

    mi = entropy - expectation_across_runs

    if mean_across_data:
        mi = mi.mean(axis=-1)

    return mi


def get_limits(n_models, n_classes):
    logits = []
    for i in range(n_models):
        model = [[MIN] * n_classes]
        model[0][i % n_classes] = MAX
        logits.append(model)

    logits = np.array(logits)

    uncertainty_limits = {
        "variation_ratio": variation_ratio(logits)[0],
        "predictive_entropy": predictive_entropy(logits)[0],
        "mutual_information": mutual_information(logits)[0],
    }

    return uncertainty_limits
