#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import numpy as np
from scipy.special import softmax
from scipy.stats import mode

# Internal libraries

MAX = int(2e16)
MIN = -int(2e16)


def get_pred(logits):
    # logits.shape = (None, ..., num_runs, batch_size, num_classes)
    pred_per_run = np.argmax(logits, axis=-1)
    return mode(pred_per_run, axis=-2)[0].squeeze()


def get_probs(logits):
    # expected input logits.shape = (None, ..., num_runs, batch_size, num_classes)
    # make it batch_first
    # (None, ..., batch_size, num_runs, num_classes)
    transpose_shape = np.arange(len(logits.shape) - 3).tolist() + [-2, -3, -1]
    return softmax(logits.transpose(transpose_shape), axis=-1)


def variation_ratio(logits, get_all=False):
    if not get_all:
        probs = get_probs(logits)
    else:
        probs = logits
    means = probs.mean(axis=-2)
    var_ratio = 1 - means[np.arange(means.shape[-3]), np.argmax(means, axis=-1)]
    return var_ratio


def predictive_entropy(logits, get_all=False, return_probs=False):
    if not get_all:
        # (None, ..., batch_size, num_runs, num_classes)
        probs = get_probs(logits)
    else:
        probs = logits

    # (None, ..., batch_size, num_classes)
    means = np.clip(probs.mean(axis=-2), 1e-16, None)

    # (None, ..., batch_size)
    pred_ent = -np.sum(np.multiply(means, np.log2(means)), axis=-1)

    if return_probs:
        return pred_ent, probs
    else:
        return pred_ent


def mutual_information(logits, get_all=False):
    if not get_all:
        entropy, probs = predictive_entropy(logits, return_probs=True)
    else:
        # (None, ..., batch_size, num_runs, num_classes)
        probs = logits
        # (None, ..., batch_size)
        entropy = predictive_entropy(probs, get_all=True)

    # (None, ..., batch_size, num_runs, num_classes)
    probs = np.clip(probs, 1e-16, None)
    # (None, ..., batch_size)
    # mi = H(X) - H(X|Y) 
    exp_value = np.multiply(probs, np.log2(probs)).sum(axis=-1).mean(axis=-1)
    return np.abs(entropy + exp_value)


# TODO rename get_all -> from_logits
def get_all_uncertainties(logits):
    probs = get_probs(logits)
    return [variation_ratio(probs, get_all=True),
            predictive_entropy(probs, get_all=True),
            mutual_information(probs, get_all=True)]


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
