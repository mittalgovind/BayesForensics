#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from scipy.stats import mode
from scipy.special import softmax
import numpy as np
import tensorflow as tf

# Internal libraries

MAX = int(2e16)
MIN = -int(2e16)


def get_pred(logits):
    # logits.shape = (num_runs, batch_size, num_classes)
    # make it batch_first
    # (batch_size, num_runs, num_classes)
    logits = np.transpose(logits, (1, 0, 2))
    # (batch_size, num_runs)
    pred_per_run = np.argmax(logits, axis=-1)
    # (batch_size, )
    batch_pred = mode(pred_per_run, axis=1)[0]
    return batch_pred


def get_probs_passes_logits(logits):
    # logits.shape = (num_runs, batch_size, num_classes)
    # make it batch_first
    # (batch_size, num_runs, num_classes)
    batch_logits = np.transpose(logits, (1, 0, 2))
    probs = np.array([[softmax(run) for run in logits] for logits in batch_logits])
    n_passes = probs.shape[1]
    return probs, n_passes, logits


def variation_ratio(logits):
    probs, n_passes, _ = get_probs_passes_logits(logits)
    means = np.array([[np.sum(c) / n_passes for c in run.T] for run in probs])
    var_ratio = 1 - means[np.arange(means.shape[0]), np.argmax(means, axis=-1)]
    return var_ratio


def predictive_entropy(logits):
    probs, n_passes, _ = get_probs_passes_logits(logits)
    means = np.array([[np.sum(c) / n_passes for c in run.T] for run in probs])

    pred_ent = -np.sum(
        np.multiply(means, np.log2(np.clip(means, 1e-16, None))), axis=-1
    )
    return pred_ent


def mutual_information(logits):
    probs, n_passes, logits = get_probs_passes_logits(logits)
    pred_ent = predictive_entropy(logits)
    clipped_pred = np.clip(probs, 1e-16, None)
    exp_value = np.array(
        [
            np.divide(np.sum(np.multiply([prob], np.log2(prob))), n_passes)
            for prob in clipped_pred
        ]
    )

    return pred_ent + exp_value


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
