#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
from scipy.stats import mode, entropy
from scipy.special import softmax
import numpy as np

# Internal libraries


def get_pred(logits):
    # logits.shape = (num_runs, batch_size, num_classes)
    # make it batch_first
    # (batch_size, num_runs, num_classes)
    logits = logits.numpy().transpose((1, 0, 2))
    # (batch_size, num_runs)
    pred_per_run = np.argmax(logits, axis=-1)
    # (batch_size, )
    batch_pred = mode(pred_per_run, axis=1)[0]
    return batch_pred


def get_probs_passes_logits(logits):
    # logits.shape = (num_runs, batch_size, num_classes)
    # make it batch_first
    # (batch_size, num_runs, num_classes)
    batch_logits = logits.numpy().transpose((1, 0, 2))
    probs = np.array(
        [[softmax(run) for run in logits] for logits in batch_logits])
    n_passes = probs.shape[1]
    return probs, n_passes, logits


def variation_ratio(logits):
    # set_trace()
    probs, n_passes, _ = get_probs_passes_logits(logits)
    means = np.array([[np.sum(c) / n_passes for c in run.T] for run in probs])
    var_ratio = 1 - means[np.argmax(means, axis=-1)]
    return var_ratio


def predictive_entropy(logits):
    probs, n_passes, _ = get_probs_passes_logits(logits)
    means = np.array([[np.sum(c) / n_passes for c in run.T] for run in probs])

    pred_ent = -np.sum(np.multiply(means, np.log(np.clip(means, 1e-12, None))),
                       axis=-1)
    return pred_ent


def mutual_information(logits):
    probs, n_passes, logits = get_probs_passes_logits(logits)
    pred_ent = predictive_entropy(logits)
    clipped_pred = np.clip(probs, 1e-12, None)
    exp_value = np.array(
        [np.divide(np.sum(np.multiply([prob], np.log(prob))), n_passes) for
         prob in clipped_pred])

    return pred_ent + exp_value
