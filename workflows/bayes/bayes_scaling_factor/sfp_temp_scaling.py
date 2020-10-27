#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# External libraries
import tensorflow as tf
import numpy as np

# Internal Libraries
from helpers import dataset, utils, plots, stats
from workflows.bayes.bayes_base.temp_scaling import TemperatureScaling


class SFP_temp(TemperatureScaling):
    def __init__(self, model, batch_size):
        super().__init__(model, batch_size)

    def preprocess(self, batch, return_labels=False):
        scales = (0.25, 1)
        n_classes = 31
        patch_size = 128
        classes = np.linspace(*scales, num=n_classes)
        sf = tf.random.uniform((1,), *scales)
        batch = tf.image.resize(batch,
                                [int(sf * patch_size), int(sf * patch_size)])
        class_id = stats.quantize(sf.numpy(), classes, True)
        labels = np.repeat(class_id, self.batch_size).reshape((-1, 1))
        if return_labels:
            return batch, tf.convert_to_tensor(labels)
        else:
            return batch
