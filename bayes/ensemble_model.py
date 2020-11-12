#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University Abu Dhabi
# By: Marcelo Sandoval-Castaneda (marcelo.sc@nyu.edu)

# Standard libraries
from abc import abstractmethod
from copy import deepcopy
import os

# External libraries
import tensorflow as tf

# Internal libraries


# Brier score as loss. (Lakshminarayanan et al. 2017)
def brier_loss(y_true, y_pred, from_logits=False):
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
        
    return tf.reduce_mean(tf.reduce_sum((y_pred - tf.cast(y_true, tf.float32))**2, axis=1))


class DeepEnsemble:
    def __init__(self, base_model, n_models):
        self.models = [deepcopy(base_model) for _ in range(n_models)]
        self.n_models = n_models
        
        for i in range(self.n_models):
            self.models[i].class_name = f'ensemble_model_{i}'
    
    def __call__(self, inputs, training=False):
        return tf.convert_to_tensor([self.models[i](inputs, training=training) for i in range(n_models)])
    
    def load_weights(self, weights_dir):
        for i in range(self.n_models):
            self.models[i].load_model(f'{weights_dir}/ensemble_{i}')
    
    def save_weights(self, weights_dir):
        for i in range(self.n_models):
            self.models[i].save_model(f'{weights_dir}/ensemble_{i}')
    
    @abstractmethod
    def preprocess(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError
