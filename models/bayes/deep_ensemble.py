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


class DeepEnsemble:
    """
    Base class for Deep Ensemble Models.

    Parameters
    ----------
    base_model : models.tf_model.TFModel
        The base model from which all models within the DeepEnsemble
        are generated from.
    n_models : int
        The number of models contained within the Deep Ensemble.
    """

    def __init__(self, models):
        self.models = models
        self.n_models = len(models)
        
        '''
        for i in range(self.n_models):
            self.models[i]._create_model()
        '''

    def __call__(self, inputs, training=False):
        return tf.convert_to_tensor(
            [self.models[i](inputs, training=training) for i in range(self.n_models)]
        )

    def load_model(self, dirname):
        """
        Load weights from the specified directory.
        The directory from which to get the weights should have the
        following structure:

        target directory
        |-- ensemble_001
        |   |-- weights file
        |-- ensemble_002
        |   |-- weights file
        |-- ...

        Parameters
        ----------
        dirname : str
            Path to the directory where the weights are stored.
        """
        for i in range(self.n_models):
            path = os.path.join(dirname, f"ensemble_{i:03d}")
            self.models[i].load_model(path)

    def save_model(self, dirname):
        """
        Save weights to the specified directory.
        The resulting file structure is as follows:

        target directory
        |-- ensemble_001
        |   |-- weights file
        |-- ensemble_002
        |   |-- weights file
        |-- ...

        Parameters
        ----------
        dirname : str
            Path to the directory where the weights will be saved.
        """
        for i in range(self.n_models):
            path = os.path.join(dirname, f"ensemble_{i:03d}")
            self.models[i].save_model(path)
