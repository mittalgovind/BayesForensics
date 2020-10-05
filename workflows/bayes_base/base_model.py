#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
from abc import abstractmethod

# External libraries

# Internal libraries
from models.tfmodel import TFModel


class BayesBaseModel(TFModel):
    """Defines a Tensorflow model (keras or not)."""

    def __init__(self):
        super().__init__()
        # Put all the layer instances used in the forward (call) pass

    @abstractmethod
    def create_model(self):
        """Construct the self._model variable with un-compiled keras model."""
        raise NotImplementedError
