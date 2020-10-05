#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries

# Internal libraries

model = SFP(c_filters=(32, 32, 32, 32),
            d_filters=(32, 16, n_classes), kernel=5,
            activation='leaky_relu', trainable_residual=True,
            drop=0.1, append_rgb=False)