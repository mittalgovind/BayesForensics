#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os

# External libraries
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Dropout, Dense, Input, Conv2D
from hyperopt import hp, fmin, tpe, Trials, tpe, partial
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, \
    ModelCheckpoint

# Internal libraries
from helpers.dataset import Dataset
from models.layers import ConstrainedConv2D, PaddedConv2D
from models.jpeg import JPEG
from helpers.tf_helpers import activation_mapping


v_images = 256
t_images = 2048
batch_size = 256
patch_size = 64
epochs = 1500
# data_dir = "/scratch/gm2724/data/rgb/native12k"
# save_dir = "/scratch/gm2724/nip_runs/hyperopt"
data_dir = "/home/govind/Workspace/neural-imaging-dev/data/rgb/native12k"
save_dir = "./outputs"

if True:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from pdb import set_trace
def train_network(parameters):
    try:
        model = make_model(activation='leaky_relu', append_rgb=True,
                           **parameters)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
    except:
        print("model cannot be created")
        return np.inf

    try:
        # set_trace()
        history = model.fit(
                x=data.get_training_generator(batch_size, patch_size),
                validation_data=data.get_validation_generator(batch_size),
                epochs=epochs,
                batch_size=batch_size, verbose=0,
                callbacks=callbacks_list,
                steps_per_epoch=t_images//batch_size,
                validation_steps=v_images//batch_size
            )
        print("finished training this model")
        loss = min(history.history['loss'])
        accuracy = max(history.history['accuracy'])
        tf.keras.backend.clear_session()
    except:
        accuracy = 0
        loss = np.inf

    print("Loss: {}".format(loss))
    print("Accuracy: {:.2%}".format(accuracy))

    return loss

flags = {
    "batch_size": batch_size,
    "lr": 5e-4,
    "patch_size": patch_size,
    "save_dir": save_dir,
    "save_every": 100,
    "n_runs": 50,
}

data = Dataset(
    data_directory=data_dir,
    load="y",
    n_images=t_images,
    v_images=v_images,
    randomize=69,
    val_rgb_patch_size=patch_size,
)

parameter_space = {
    'conv_layers': hp.choice('conv_layers', [2, 4, 6]),
    'dense_layers': hp.choice('dense_layers', [0, 1, 2, 4]),
    'dense_units': hp.choice('dense_units', [128, 256, 512]),
    'filters': hp.choice('filters', [8, 16, 32, 64, 128]),
    'kernel': hp.choice('kernel', [3, 5]),
    'pool_size': hp.choice('pool_size', [1, 2]),
    "filter_multiplier": hp.choice("filter_multiplier", [1, 2]),
    "dense_multiplier": hp.choice("dense_multiplier", [1, 0.5])
    }

algo = partial(tpe.suggest,
               n_EI_candidates=1000,
               gamma=0.2,
               n_startup_jobs=50)

fmin(fn=train_network,
     space=parameter_space,
     algo=algo,
     max_evals=300,
     show_progressbar=True)
