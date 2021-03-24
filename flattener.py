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
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

# Internal libraries
from helpers.dataset import Dataset
from helpers.plots import perf
from models.layers import ConstrainedConv2D, PaddedConv2D
from models.jpeg import JPEG
from helpers.tf_helpers import activation_mapping
from workflows.jpeg_double_compression import train, JPEGDoubleCompression
from helpers.results_data import ResultCache

patch_size = 64

if "CLUSTER" in os.environ and os.environ["CLUSTER"] == "GREENE":
    data_dir = "/scratch/gm2724/data/rgb/native12k"
    save_dir = "/scratch/gm2724/nip_runs/flattener"
    v_images = 256
    t_images = 2048
    batch_size = 256
    epochs = 2000
else:
    data_dir = "/home/govind/Workspace/neural-imaging-dev/data/rgb/native12k"
    save_dir = "./outputs"
    v_images = 64
    t_images = 64
    batch_size = 64
    epochs = 5


class JPEGDoodleCompression(JPEGDoubleCompression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        """Need to override to specify model architecture."""
        layers = []
        # Setup conv layers
        for _ in range(self.conv_layers):
            layers.append(
                self.conv2d(self.filters, self.kernel, activation=self.activation)
            )
            layers.append(MaxPool2D(pool_size=(self.pool_size, self.pool_size)))

        layers.append(tf.keras.layers.Flatten())
        # Setup dense layers
        for i in range(self.dense_layers):
            last_layer = i == self.dense_layers - 1
            act = self.activation if not last_layer else None
            dense_units = self.dense_units // (1.5 ** i) if not last_layer else 2
            layers.append(self.dense(dense_units, activation=act))
            if self.drop_rate > 0 and not last_layer:
                layers.append(self.dropout(self.drop_rate))

        # make a custom keras model
        inputs = Input(shape=(patch_size, patch_size, 3))
        if self.append_rgb:
            # concatenate residual if append_rgb is true
            outputs = tf.keras.layers.concatenate(
                [inputs, ConstrainedConv2D(trainable=True)(inputs)])
        else:
            outputs = inputs
        for layer in layers:
            outputs = layer(outputs)

        self._model = tf.keras.models.Model(inputs, outputs)
        if self.tensorboard:
            self.tensorboard.set_model(model=self._model)



from pdb import set_trace


def train_network(parameters):
    cache = ResultCache(["{step}.npz"], prefix=save_dir)
    try:
        model = JPEGDoubleCompression(method="vanilla", **parameters)
        model.summary()
    except:
        print("model cannot be created")
        return np.inf

    performance = train(
        model,
        epochs,
        data,
        batch_size,
        cache,
        (75, 100),
        patch_size,
        1e-4,
        JPEG(),
        100,
        save_dir,
    )

    print("finished training this model")
    # set_trace()
    model.save(save_dir + "flattener.h5")
    fig=perf(performance)
    fig.savefig(save_dir+'train.pdf')
    # loss = min(history.history['loss'])
    # accuracy = max(history.history['accuracy'])
    tf.keras.backend.clear_session()
    # accuracy = 0
    # loss = np.inf

    # print("Loss: {}".format(loss))
    # print("Accuracy: {:.2%}".format(accuracy))


data = Dataset(
    data_directory=data_dir,
    load="y",
    n_images=t_images,
    v_images=v_images,
    randomize=69,
    val_rgb_patch_size=patch_size,
)

# callbacks_list = [
#     EarlyStopping(monitor='loss', patience=150,
#                   restore_best_weights=True, min_delta=0.01)]
"""ModelCheckpoint(
    os.path.join(save_dir, 'model_save.h5'),
    monitor='loss'),
CSVLogger(
    filename=os.path.join(save_dir, 'model_save.log'),
    append=True)"""

parameters = {
    "conv_layers": 4,
    "dense_layers": 3,
    "dense_units": 64,
    "filters": 32,
    "kernel": 3,
    "pool_size": 2,
}

train_network(parameters)
