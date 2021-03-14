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
data_dir = "/scratch/gm2724/data/rgb/native12k"
save_dir = "/scratch/gm2724/nip_runs/hyperopt"
# data_dir = "/home/govind/Workspace/neural-imaging-dev/data/rgb/native12k"
# save_dir = "./outputs"

if True:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class OptimDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess_batch(self, inputs, batch_size, codec, qf):
        """To preprocess input batch before training"""
        QF1 = int(tf.random.uniform((1,), *qf).numpy())
        QF2 = int(tf.random.uniform((1,), *qf).numpy())
        batch_single_compressed = codec.process(inputs, QF2)
        # compressing with QF1 before QF2, to give compression history to batch.
        batch_double_compressed = codec.process(
            codec.process(inputs, QF1), QF2)

        images = tf.concat(
            (batch_single_compressed, batch_double_compressed), axis=0
        )
        labels = tf.concat((tf.zeros(batch_size), tf.ones(batch_size)),
                           axis=-1)

        return images, labels

    def get_training_generator(self, batch_size, rgb_patch_size,
                               discard="flat"):
        """
        Get a generator for training data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_training_generator(batch_size, rgb_patch_size, discard),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """

        while True:
            for batch_id in range(self.count_training // batch_size):
                batch = self.next_training_batch(
                    batch_id, batch_size, rgb_patch_size, discard
                )
                images, labels = self.preprocess_batch(inputs=batch,
                                                       batch_size=batch_size,
                                                       codec=JPEG(),
                                                       qf=(75, 100))
                yield images, labels

    def get_validation_generator(self, batch_size):
        """
        Get a generator for validation data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_validation_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """
        while True:
            for batch_id in range(self.count_validation // batch_size):
                batch = self.next_training_batch(batch_id, batch_size)
                images, labels = self.preprocess_batch(inputs=batch,
                                                       batch_size=batch_size,
                                                       codec=JPEG(
                                                           codec="libjpeg"),
                                                       qf=(60, 100))
                yield images, labels


def make_model(kernel, activation, conv_layers, dense_layers, dense_units,
               filters, pool_size, append_rgb, ):
    """Need to override to specify model architecture."""
    activation = activation_mapping[activation]
    layers = []
    # Setup conv layers
    for _ in range(conv_layers):
        layers.append(
            Conv2D(filters, kernel, activation=activation)
        )
        layers.append(
            MaxPool2D(pool_size=(pool_size, pool_size)))

    layers.append(tf.keras.layers.GlobalAveragePooling2D())
    # Setup dense layers
    for i in range(dense_layers):
        last_layer = i == dense_layers - 1
        act = activation if not last_layer else None
        dense_units = dense_units // (
                1.5 ** i) if not last_layer else 2
        layers.append(Dense(dense_units, activation=act))
        if not last_layer:
            layers.append(Dropout(0.1))

    # make a custom keras model
    inputs = Input(shape=(patch_size, patch_size, 3))
    # if append_rgb:
    #     # concatenate residual if append_rgb is true
    #     outputs = tf.keras.layers.concatenate(
    #         [inputs, ConstrainedConv2D(trainable=True)(inputs)])
    # else:
    outputs = inputs

    for layer in layers:
        outputs = layer(outputs)

    return tf.keras.models.Model(inputs, outputs)


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


data = OptimDataset(
    data_directory=data_dir,
    load="y",
    n_images=t_images,
    v_images=v_images,
    randomize=69,
    val_rgb_patch_size=patch_size,
)

callbacks_list = [
    EarlyStopping(monitor='loss', patience=150,
                  restore_best_weights=True, min_delta=0.01)]
"""ModelCheckpoint(
    os.path.join(save_dir, 'model_save.h5'),
    monitor='loss'),
CSVLogger(
    filename=os.path.join(save_dir, 'model_save.log'),
    append=True)"""


parameter_space = {
    'conv_layers': hp.choice('conv_layers', [2, 3, 4, 5]),
    'dense_layers': hp.choice('dense_layers', [2, 3, 4]),
    'dense_units': hp.choice('dense_units', [64, 128, 192, 256, 384]),
    'filters': hp.choice('filters', [32, 64, 128, 150]),
    'kernel': hp.choice('kernel', [3, 5, 7]),
    'pool_size': hp.choice('pool_size', [2, 4, 6]),
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
