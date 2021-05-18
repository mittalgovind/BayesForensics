import os

import tensorflow as tf
from loguru import logger


class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="epoch", logs=None):
        """Initializer.

        Args:
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        self.iteration = 0
        logs = logs or {}
        if freq not in ["batch", "epoch"]:
            raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = freq
        super(TuneReporter, self).__init__()

    def on_batch_end(self, batch, logs=None):
        from ray import tune
        logs = logs or {}
        if not self.freq == "batch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"))

    def on_epoch_end(self, batch, logs=None):
        from ray import tune
        logs = logs or {}
        if not self.freq == "epoch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, val_loss=logs['val_loss'],
                        mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, val_loss=logs['val_loss'],
                        mean_accuracy=logs.get("accuracy"))


def get_callbacks(path, model_name=None, save_freq=0, monitor='loss',
                  patience=200,
                  tensorboard=False, verbose=0, save_best_only=True,
                  min_delta=0.001):
    """callbacks list for keras models."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                         patience=patience,
                                         min_delta=min_delta,
                                         verbose=verbose),
        TuneReporter(freq="epoch")
    ]

    if verbose == 0:
        from tqdm.keras import TqdmCallback
        callbacks.append(TqdmCallback())
    if not model_name:
        model_name = 'model.h5'

    if save_freq != 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(path, model_name),
            save_weights_only=True,
            monitor=monitor, mode='auto',
            save_best_only=save_best_only,
            save_freq=save_freq,
            verbose=verbose
        ))


    if tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            os.path.join(path, 'tensorboard.log')))

    return callbacks
