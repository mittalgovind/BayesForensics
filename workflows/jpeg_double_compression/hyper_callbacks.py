import os

import tensorflow as tf
from loguru import logger


class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, logs=None):
        """Initializer.

        Args:
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        self.iteration = 0
        logs = logs or {}
        super(TuneReporter, self).__init__()

    def on_epoch_end(self, batch, logs=None):
        from ray import tune
        logs = logs or {}
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


# def get_callbacks(path, model_name=None, save_freq=0, monitor='loss',
#                   tensorboard=False, verbose=0, save_best_only=True):
#     """callbacks list for keras models."""
#     callbacks = [TuneReporter()]
#
#     if verbose == 0:
#         from tqdm.keras import TqdmCallback
#         callbacks.append(TqdmCallback(verbose=verbose))
#     if not model_name:
#         model_name = 'model.h5'
#
#     if save_freq != 0:
#         callbacks.append(tf.keras.callbacks.ModelCheckpoint(
#             filepath=os.path.join(path, model_name),
#             save_weights_only=True,
#             monitor=monitor, mode='auto',
#             save_best_only=save_best_only,
#             save_freq=save_freq,
#             verbose=verbose
#         ))
#
#
#     if tensorboard:
#         callbacks.append(tf.keras.callbacks.TensorBoard(
#             os.path.join(path, 'tensorboard.log')))
#
#     return callbacks
