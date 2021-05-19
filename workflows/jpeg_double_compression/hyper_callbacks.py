import tensorflow as tf


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
