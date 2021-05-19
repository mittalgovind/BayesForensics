#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import sys
import os

# External libraries
import tensorflow as tf
import numpy as np

# Internal libraries
from helpers.utils import progress_bar
from helpers.stats import quantize
from helpers.plots import perf
from helpers.uncertainty import get_pred
from helpers.dataset import Dataset
from models.jpeg import JPEG

# Hacky fix
sys.path.append("/scratch/jms1595/neural-imaging-dev/")


class ScalingFactorDataset(Dataset):
    def __init__(
            self,
            scales,
            patch_size,
            sampling_method,
            n_classes,
            codec=None,
            jpeg_quality=100,
            **kwargs,
    ):
        """
        Subclass of helpers.dataset.Dataset class.

        Attributes
        ----------
        scales : tuple
            Range of values for scaling factor.
        patch_size : int
            Side length in pixels of the square patch.
        sampling_method : str
            Method to be used for sampling. Can be one of 'nearest',
             'bilinear', 'bicubic', 'lanczos3', or 'random'.
        n_classes : int
            Number of classes to split the scales range into.
        codec : str or None
            JPEG Compression type to preprocess batch with. None = no compression.
        jpeg_quality : int
            JPEG quality to compress with.
        """
        super().__init__(val_rgb_patch_size=patch_size, **kwargs)
        self.scales = (float(scales.split(",")[0]),
                       float(scales.split(",")[1]))
        self.patch_size = patch_size
        self.sampling_method = sampling_method
        self.methods = ["nearest", "bilinear", "bicubic", "lanczos3"]
        self.random_method = self.sampling_method == "random"
        self.classes = np.linspace(*self.scales, num=n_classes)
        if codec:
            self.codec = JPEG(quality=jpeg_quality, codec=codec)
        else:
            self.codec = None

    def preprocess_batch(self, batch, **kwargs):
        """
        Resize a batch with the desired scaling factor and sampling method.
        Includes JPEG compression when required.
        Returns the resized batch and their corresponding labels.

        Parameters
        ----------
        batch : np.array
            Batch to be preprocessed.

        Returns
        -------
        rescaled_images : tf.Tensor
            Tensor containing the resized batch.
        sf_labels : tf.Tensor
            Tensor containing the target labels.
        """

        if 'sf' in kwargs:
            sf = float(kwargs['sf'])
        else:
            sf = tf.random.uniform((1,), *self.scales)[0].numpy()

        resized_size = int(sf * self.patch_size)

        # Choose sampling method.
        if self.random_method:
            m = tf.random.choice(self.methods)
        else:
            m = self.sampling_method

        # Resize batch.
        rescaled_images = tf.image.resize(batch, [resized_size, resized_size],
                                          method=m)
        class_id = quantize([sf], self.classes, return_indices=True)
        sf_labels = tf.reshape(tf.repeat(class_id, batch.shape[0]), (-1, 1))

        # Convert to JPEG if a codec is passed.
        if self.codec:
            rescaled_images = self.codec.process(rescaled_images)

        return rescaled_images, sf_labels


# TODO (Marcelo) refactor in future release
def train_ensemble(
        model,
        epochs,
        data,
        batch_size,
        cache,
        codec,
        patch_size,
        scales,
        classes,
        sampling_method,
        save_dir,
        lr,
        methods,
        save_every,
        adversarial,
        **kwargs,
):
    """
    Trains models inside the Deep Ensemble.

    Parameters
    ----------
    model : sfp_ensemble.SFPEnsemble
        Model to be trained.
    epochs : int
        Number of epochs to train for.
    data : helpers.dataset.Dataset
        Dataset that will be used to load training images.
    batch_size : int
        Size of batch at every training step.
    cache : helpers.results_data.ResultCache
        Structure for naming performance files.
    codec : models.jpeg.JPEG
        Codec for conversion of images into JPEG. Use None for no conversion.
    patch_size : int
        Patch size for the model.
    scales : tuple
        Min and max value for scaling factor classes.
    classes : np.array
        List of all the possible scaling factors to use.
    save_dir : string
        Path for the model to be saved.
    lr : float
        Learning rate.
    sampling_method : string
        Sampling method to use for training. Can be one of "random", "nearest", "bilinear",
        "bicubic", or "lanczos3".
    methods : list of string
        List of sampling methods to be used if sampling_method is "random".
    adversarial : bool
        Whether or not to use adversarial training.
    epsilon : float
        Epsilon value for adversarial training.
    """

    random_method = sampling_method == "random"
    epsilon = None
    if adversarial:
        epsilon = kwargs["epsilon"]

    n_batches = data.count_training // batch_size

    # Different performance dictionary for each model.
    performance = [
        {"loss": {"training": []}, "accuracy": {"training": []}} for _ in
        model.models
    ]
    n_batches = data.count_training // batch_size
    opt = tf.keras.optimizers.Adam(lr)
    loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    with progress_bar(epochs, "Training") as pbar:
        for epoch in range(epochs):
            # Tracking losses separately.
            losses = [0 for _ in model.models]
            accuracies = [0 for _ in model.models]

            for batch_id in range(n_batches):
                # Get next training batch.
                batch_y = data.next_training_batch(batch_id, batch_size,
                                                   patch_size)

                batch_yy, batch_sf = preprocess_batch(
                    batch_y,
                    scales,
                    patch_size,
                    sampling_method,
                    random_method,
                    methods,
                    classes,
                    codec,
                )

                for i in range(model.n_models):
                    # Create adversarial batch.
                    if adversarial:
                        with tf.GradientTape() as tape:
                            tape.watch(batch_yy)
                            logits = model.models[i](batch_yy, training=True)
                            loss = loss_criterion(batch_sf, logits)

                        grad_adv = tape.gradient(loss, batch_yy)
                        sign_grads = tf.sign(grad_adv)
                        batch_adv = tf.clip_by_value(
                            batch_yy + sign_grads * epsilon, 0, 1
                        )

                    with tf.GradientTape() as tape:
                        logits = model.models[i](batch_yy, training=True)
                        loss = loss_criterion(batch_sf, logits)

                        # Loss becomes the sum of both adversarial loss and regular training loss.
                        if adversarial:
                            loss += loss_criterion(
                                batch_sf,
                                model.models[i](batch_adv, training=True)
                            )

                    grads = tape.gradient(
                        loss, model.models[i]._model.trainable_variables
                    )
                    opt.apply_gradients(
                        zip(grads, model.models[i]._model.trainable_variables)
                    )

                    predictions = get_pred(tf.convert_to_tensor([logits]))
                    # Update loss counter.
                    losses[i] += loss.numpy()
                    accuracies[i] += np.mean(predictions == batch_sf)

            # Save losses.
            for i in range(model.n_models):
                performance[i]["loss"]["training"].append(
                    losses[i] / n_batches)
                performance[i]["accuracy"]["training"].append(
                    accuracies[i] / n_batches)

            pbar.set_postfix(loss=np.mean(losses) / n_batches)
            pbar.update(1)

            if (epoch + 1) % save_every == 0:
                model.save_model(dirname=save_dir)
                for i in range(model.n_models):
                    fig = perf(performance[i], results="training")
                    fig.savefig(
                        os.path.join(
                            save_dir,
                            f"ensemble_{i:03d}",
                            "training_progress".format(epoch + 1),
                        )
                    )

    return performance
