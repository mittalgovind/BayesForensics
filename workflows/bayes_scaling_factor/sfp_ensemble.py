#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University Abu Dhabi
# By: Marcelo Sandoval-Castaneda (marcelo.sc@nyu.edu)

# Standard libraries
import sys

# External libraries
import tensorflow as tf
import numpy as np

# Hacky fix
sys.path.append("/scratch/jms1595/neural-imaging-dev/")

# Internal libraries
from models.bayes import DeepEnsemble
from helpers.utils import progress_bar
from helpers.stats import quantize


class SFPDeepEnsemble(DeepEnsemble):
    def __init__(self, base_model, n_models):
        super().__init__(base_model, n_models)

    def preprocess(
        self,
        batch,
        scales,
        patch_size,
        sampling_method,
        random_method,
        methods,
        classes,
    ):
        """
        Resize a batch with the desired scaling factor and sampling method.
        Returns the resized batch and their corresponding labels.

        Parameters
        ----------
        batch : list of np.array
            Batch to be preprocessed.
        scales : tuple
            Range of values for scaling factor.
        patch_size : int
            Side length in pixels of the square patch.
        sampling_method : str
            Method to be used for sampling.
            Can be one of 'nearest', 'bilinear', 'bicubic', 'lanczos3', or 'random'.
        random_method : bool
            Whether sampling_method is 'random' or not.

        Returns
        -------
        batch_processed : tf.Tensor
            Tensor containing the resized batch.
        batch_sf : tf.Tensor
            Tensor containing the target labels.
        """
        # Random scaling factor.
        sf = tf.random.uniform((1,), *scales)
        resized_size = int(sf * patch_size)

        # Choose sampling method.
        if random_method:
            method_idx = tf.random.shuffle([0, 1, 2, 3])[0]
            m = methods[method_idx]
        else:
            m = sampling_method

        # Resize batch.
        batch_processed = tf.image.resize(batch, [resized_size, resized_size], method=m)
        class_id = quantize(sf.numpy(), classes, return_indices=True)
        batch_sf = tf.reshape(tf.repeat(class_id, batch.shape[0]), (-1, 1))

        return batch_processed, batch_sf

    def train(self, epochs, data, batch_size, cache, **kwargs):
        """
        Trains models inside the Deep Ensemble.

        Parameters
        ----------
        epochs : int
            Number of epochs to train for.
        data : helpers.dataset.Dataset
            Dataset that will be used to load training images.
        batch_size : int
            Size of batch at every training step.
        cache : helpers.results_data.ResultCache
            Structure for naming performance files.
        kwargs
        """

        patch_size = kwargs["patch_size"]
        scales = kwargs["scales"]
        classes = kwargs["classes"]
        sampling_method = kwargs["sampling_method"]
        save_dir = kwargs["save_dir"]
        lr = kwargs["lr"]
        random_method = sampling_method == "random"
        methods = kwargs["methods"]

        n_batches = data.count_training // batch_size

        # Different performance dictionary for each model.
        performance = [{"loss": {"training": []}} for _ in range(self.n_models)]
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(lr)

        with progress_bar(epochs, "Training") as pbar:
            for epoch in range(epochs):
                # Tracking losses separately.
                losses = [0 for _ in self.models]

                for batch_id in range(n_batches):
                    # Get next training batch.
                    batch_y = data.next_training_batch(batch_id, batch_size, patch_size)

                    batch_yy, batch_sf = self.preprocess(
                        batch_y,
                        scales,
                        patch_size,
                        sampling_method,
                        random_method,
                        methods,
                        classes,
                    )

                    for i in range(self.n_models):
                        with tf.GradientTape() as tape:
                            loss = loss_criterion(
                                batch_sf, self.models[i](batch_yy, training=True)
                            )

                        grads = tape.gradient(loss, self.models[i].variables)
                        opt.apply_gradients(zip(grads, self.models[i].variables))

                        # Update loss counter.
                        losses[i] += loss.numpy()

                # Save losses.
                for i in range(self.n_models):
                    performance[i]["loss"]["training"].append(losses[i] / n_batches)

                pbar.set_postfix(loss=np.mean(losses) / n_batches)
                pbar.update(1)

                if (epoch + 1) % 100 == 0:
                    self.save_weights(save_dir)

        cache.save(
            {"performances": performance},
            step="performance",
            sampling_method=sampling_method,
        )
