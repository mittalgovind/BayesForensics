#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf
import numpy as np
import os
from loguru import logger

# Internal libraries
from helpers.utils import progress_bar
from helpers.plots import perf


def preprocess_batch(inputs, batch_size, codec, qf):
    """To preprocess input batch before training"""
    QF1 = int(tf.random.uniform((1,), *qf).numpy())
    QF2 = int(tf.random.uniform((1,), *qf).numpy())
    batch_single_compressed = codec.process(inputs, QF2)
    # compressing with QF1 before QF2, to give compression history to batch.
    batch_double_compressed = codec.process(codec.process(inputs, QF1), QF2)

    images = tf.concat((batch_single_compressed, batch_double_compressed), axis=0)
    labels = tf.concat((tf.zeros(batch_size), tf.ones(batch_size)), axis=-1)

    return images, labels


def train(
    model,
    epochs,
    data,
    batch_size,
    cache,
    qf,
    patch_size,
    lr,
    codec,
    save_every,
    save_dir,
    patience=150,
    min_delta=0.01,
    **kwargs
):
    performance = {"loss": {"training": []}, "accuracy": {"training": []}}
    n_batches = data.count_training // batch_size
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with progress_bar(epochs, "Training") as pbar:
        for epoch in range(epochs):
            losses = 0.0
            accuracies = 0.0

            for batch_id in range(n_batches):
                batch = data.next_training_batch(batch_id, batch_size, patch_size)
                batch, labels = preprocess_batch(batch, batch_size, codec, qf)
                with tf.GradientTape() as tape:
                    predictions = model(batch, training=True)
                    loss = loss_criterion(labels, predictions)

                grads = tape.gradient(loss, model._model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model._model.trainable_variables))

                losses += loss.numpy()

                accuracies += np.mean(predictions == labels)

            performance["loss"]["training"].append(losses / n_batches)
            performance["accuracy"]["training"].append(accuracies / n_batches)

            pbar.set_postfix(loss=losses / n_batches)
            pbar.update(1)

            if (epoch + 1) % save_every == 0:
                model.save_model(dirname=save_dir)
                fig = perf(performance, results="training")
                fig.savefig(
                    os.path.join(save_dir, "training_progress".format(epoch + 1))
                )

            if (epoch + 1) % patience == 0 and (
                performance["loss"]["training"][-patience]
                - performance["loss"]["training"][-1]
            ) >= min_delta:
                logger.log(
                    1,
                    "Loss did not decrease by {} in {} epochs. Stopping training.".format(
                        min_delta, patience
                    ),
                )
                break

        if cache:
            cache.save(performance, step="performance")

    return performance
