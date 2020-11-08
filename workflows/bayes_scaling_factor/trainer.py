#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import json

# External libraries
import tensorflow as tf
import numpy as np

# Internal libraries
from helpers.utils import progress_bar
from helpers.stats import quantize


def train(model, epochs, data, batch_size, cache, **kwargs):
    """

    Parameters
    ----------
    model
    epochs
    data
    batch_size
    cache
    kwargs

    Returns
    -------

    """
    patch_size = kwargs['patch_size']
    scales = kwargs['scales']
    classes = kwargs['classes']
    sampling_method = kwargs['sampling_method']
    save_dir = kwargs['save_dir']
    lr = kwargs['lr']
    random_method = sampling_method == 'random'

    n_batches = data.count_training // batch_size

    performance = {'loss': {'training': []}}
    loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    opt = tf.keras.optimizers.Adam(lr)

    with progress_bar(epochs, 'Training') as pbar:
        for epoch in range(epochs):
            losses = 0

            for batch_id in range(n_batches):
                batch_y = data.next_training_batch(batch_id, batch_size,
                                                   patch_size)
                sf = tf.random.uniform((1,), *scales)
                resized_size = int(sf * patch_size)

                if random_method:
                    method_idx = tf.random.shuffle([0, 1, 2, 3])
                    m = methods[rand_idx]
                else:
                    m = sampling_method

                batch_yy = tf.image.resize(batch_y,
                                           [resized_size, resized_size],
                                           method=m)
                class_id = quantize(sf.numpy(), classes, return_indices=True)
                batch_sf = np.repeat(class_id, batch_size).reshape((-1, 1))

                with tf.GradientTape() as tape:
                    loss = loss_criterion(batch_sf,
                                          model(batch_yy, training=True))

                grads = tape.gradient(loss, model._model.trainable_variables)
                opt.apply_gradients(zip(grads,
                                        model._model.trainable_variables))

                # Update loss counter
                losses += loss.numpy()

            performance['loss']['training'].append(losses / n_batches)

            pbar.set_postfix(loss=losses / n_batches)
            pbar.update(1)

            if (epoch + 1) % 100 == 0:
                model.save_model(dirname=save_dir)

        cache.save(performance, step='performance',
                   sampling_method=sampling_method)

    return model


def run_tests(model, sampling_method, data, methods, classes, batch_size,
              patch_size, num_runs, cache):
    """

    Parameters
    ----------
    model : BayesModel()
        Bayes model for running tests
    sampling_method : str

    data
    methods
    classes
    batch_size
    patch_size
    num_runs
    cache

    Returns
    -------

    """
    tests_summary = {'runs': []}

    n_val_batches = data.count_validation // batch_size
    num_eval = n_val_batches * len(classes) * len(methods)
    sfs = tf.convert_to_tensor((classes * patch_size).astype(int))

    with progress_bar(num_eval, 'Evaluation') as pbar:
        for batch_id in range(n_val_batches):
            test_batch = data.next_validation_batch(batch_id, batch_size)

            for m, method in enumerate(methods):
                for s, sf in enumerate(sfs):
                    rescaled = tf.image.resize(test_batch, [sf, sf],
                                               method=method)

                    logits = tf.convert_to_tensor(
                        [model(rescaled, training=False)
                         for _ in range(num_runs)])

                    tests_summary['runs'].append({'sf': sf,
                                                  'method': method,
                                                  'logits': logits})
                    pbar.update(1)

    cache.save(tests_summary, step='tests', sampling_method=sampling_method)
