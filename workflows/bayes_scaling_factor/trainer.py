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


def train(model, epochs, data, batch_size, **kwargs):
    patch_size = kwargs['patch_size']
    scales = kwargs['scales']
    classes = kwargs['classes']
    sampling_method = kwargs['sampling_method']
    save_dir = kwargs['save_dir']
    lr = kwargs['lr']

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
                if sampling_method == 'random':
                    m = tf.random.choice(
                        ['nearest', 'bilinear', 'bicubic', 'lanczos3'])
                else:
                    m = sampling_method

                batch_yy = tf.image.resize(batch_y, [int(sf * patch_size),
                                                     int(sf * patch_size)],
                                           method=m)
                class_id = quantize(sf.numpy(), classes, True)
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

    with open(f'./output/performance_{sampling_method}.json',
              'w') as f:
        json.dump(performance, f, indent=4)


def run_tests(model, model_method, data, methods, classes, batch_size,
              patch_size, num_runs):
    tests_summary = {'runs': []}

    test_batch = data.next_validation_batch(0, batch_size)
    for sf in classes:
        for method in methods:
            rescaled = np.asarray(tf.image.resize(test_batch,
                                                  [int(sf * patch_size),
                                                   int(sf * patch_size)],
                                                  method=method))

            for i in range(len(rescaled)):
                logits = []
                for j in range(num_runs):
                    logits.append(model(
                        tf.expand_dims(rescaled[i].astype(float), axis=0),
                        training=False).numpy().tolist()[0])
                tests_summary['runs'].append({'sf': sf,
                                              'method': method,
                                              'img_id': i,
                                              'logits': logits})

    with open(f'bayesian-train-test/tests_{model_method}.json', 'w') as f:
        json.dump(tests_summary, f, indent=4)

#
#
# def train(model, data, method, epochs, classes, n_classes, n_batches,
#           batch_size, patch_size, scales):
#     opt = tf.keras.optimizers.Adam(1e-3)
#     loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#     performance = {'loss': {'training': []}}
#
#     with utils.progress_bar(epochs, 'Train') as pbar:
#         for epoch in range(epochs):
#
#             losses = 0
#
#             for batch_id in range(n_batches):
#                 batch_y = data.next_training_batch(batch_id, batch_size,
#                                                    patch_size)
#                 sf = tf.random.uniform((1,), *scales)
#
#                 if method == 'random':
#                     m = np.random.choice(
#                         ['nearest', 'bilinear', 'bicubic', 'lanczos3'])
#                 else:
#                     m = method
#
#                 batch_yy = tf.image.resize(batch_y,
#                                            [int(sf * patch_size),
#                                             int(sf * patch_size)],
#                                            method=m)
#
#                 class_id = stats.quantize(sf.numpy(), classes, True)
#                 batch_sf = np.repeat(class_id, batch_size).reshape((-1, 1))
#
#                 with tf.GradientTape() as tape:
#                     loss = loss_op(batch_sf, model(batch_yy, training=True))
#
#                 grads = tape.gradient(loss, model.trainable_variables)
#                 opt.apply_gradients(zip(grads, model.trainable_variables))
#
#                 # Update loss counter
#                 losses += loss.numpy()
#
#             performance['loss']['training'].append(losses / n_batches)
#
#             pbar.set_postfix(loss=losses / n_batches)
#             pbar.update(1)
#
#     model.save_weights(f'bayesian-train-test/weights_{method}.h5')
#
#     with open(f'bayesian-train-test/performance_{method}.json', 'w') as f:
#         json.dump(performance, f, indent=4)
#
#     return model
