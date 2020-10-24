#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:38:17 2020

@author: pkorus
"""

import numpy as np
import tensorflow as tf

from helpers import dataset, utils, plots, stats
from helpers import tf_helpers as tfh
from bayesian.sfp import SFP

# tfh.disable_warnings()
# tfh.disable_gpu()
utils.setup_logging()


# %%


# %% Load data

# TODO - native12k - coming straight from the cameras
# TODO - add jpeg - resize - jpeg again

data = dataset.Dataset('/Users/govindmittal/PycharmProjects/neural-imaging-dev-2/native12k',
                       load='y', n_images=64, v_images=64)

# %% Training loop

scales = (0.25, 1)
epochs = 1
batch_multip = 4
batch_size = 64
patch_size = 128
n_classes = 30 + 1

classes = np.linspace(*scales, num=n_classes)

print(f'{n_classes}: {classes.tolist()}')

# %%
n_batches = data.count_training // batch_size
# Model
model = SFP(c_filters=(32, 32, 32, 32),
            d_filters=(32, 16, n_classes), kernel=5,
            activation='leaky_relu', trainable_residual=True,
            drop=0.1, append_rgb=False)
opt = tf.keras.optimizers.Adam(1e-3)
loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

performance = {'loss': {'training': []}}

with utils.progress_bar(epochs, 'Traing') as pbar:
    for epoch in range(epochs):

        losses = 0

        for batch_id in range(n_batches):
            batch_y = data.next_training_batch(batch_id, batch_size,
                                               patch_size)
            sf = tf.random.uniform((1,), *scales)

            batch_yy = tf.image.resize(batch_y, [int(sf * patch_size),
                                                 int(sf * patch_size)])

            class_id = stats.quantize(sf.numpy(), classes, True)
            batch_sf = np.repeat(class_id, batch_size).reshape((-1, 1))

            with tf.GradientTape() as tape:
                loss = loss_op(batch_sf, model(batch_yy, training=True))

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            # Update loss counter
            losses += loss.numpy()

        performance['loss']['training'].append(losses / n_batches)

        pbar.set_postfix(loss=losses / n_batches)
        pbar.update(1)

model.load_weights('sf_bnn_run/bnn_7k.h5')
"""
import pickle
performance = pickle.load(open('sf_bnn_run/performance_7k.pkl', 'rb'))
plots.perf(performance)

# %% Testing loop
v_epochs = 1 # 5000
n_centroids = 100

centroids = np.linspace(*scales, num=n_centroids)

v_batches = data.count_validation // batch_size

sf_history = np.zeros((v_epochs, batch_size))
SF_history = np.zeros((v_epochs, batch_size))
sf_stats = np.zeros((n_centroids, n_classes))


for i in utils.progress_bar(range(v_epochs)):

    for batch_id in range(v_batches):

        batch_y = data.next_validation_batch(batch_id, batch_size)
        sf = tf.random.uniform((1,), *scales)
        batch_Y = tf.image.resize(batch_y,
                                  [int(sf * patch_size), int(sf * patch_size)])
        SFP = model(batch_Y) # dropout is still active
        SFI = SFP.numpy().argmax(axis=1)
        SF = classes[SFI]

        sf_history[i] = sf.numpy().ravel()
        SF_history[i] = SF.ravel()

        si = stats.quantize(sf.numpy(), centroids, return_indices=True)
        for pi in SFI:
            sf_stats[si, pi] += 1

sf_stats = pickle.load(open("sf_stats_2k.pkl", "rb"))
sf_history = pickle.load(open("sf_history_2k.pkl", "rb"))[:1800]
SF_history = pickle.load(open("SF_history_2k.pkl", "rb"))[:1800]
plots.set_interactive(True)
plots.correlation(sf_history, SF_history, 'scaling factor', 'prediction')
# TODO - save the figure in the morning

# %%
n_centroids = 100
n_classes = 31
sf_stats = pickle.load(open("sf_bnn_run/sf_stats_2k.pkl", "rb"))
fig = plots.image(sf_stats.T)
ax = fig.gca()

ax.set_ylabel('Predicted class')
ax.set_xlabel('Ground truth sf (quantized)')
ax.set_yticks([0, n_classes - 1])
ax.set_yticklabels(scales)
ax.set_xticks([0, n_centroids - 1])
ax.set_xticklabels(scales)
ax.plot([0, n_centroids - 1], [0, n_classes - 1], 'r:')

# %% Show probability slices

fig, axes = plots.sub(1)

for i in range(0, n_centroids, 10):
    axes[0].plot(sf_stats[i], '-')

# %% Show a single example and decision variance

batch_id = np.random.randint(data.count_validation)

n_samples = 50

batch_y = data.next_validation_batch(batch_id, 1)
# sf = tf.random.uniform((1,), *scales)
sf = 0.1
logits = np.zeros((n_samples, n_classes))

# TODO - common interface for BNN models
#   - add inference method

for n in range(n_samples):
    # TODO - change to skimage for resizing ()
    # TODO - choose different interpolation methods for resizing.
    batch_Y = tf.image.resize(batch_y,
                              [int(sf * patch_size), int(sf * patch_size)])
    logits[n, :] = model(batch_Y, training=False).numpy()
    SFI = logits[n].argmax()
    SF = classes[SFI]

plots.set_interactive(True)
fig, axes = plots.sub(4, ncols=2)

plots.image(batch_y, axes=axes[0])
plots.image(batch_Y.numpy(), axes=axes[1])

plots.intervals(classes, logits, axes=axes[2], xlabel='Predicted sf (class)',
                ylabel='Logits')
axes[2].plot([sf, sf], axes[2].get_ylim(), 'k:')

plots.hist([classes[logits.argmax(axis=1)].ravel()], classes,
           ['predicted sf (class)'], axes=axes[3])
axes[3].plot([sf, sf], axes[3].get_ylim(), 'k:')
axes[3].set_xlim(scales)
axes[3].set_xlabel('Predicted sf')

print(f'Predicted sf (classes): {classes[logits.argmax(axis=1)].round(2).tolist()}')
fig.show()
"""

# %%
from skimage.transform import rescale
from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image, ImageOps
from numpy import asarray
import matplotlib.pyplot as plt

image = asarray(Image.open('./sf_bnn_run/d90_01925_90.png'))
image_patch = image
# image_patch = Image.fromarray(extract_patches_2d(image, (89, 89), max_patches=1)[0])

# sf = 0.1
# image_rescaled = rescale(image, sf, anti_aliasing=False)
# turn off AA so that no need for extra smoothening

batch_id = np.random.randint(data.count_validation)
n_samples = 50
n_classes = 31
resizer = 'xnview'
# sf = tf.random.uniform((1,), *scales)
sf = 0.90
# image_patch = Image.fromarray((data.next_validation_batch(10, 1)[0]*255).astype(np.uint8))
# size = (int(sf * 128), int(sf * 128))
logits = np.zeros((n_samples, n_classes))
# batch_Y = asarray(image_patch.resize(size, resample=Image.BICUBIC))
batch_Y = asarray(image_patch)

for n in range(n_samples):
    # batch_Y = tf.image.resize(image_patch, [int(sf * patch_size), int(sf * patch_size)])
    logits[n, :] = model(tf.expand_dims(batch_Y.astype(float), axis=0), training=False).numpy()
    SFI = logits[n].argmax()
    SF = classes[SFI]

plots.set_interactive(True)
fig, axes = plots.sub(4, ncols=2)
plots.image(asarray(image_patch), axes=axes[0])
plots.image(batch_Y, axes=axes[1])

# plots.image(batch_y.numpy(), axes=axes[0])
# plots.image(batch_Y.numpy(), axes=axes[1])

plots.intervals(classes, logits, axes=axes[2], xlabel='Predicted sf (class)',
                ylabel='Logits')
axes[2].plot([sf, sf], axes[2].get_ylim(), 'k:')

plots.hist([classes[logits.argmax(axis=1)].ravel()], classes,
           ['predicted sf (class)'], axes=axes[3])
axes[3].plot([sf, sf], axes[3].get_ylim(), 'k:')
axes[3].set_xlim(scales)
axes[3].set_xlabel('Predicted sf')
print(f'Predicted sf (classes): {classes[logits.argmax(axis=1)].round(2).tolist()}')
fig.savefig('sf_bnn_run/results/native12k_patch_1_{}_{:.2f}.png'.format(resizer, float(sf)))
fig.show()
