#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os
from loguru import logger

# External libraries
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Dropout, Dense, Input, Conv2D
from hyperopt import hp, fmin, tpe, Trials, tpe, partial
import numpy as np
import pickle
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, \
    ModelCheckpoint

# Internal libraries
from helpers.dataset import Dataset
from helpers.results_data import ResultCache
from helpers.plots import perf
from models.layers import ConstrainedConv2D, PaddedConv2D
from models.jpeg import JPEG
from workflows.jpeg_double_compression import train, JPEGDoubleCompression
from helpers.tf_helpers import activation_mapping

v_images = 64
t_images = 2048
batch_size = 256
patch_size = 64
epochs = 2000
data_dir = "/scratch/gm2724/data/rgb/native12k"
base_save_dir = "/scratch/gm2724/nip_runs/hyperopt_again"
# data_dir = "/home/govind/Workspace/neural-imaging-dev/data/rgb/native12k"
# save_dir = "./outputs"
logger.add("/scratch/gm2724/nip_runs/hyperopt_again/thread_1.txt")

from pdb import set_trace

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_network(parameters):
    global run, trials
    logger.info("======RUN - {} ======".format(run))
    logger.info(parameters)
    save_dir = os.path.join(base_save_dir, str(run))
    cache = ResultCache(["performance.npz"], prefix=save_dir)
    try:
        model = JPEGDoubleCompression(method="vanilla", **parameters)
        model.summary()
    except:
        logger.info("model cannot be created")
        return np.inf

    try:
        performance = train(
            model=model,
            epochs=epochs,
            data=data,
            batch_size=batch_size,
            cache=cache,
            qf=(70, 100),
            patch_size=patch_size,
            lr=5e-4,
            codec=JPEG(codec="libjpeg"),
            save_dir=save_dir,
            patience=200,
            save_every=100,
        )

        fig = perf(performance)
        fig.savefig(os.path.join(save_dir, 'train.png'))
        run += 1
        tf.keras.backend.clear_session()
        loss = min(performance["loss"]["training"])
        logger.info("Loss = {:.3f}".format(loss))
        return loss
    except:
        print("model cannot be trained!")
        return np.inf


data = Dataset(
    data_dir=data_dir,
    load="y",
    n_images=t_images,
    v_images=v_images,
    seed=69,
    val_rgb_patch_size=patch_size,
)

parameter_space = {
    'conv_layers': hp.choice('conv_layers', [2, 3, 4, 5]),
    'filters': hp.choice('filters', [16, 32, 64, 128]),
    "filter_multiplier": hp.choice("filter_multiplier", [1, 2]),
    'kernel': hp.choice('kernel', [3, 5]),
    'pool_size': hp.choice('pool_size', [1, 2]),
    'dense_layers': hp.choice('dense_layers', [0, 1, 2, 4]),
    'dense_units': hp.choice('dense_units', [128, 256, 384]),
    "dense_multiplier": hp.choice("dense_multiplier", [1, 0.5])
}

run = 1


def run_trials():

    global run
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("jpg_models.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        run = max_trials
        print("Rerunning from {} trials to {} (+{}) trials".format(
            len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    algo = partial(tpe.suggest,
                   n_EI_candidates=1000,
                   gamma=0.2,
                   n_startup_jobs=20)

    best = fmin(fn=train_network, space=parameter_space, algo=algo,
                max_evals=max_trials, trials=trials)

    logger.info("Best:", best)

    # save the trials object
    with open("jpg_models.hyperopt", "wb") as f:
        pickle.dump(trials, f)


# loop indefinitely and stop whenever you like
while True:
    run_trials()


# TODO make plots like in the fifty paper.
# TODO how to make this distributed? Ray ML ..
