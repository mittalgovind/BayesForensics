#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os
import numpy as np
import pickle

# External libraries
from hyperopt import hp, fmin, tpe, Trials, tpe, partial, STATUS_OK, \
    STATUS_FAIL
from loguru import logger
import tensorflow as tf
from tensorboard.plugins.hparams import api as tfhp
from tqdm.keras import TqdmCallback

# Internal Libraries
from workflows.scaling_factor.dataset import ScalingFactorDataset
from workflows.scaling_factor.flow import ScalingFactor


def create_keras_model(parameters, classes, patch_size):
    try:
        model = ScalingFactor(
            uncertainty_method="vanilla",
            n_classes=classes,
            patch_size=patch_size,
            use_bn=True,
            hyperoptimize=True,
            **parameters
        )
        return model._model
    except:
        logger.info("model cannot be created")
        return None


def get_tf_hparams(tf_space, parameters):
    i = 0
    hparams = {}
    for key, val in parameters.items():
        hparams[tf_space[i]] = val
        i += 1

    return hparams


class Trainable:
    def __init__(self, root, lr, save_dir, epochs, memory_growth,
                 verbose, train_data, val_data, callbacks, tf_space,
                 classes, patch_size, counter, validation_freq):
        self.epochs = epochs
        self.root = root
        self.lr = lr
        self.save_dir = save_dir
        self.memory_growth = memory_growth
        self.set_once = True
        self.verbose = verbose
        self.classes = classes
        self.patch_size = patch_size
        self.train_data = train_data
        self.val_data = val_data
        self.callbacks = callbacks
        self.tf_space = tf_space
        self.counter = counter
        self.validation_freq = validation_freq

    def train(self, config):
        if self.set_once and self.memory_growth:
            physical_devices = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            self.set_once = False
        print(config)
        strategy = tf.distribute.MirroredStrategy()
        logger.info(
            'Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            model = create_keras_model(config, self.classes, self.patch_size)
            # ON CREATION FAILURE
            if not model:
                return np.inf

            loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)
            optimizer = tf.keras.optimizers.Adam(self.lr)

            model.compile(optimizer, loss=loss_criterion, metrics=["accuracy"])

        config = get_tf_hparams(self.tf_space, config)
        logdir = os.path.join(self.save_dir, 'run-{}'.format(self.counter))
        self.counter += 1
        writer = tf.summary.create_file_writer(logdir)
        with writer.as_default():
            tfhp.hparams(config)
            try:
                history = model.fit(
                    x=self.train_data,
                    validation_data=self.val_data,
                    epochs=self.epochs,
                    verbose=self.verbose,
                    callbacks=self.callbacks,
                    validation_steps=self.classes - 1,
                    validation_freq=self.validation_freq,
                )
                rval = {'loss': np.mean(history.history['val_loss'][-6:]),
                        'status': STATUS_OK}
                for i in history.epoch:
                    tf.summary.scalar('Accuracy',
                                      history.history['accuracy'][i],
                                      step=i + 1)
                    tf.summary.scalar('Loss',
                                      history.history['loss'][i],
                                      step=i + 1)
                for i, val in enumerate(history.history['val_accuracy']):
                    tf.summary.scalar('Val Accuracy',
                                      history.history['val_accuracy'][i],
                                      step=i * self.validation_freq + 1)
                    tf.summary.scalar('Val Loss',
                                      history.history['val_loss'][i],
                                      step=i * self.validation_freq + 1)
                writer.close()
            except:
                logger.info(logdir + ' crashed')
                rval = {'loss': np.inf, 'status': STATUS_FAIL}
            tf.keras.backend.clear_session()
        return rval


def create_search_space():
    # NAMES NEEDS TO BE IN LEXICOGRAPHICAL ORDER
    hspace = {
        "conv_layers": hp.choice("conv_layers", [4, 5, 6]),
        "dense_dropout": hp.choice("dense_dropout", [0.0, 0.1]),
        "dense_layers": hp.choice("dense_layers", [1, 2, 3, 4]),
        "dense_units": hp.choice("dense_units", [128, 256, 384]),
        "filter_multiplier": hp.choice("filter_multiplier", [1, 2]),
        "filters": hp.choice("filters", [32, 64, 96, 128]),
        "kernel": hp.choice("kernel", [3, 5]),
        "pool_size": hp.choice("pool_size", [1, 2])
    }
    # NAMES NEEDS TO BE IN LEXICOGRAPHICAL ORDER
    tf_hspace = [
        tfhp.HParam("conv_layers", tfhp.Discrete([4, 5, 6])),
        tfhp.HParam("dense_dropout", tfhp.Discrete([0.0, 0.1])),
        tfhp.HParam("dense_layers", tfhp.Discrete([1, 2, 3, 4])),
        tfhp.HParam("dense_units", tfhp.Discrete([128, 256, 384])),
        tfhp.HParam("filter_multiplier", tfhp.Discrete([1, 2])),
        tfhp.HParam("filters", tfhp.Discrete([32, 64, 96, 128])),
        tfhp.HParam("kernel", tfhp.Discrete([3, 5])),
        tfhp.HParam("pool_size", tfhp.Discrete([1, 2])),
    ]
    # placeholding. Changes later.
    best_config = {
        "filters": 32,
        "filter_multiplier": 2,
        "conv_layers": 4,
        "kernel": 3,
        "dense_layers": 1,
        "dense_units": 256,
        "pool_size": 2,
        "dense_dropout": 0.1
    }

    return hspace, tf_hspace, best_config


def main(args):
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(7861)
    search_space, tf_search_space, best_config = create_search_space()
    logger.info("Initializing scheduler and search algorithms")

    data = ScalingFactorDataset(
        load="y",
        data_dir=os.path.join(args.root, 'native12k'),
        n_images=args.n_images,
        v_images=args.v_images,
        seed=7861,
        val_rgb_patch_size=args.patch_size,
        n_classes=args.classes,
        scales="0.25,1.0",
        sampling_method="bilinear",
        codec=None,
        batch_size=args.bs,
        xla=True
    )

    train_data = data.get_training_pipeline().prefetch(tf.data.AUTOTUNE)
    val_data = data.get_validation_pipeline().prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    with tf.summary.create_file_writer(args.save_dir).as_default():
        tfhp.hparams_config(
            hparams=tf_search_space,
            metrics=[tfhp.Metric('Accuracy'),
                     tfhp.Metric('Val Accuracy'),
                     tfhp.Metric('Loss'),
                     tfhp.Metric('Val Loss')
                     ],
        )

    callbacks = []
    if args.verbose > 0:
        callbacks.append(TqdmCallback(verbose=0))
        args.verbose = 0

    if args.tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard())

    logger.info("Starting hyperparameter tuning")

    algo = partial(tpe.suggest,
                   gamma=0.2,
                   n_startup_jobs=int(0.08 * args.num_samples))
    rcode = 1
    while rcode > 0:
        rcode = run_trials(args, search_space, train_data, val_data,
                           callbacks, tf_search_space, algo)

    trials = pickle.load(
        open(os.path.join(args.save_dir, "sfp_models.hyperopt"), "rb"))
    best_trial = trials.best_trial
    logger.info(f'Best trial: {best_trial}')
    id = best_trial['tid']
    i = 0
    for key, vals in trials.vals.items():
        best_config[key] = tf_search_space[i].domain.values[vals[id]]
        i += 1
    logger.info(f'Best config: {best_config}')
    logger.info("Training completed")


def run_trials(args, search_space, train_data, val_data, callbacks,
               tf_search_space, algo, max_evals=5):
    try:
        trials = pickle.load(
            open(os.path.join(args.save_dir, "sfp_models.hyperopt"), "rb"))
        if len(trials.trials) >= args.num_samples:
            return -1
        logger.info("Running from {} to {} trials ".format(
            len(trials.trials) + 1, len(trials.trials) + max_evals))
    except:
        logger.info('Created a new Trials() object')
        trials = Trials()

    trainer = Trainable(args.root, args.lr, args.save_dir,
                        args.epochs, args.memory_growth, args.verbose,
                        train_data, val_data, callbacks, tf_search_space,
                        args.classes, args.patch_size, len(trials.trials),
                        args.validation_freq)

    fmin(fn=trainer.train,
         space=search_space,
         algo=algo,
         max_evals=max_evals + len(trials),
         show_progressbar=True, trials=trials)

    # save the trials object
    with open(os.path.join(args.save_dir, "sfp_models.hyperopt"), "wb") as f:
        pickle.dump(trials, f)

    return 1


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Hyperopt")
        parser.add_argument("--epochs", default=175, type=int)
        parser.add_argument("--verbose", default=0, type=int)
        parser.add_argument("--bs", default=64, type=int)
        parser.add_argument("--num-samples", default=350, type=int)
        parser.add_argument("--n-images", default=1024, type=int)
        parser.add_argument("--v-images", default=128, type=int)
        parser.add_argument("--patch-size", default=128, type=int)
        parser.add_argument("--classes", default=31, type=int)
        parser.add_argument("--validation-freq", default=15, type=int)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--save-dir",
                            default='/scratch/gm2724/nip_runs/sfp_hyper',
                            type=str)
        parser.add_argument("--root",
                            default='/scratch/gm2724/data/rgb',
                            type=str)
        parser.add_argument("--memory-growth", action='store_true',
                            default=False)
        parser.add_argument("--tensorboard", action='store_true',
                            default=False)
        main(parser.parse_args())
    except:
        import traceback

        logger.error(traceback.format_exc())
        raise
