#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os
import numpy as np

# External libraries
from hyperopt import hp, fmin, tpe, Trials, tpe, partial, STATUS_OK
from loguru import logger
import tensorflow as tf
from dataset import ScalingFactorDataset
from tqdm.keras import TqdmCallback
from flow import ScalingFactor
from tensorboard.plugins.hparams import api as tfhp


def create_keras_model(parameters, classes, patch_size):
    try:
        model = ScalingFactor(
            uncertainty_method="vanilla",
            n_classes=16,
            patch_size=64,
            use_bn=True,
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
                 classes=16, patch_size=64):
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
        self.counter = 1

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
            history = model.fit(
                x=self.train_data,
                validation_data=self.val_data,
                epochs=self.epochs,
                verbose=0,
                callbacks=self.callbacks,
            )
            tf.summary.scalar('accuracy',
                              np.mean(history.history['accuracy'][-10:]),
                              step=1)
            for i in range(self.epochs):
                tf.summary.scalar('acc',
                                  history.history['accuracy'][i],
                                  step=i + 1)
                tf.summary.scalar('val_acc',
                                  history.history['val_accuracy'][i],
                                  step=i + 1)
                tf.summary.scalar('loss',
                                  history.history['loss'][i],
                                  step=i + 1)
                tf.summary.scalar('val_loss',
                                  history.history['val_loss'][i],
                                  step=i + 1)
            writer.close()

        return {'loss': np.mean(history.history['val_loss'][-10:]),
                'status': STATUS_OK}


np.random.seed(5)


def create_search_space():
    # NAMES NEEDS TO BE IN LEXICOGRAPHICAL ORDER
    hspace = {
        "conv_layers": hp.choice("conv_layers", [3, 4, 5]),
        "dense_dropout": hp.choice("dense_dropout", [0.0, 0.05, 0.5]),
        "dense_layers": hp.choice("dense_layers", [0, 1, 2, 3]),
        "dense_units": hp.choice("dense_units", [128, 256, 384]),
        "filter_multiplier": hp.choice("filter_multiplier", [1, 2]),
        "filters": hp.choice("filters", [16, 32, 64, 128]),
        "kernel": hp.choice("kernel", [3, 5]),
        "pool_size": hp.choice("pool_size", [1, 2])
    }
    # NAMES NEEDS TO BE IN LEXICOGRAPHICAL ORDER
    tf_hspace = [
        tfhp.HParam("conv_layers", tfhp.Discrete([3, 4, 5])),
        tfhp.HParam("dense_dropout", tfhp.Discrete([0.0, 0.05, 0.5])),
        tfhp.HParam("dense_layers", tfhp.Discrete([1, 2, 3, 4])),
        tfhp.HParam("dense_units", tfhp.Discrete([128, 256, 384])),
        tfhp.HParam("filter_multiplier", tfhp.Discrete([1, 2])),
        tfhp.HParam("filters", tfhp.Discrete([16, 32, 64, 128])),
        tfhp.HParam("kernel", tfhp.Discrete([3, 5])),
        tfhp.HParam("pool_size", tfhp.Discrete([1, 2])),
    ]
    # JUST FOR REFERENCE. TBD.
    good = {
        "filters": 32,
        "filter_multiplier": 2,
        "conv_layers": 4,
        "kernel": 3,
        "dense_layers": 1,
        "dense_units": 256,
        "pool_size": 2,
        "dense_dropout": 0.1
    }

    return hspace, tf_hspace


def main(args):
    # Create save directory
    try:
        os.rmdir(args.save_dir)
    except:
        print('overwriting')
    os.makedirs(args.save_dir, exist_ok=True)

    search_space, tf_search_space = create_search_space()
    logger.info("Initializing scheduler and search algorithms")

    algo = partial(tpe.suggest,
                   n_EI_candidates=1000,
                   gamma=0.2,
                   n_startup_jobs=50)

    data_train = np.load(
        os.path.join(args.root, 'native12k_qM.npy'))[
                 :25 * args.n_images]
    data_val = np.load(
        os.path.join(args.root, 'native12k_20k_val.npy'))[
               :20 * args.v_images]
    data = ScalingFactorDataset(
        load="y",
        n_images=args.n_images,
        v_images=args.v_images,
        seed=69,
        val_rgb_patch_size=64,
        n_classes=16,
        scales="0.25,1.0",
        sampling_method="random",
        codec=None,
        preloaded_rgb_train_data=data_train,
        preloaded_rgb_val_data=data_val,
        batch_size=args.bs,
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
            metrics=[tfhp.Metric('accuracy', display_name='Accuracy'),
                     tfhp.Metric('acc'),
                     tfhp.Metric('val_acc'),
                     tfhp.Metric('loss'),
                     tfhp.Metric('val_loss')
                     ],
        )

    callbacks = [  # tf.keras.callbacks.TensorBoard(tb_log),
        TqdmCallback(verbose=args.verbose)]

    trainer = Trainable(args.root, args.lr, args.save_dir,
                        args.epochs, args.memory_growth, args.verbose,
                        train_data, val_data, callbacks, tf_search_space,
                        16, 64)
    logger.info("Starting hyperparameter tuning")
    trials = Trials()
    best_config = fmin(fn=trainer.train,
                       space=search_space,
                       algo=algo,
                       max_evals=args.num_samples,
                       show_progressbar=True, trials=trials)

    # best_config = analysis.get_best_config(metric="val_loss", mode='min')
    logger.info(f'Best config: {best_config}')

    if best_config is None:
        logger.error(f'Optimization failed')
    else:
        logger.info("Saving best model config")
        with open(os.path.join(args.save_dir, 'best_config.json'),
                  'w') as f:
            import json
            json.dump(best_config, f, indent=4)

    logger.info("Training completed")


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Hyperopt")
        parser.add_argument("--epochs", default=100, type=int)
        parser.add_argument("--verbose", default=0, type=int)
        parser.add_argument("--bs", default=2048, type=int)
        parser.add_argument("--num-samples", default=250, type=int)
        parser.add_argument("--n-images", default=1024, type=int)
        parser.add_argument("--v-images", default=1024, type=int)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--save-dir",
                            default='/scratch/gm2724/nip_runs/sfp_hyper',
                            type=str)
        parser.add_argument("--root",
                            default='/scratch/gm2724/data/rgb',
                            type=str)
        parser.add_argument("--memory-growth", action='store_true',
                            default=False)

        main(parser.parse_args())
    except:
        import traceback

        logger.error(traceback.format_exc())
        raise
