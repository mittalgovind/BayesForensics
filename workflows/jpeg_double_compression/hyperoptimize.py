#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os
import numpy as np

# External libraries
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from hyperopt import hp
from loguru import logger
from hyper_callbacks import get_callbacks


def create_keras_model(parameters):
    try:
        from flow import JPEGDoubleCompression
        model = JPEGDoubleCompression(method='vanilla', patch_size=64,
                                      **parameters)
        return model._model
    except RuntimeError:
        logger.info("model cannot be created")
        return None


class Trainable:
    def __init__(self, root, batch_size, lr, save_dir, epochs, n_images,
                 v_images, memory_growth):
        self.epochs = epochs
        self.root = root
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir
        self.n_images = n_images
        self.v_images = v_images
        self.memory_growth = memory_growth
        self.set_once = True

    def train(self, config, data_train=None, data_val=None):
        import tensorflow as tf
        from dataset import DoubleCompressionDataset
        from models.jpeg import JPEG
        if self.set_once and self.memory_growth:
            physical_devices = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            self.set_once = False

        data = DoubleCompressionDataset(
            load="y",
            n_images=self.n_images,
            v_images=self.v_images,
            randomize=69,
            val_rgb_patch_size=64,
            calc_pywt_residual=False,
            qf_train="75,95",
            qf_test="75,95",
            codec=JPEG(codec='soft'),
            data_train=data_train,
            data_val=data_val
        )

        model = create_keras_model(config)

        # ON CREATION FAILURE
        if not model:

            history = tf.keras.callbacks.History()
            history.history = {'loss': np.inf, 'accuracy': 0, 'val_acc': 0,
                               'val_loss': np.inf}
            return history

        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        optimizer = tf.keras.optimizers.Nadam(self.lr)

        model.compile(optimizer, loss=loss_criterion, metrics=["accuracy"])
        callbacks = get_callbacks(self.save_dir, verbose=0)
        history = model.fit(
            x=data.get_training_generator(self.batch_size, 64),
            validation_data=data.get_validation_generator(self.batch_size),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=callbacks,
            steps_per_epoch=data.count_training // self.batch_size,
            validation_steps=data.count_validation // self.batch_size,
        )
        return history


np.random.seed(5)


def create_search_space():
    hspace = {
        "conv_layers": hp.choice("conv_layers", [2, 3, 4, 5]),
        "dense_layers": hp.choice("dense_layers", [1, 2, 3, 4]),
        "dense_units": hp.choice("dense_units", [128, 256, 512]),
        "filters": hp.choice("filters", [16, 32, 64, 128]),
        "kernel": hp.choice("kernel", [3, 5]),
        "pool_size": hp.choice("pool_size", [1, 2]),
        "filter_multiplier": hp.choice("filter_multiplier", [1, 2]),
        "dense_multiplier": hp.choice("dense_multiplier", [0.5, 1])
    }
    good = {
        "conv_layers": 5,
        "dense_layers": 4,
        "dense_units": 512,
        "filters": 128,
        "kernel": 3,
        "pool_size": 1,
        "dense_multiplier": 1,
        "filter_multiplier": 2,
    }
    return hspace, good


def main(args):
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Initializing ray")
    ray.init(configure_logging=False, num_cpus=args.cpus, num_gpus=args.gpus)

    logger.info("Initializing ray search space")
    search_space, initial_best_config = create_search_space()

    logger.info("Initializing scheduler and search algorithms")
    # Use HyperBand scheduler to earlystop unpromising runs
    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration',
                                        metric="val_loss",
                                        mode="min")

    # Use bayesian optimisation provided by hyperopt
    search_alg = HyperOptSearch(space=search_space,
                                metric="val_loss",
                                mode="min",
                                points_to_evaluate=[initial_best_config])

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

    logger.info("Initializing ray Trainable")
    data_train = np.load(
        os.path.join(args.root, 'data/rgb/native12k_1M_1.npy'))[
                 :512 * args.n_images]
    data_val = np.load(
        os.path.join(args.root, 'data/rgb/native12k_20k_val.npy'))[
               :20 * args.v_images]

    if args.days > 0:
        time_budget_s = int(args.days * 3600 - 30*60)
    else:
        time_budget_s = None

    if args.gpus > 1:
        num_cpus_per_trial = args.cpus
    else:
        num_cpus_per_trial = 2

    trainer = Trainable(args.root, args.bs, args.lr, args.save_dir,
                        args.epochs, args.n_images, args.v_images,
                        args.memory_growth)

    logger.info("Starting hyperparameter tuning")
    analysis = tune.run(
        tune.with_parameters(trainer.train, data_train=data_train,
                             data_val=data_val),
        verbose=1,
        num_samples=args.num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        raise_on_failed_trial=False,
        resources_per_trial={"cpu": num_cpus_per_trial, "gpu": 1},
        resume=args.resume,
        local_dir=args.save_dir,
        log_to_file=True,
        time_budget_s=time_budget_s,
    )

    best_config = analysis.get_best_config(metric="val_loss", mode='min')
    logger.info(f'Best config: {best_config}')

    if best_config is None:
        logger.error(f'Optimization failed')
    else:
        logger.info("Saving best model config")
        with open(os.path.join(args.save_dir, 'config_on_two_gpus.json'),
                  'w') as f:
            import json
            json.dump(best_config, f, indent=4)

    logger.info("Training completed")


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Hyperopt")
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--cpus", default=2, type=int)
        parser.add_argument("--epochs", default=6, type=int)
        parser.add_argument("--days", default=0, type=int)
        parser.add_argument("--bs", default=2048, type=int)
        parser.add_argument("--num-samples", default=250, type=int)
        parser.add_argument("--n-images", default=512, type=int)
        parser.add_argument("--v-images", default=1024, type=int)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--save-dir",
                            default='/scratch/gm2724/nip_runs/ray_results/',
                            type=str)
        parser.add_argument("--root",
                            default='/scratch/gm2724',
                            type=str)
        parser.add_argument("--memory-growth", action='store_true',
                            default=False, )
        parser.add_argument("--resume", action='store_true',
                            default=False, )

        main(parser.parse_args())
    except:
        import traceback

        logger.error(traceback.format_exc())
        raise
