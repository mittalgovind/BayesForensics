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


# Internal libraries

# ray.shutdown()
# ray.init(log_to_driver=False)
# # disable_gpu()


def create_keras_model(parameters):
    try:
        from flow import JPEGDoubleCompression
        model = JPEGDoubleCompression(method='vanilla', patch_size=64,
                                      **parameters)
        return model._model
    except RuntimeError:
        print("model cannot be created")
        return None


class Trainable:
    def __init__(self, data_train, data_val, root, batch_size, lr, save_dir):
        self.data_train = data_train
        self.data_val = data_val
        self.root = root
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_dir

    def train(self, config, reporter=None):
        import tensorflow as tf
        from dataset import DoubleCompressionDataset
        from models.jpeg import JPEG

        data = DoubleCompressionDataset(
            data_directory=os.path.join(self.root, 'data/rgb/native12k'),
            load="y",
            n_images=20,
            v_images=20,
            randomize=69,
            val_rgb_patch_size=64,
            calc_pywt_residual=False,
            qf_train="75,95",
            qf_test="75,95",
            codec=JPEG(codec='soft'),
            data_train=self.data_train,
            data_val=self.data_val
        )
        model = create_keras_model(config)
        if not model:
            return None
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        optimizer = tf.keras.optimizers.Nadam(self.lr)

        model.compile(optimizer, loss=loss_criterion, metrics=["accuracy"])
        callbacks = get_callbacks(self.save_dir)
        history = model.fit(
            x=data.get_training_generator(self.batch_size, 64),
            validation_data=data.get_validation_generator(self.batch_size),
            epochs=epochs,
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
        "dense_units": 128,
        "filters": 16,
        "kernel": 5,
        "pool_size": 1,
        "dense_multiplier": 1,
        "filter_multiplier": 1,
    }
    return hspace, good


def main(args=None):
    # Create snapshot directory
    root = '/scratch/gm2724'
    # root = '.'

    epochs = 4
    batch_size = 64
    num_samples = 5
    lr = 0.001
    save_dir = os.path.join(root, 'outputs/test')
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Initializing ray")
    ray.init(configure_logging=False)

    logger.info("Initializing ray search space")
    search_space, intial_best_config = create_search_space()

    # TODO: Adapt the below parameters according to the machine configuration
    num_cpus = 2
    num_gpus = 0

    logger.info("Initializing scheduler and search algorithms")
    # Use HyperBand scheduler to earlystop unpromising runs
    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration',
                                        metric="val_loss",
                                        mode="min",
                                        grace_period=10)

    # Use bayesian optimisation provided by hyperopt
    search_alg = HyperOptSearch(space=search_space,
                                metric="val_loss",
                                mode="min",
                                points_to_evaluate=[intial_best_config])

    # # We limit concurrent trials to 1 since bayesian optimisation doesn't parallelize very well
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)

    logger.info("Initializing ray Trainable")
    # Initialize Trainable for hyperparameter tuning
    data_train = np.load(os.path.join(root, 'data/rgb/native12k_1M_1.npy'))
    data_val = np.load(
        os.path.join(root, 'data/rgb/native12k_20k_val.npy'))

    trainer = Trainable(data_train, data_val, root, batch_size, lr, save_dir)

    logger.info("Starting hyperparameter tuning")
    analysis = tune.run(trainer.train,
                        verbose=1,
                        num_samples=num_samples,
                        search_alg=search_alg,
                        scheduler=scheduler,
                        raise_on_failed_trial=True,
                        resources_per_trial={"cpu": 4,
                                             "gpu": 1}
                        )

    best_config = analysis.get_best_config(metric="val_loss", mode='min')
    logger.info(f'Best config: {best_config}')
    #
    # if best_config is None:
    #     logger.error(f'Optimization failed')
    # else:
    #     logger.info("Saving best model config")
    #     with open(os.path.join(args.snapshot_dir, 'config.json'),
    #               'w') as f:
    #         import json
    #         json.dump(best_config, f, indent=4)
    #
    #     logger.info("Waiting for GPU/CPU memory cleanup")
    #     import time;
    #     time.sleep(3)
    #
    #     logger.info(f"Refitting the model on best config")
    #     trainer = Trainable(args.train_dir, args.val_dir,
    #                         args.snapshot_dir, final_run=True)
    #     history = trainer.train(best_config, reporter=None)

    logger.info("Training completed")


if __name__ == "__main__":
    try:
        main()
    except:
        import traceback

        logger.error(traceback.format_exc())
        raise
