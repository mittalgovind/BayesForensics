#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import argparse
import sys
import os
import numpy as np

# External libraries
import tensorflow as tf
from loguru import logger

# Internal libraries
from models.jpeg import JPEG
from helpers.results_data import ResultCache
from helpers.plots import perf
from helpers.utils import setup_logging
from helpers.tf_helpers import disable_gpu, get_callbacks
from workflows.jpeg_double_compression import (
    DoubleCompressionDataset,
    parse_args,
    validate,
    JPEGDoubleCompression,
    EnsembleJPEGDoubleCompression,
    load_parameters,
    qf_plot,
)

# necessary here, as slurm executes a copy
sys.path.append(os.path.abspath("/"))


def main():
    args = parse_args()
    args.parameters = load_parameters(args.parameters)
    setup_logging()

    if args.cpu:
        disable_gpu()

    if args.memory_growth:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Change json to npz
    if os.path.isdir(os.path.abspath(args.save_dir)):
        if not args.overwrite:
            raise IsADirectoryError(
                "Output directory exists!"
                " Use --overwrite or provide another directory name."
            )
        else:
            logger.warning("Overwriting output directory.")
    else:
        os.mkdir(args.save_dir)

    cache = ResultCache(["{step}.npz"], prefix=args.save_dir)

    # data_train = np.load(
    #     os.path.join(args.use_presampled, 'data/rgb/native12k_1M_1.npy'))[
    #              :512 * args.n_train_images]
    data_val = np.load(
        os.path.join(args.use_presampled, 'data/rgb/native12k_20k_val.npy'))[
               :20 * args.n_val_images]
    # load the dataset
    data = DoubleCompressionDataset(
        data_directory=args.data_dir,
        load="y",
        n_images=args.n_train_images if not args.load_model else 0,
        v_images=args.n_val_images,
        randomize=args.seed,
        val_rgb_patch_size=args.patch_size,
        calc_pywt_residual="pywt" in args.parameters["residual_type"],
        qf_train=args.qf_train,
        qf_test=args.qf_test,
        codec=JPEG(codec=args.codec),
        presample_epochs=args.presample,
        data_val=data_val,
        data_train=None,#data_train,
        use_presampled=args.use_presampled,  # TODO hacky fix
        batch_size=args.batch_size,
    )

    train_data = data.get_training_pipeline(args.batch_size, 64).prefetch(
        tf.data.AUTOTUNE)
    val_data = data.get_validation_pipeline(args.batch_size).prefetch(
        tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    strategy = tf.distribute.MirroredStrategy()
    logger.info(
        'Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        optimizer = tf.keras.optimizers.Adam(args.lr)
        if args.uncertainty_method == "ensemble":
            model = EnsembleJPEGDoubleCompression(
                num_models=5,
                method=args.uncertainty_method,
                patch_size=args.patch_size,
                **args.parameters,
            )

        else:
            model = JPEGDoubleCompression(
                method=args.uncertainty_method,
                patch_size=args.patch_size,
                **args.parameters,
            )
            model._model.compile(optimizer, loss=loss_criterion,
                                 metrics=["accuracy"])
    if args.load_model:
        model.load_model(os.path.abspath(args.load_model))
    else:
        save_freq = args.save_every * args.n_train_images // args.batch_size
        callbacks = get_callbacks(
            args.save_dir,
            model_name=model.model_filename,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent),
            verbose=args.verbose
        ),
        train_performance = model._model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=callbacks,
            validation_freq=args.validation_freq,
        )

        # save the training performance
        fig = perf(train_performance.history)
        fig.savefig(os.path.join(args.save_dir, "training_progress.png"))

    # TODO Add calibration
    # if args.calibrate:
    #     model.set_temp(data)

    logger.info("Started Testing")
    accuracies = validate(
        model=model,
        data=data,
        batch_size=args.batch_size,
        cache=cache,
        num_runs=args.num_runs
    )
    qf_plot(data.qf_test, accuracies, args.save_dir)


if __name__ == "__main__":
    main()

"""
 ACTIONS
 - train
 - validate 
 - hyperopt
 - calibrate
"""
