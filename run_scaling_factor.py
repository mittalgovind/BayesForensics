#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import sys
import os

# External libraries
import numpy as np
import tensorflow as tf
from loguru import logger

# Internal libraries
from helpers.results_data import ResultCache
from helpers.plots import perf
from helpers.utils import setup_logging
from helpers.tf_helpers import disable_gpu, get_callbacks
from workflows.scaling_factor import (
    ScalingFactorDataset,
    validate,
    parse_args,
    ScalingFactor,
    sf_plot, plot_conf_matrix,
    load_parameters
)

# TODO (Marcelo) Do you still need this?
# necessary here, as slurm executes a copy
sys.path.append(os.path.abspath("/"))


def main():
    setup_logging()
    args = parse_args()
    args.parameters = load_parameters(args.parameters)

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
    strategy = tf.distribute.MirroredStrategy()
    logger.info(
        'Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        model = ScalingFactor(
            uncertainty_method=args.uncertainty_method,
            n_classes=args.n_classes,
            patch_size=args.patch_size,
            **args.parameters
        )
        optimizer = tf.keras.optimizers.Adam(args.lr)
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        model._model.compile(optimizer, loss=loss_criterion,
                             metrics=["accuracy"])

    loaded_val_data = np.load(
        os.path.join(args.use_presampled, 'data/rgb/native12k_20k_val.npy'))[
               :20 * args.n_val_images]

    if args.load_model:
        model.load_model(os.path.abspath(args.load_model))
        data = ScalingFactorDataset(
            load="y",
            n_images=0,
            v_images=len(loaded_val_data),
            codec=args.codec if args.jpeg_compression else None,
            data_val=loaded_val_data,
            **vars(args)
        )

    else:
        save_freq = args.save_every * args.n_train_images // args.batch_size
        callbacks = get_callbacks(
            args.save_dir,
            model_name=model.model_filename,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent),
            verbose=args.verbose
        )
        # Data prep
        data_train = np.load(
            os.path.join(args.use_presampled, 'data/rgb/native12k_1M_1.npy'))[
                     :512 * args.n_train_images]
        data = ScalingFactorDataset(
            load="y",
            n_images=0,
            v_images=len(loaded_val_data),
            codec=args.codec if args.jpeg_compression else None,
            data_train=data_train,
            data_val=loaded_val_data,
            **vars(args)
        )
        train_data = data.get_training_pipeline(args.batch_size, 64).prefetch(
            tf.data.AUTOTUNE)
        val_data = data.get_validation_pipeline(args.batch_size).prefetch(
            tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

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
    tests_summary, conf_matrix = validate(
        model=model, data=data, batch_size=args.batch_size, cache=cache,
        uncertainty_method=args.uncertainty_method, num_runs=args.num_runs
    )

    # TODO (Marcelo) this needs adapting to new tests_summary
    # sf_plot(tests_summary, data.classes, args.sampling_method, args.save_dir)
    plot_conf_matrix(conf_matrix, data.methods, data.classes, args.save_dir)


if __name__ == "__main__":
    main()
