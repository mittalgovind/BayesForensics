#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os
import numpy as np

# External libraries
import tensorflow as tf
from loguru import logger

# Internal libraries
from helpers.results_data import ResultCache
from helpers.plots import perf
from helpers.utils import setup_logging
from helpers.tf_helpers import disable_gpu, get_callbacks
from workflows.jpeg_double_compression import (
    DoubleCompressionDataset,
    parse_args,
    validate,
    JPEGDoubleCompression,
    TFJPEG,
    load_parameters,
    qf_plot,
)


def main():
    args = parse_args()
    args.parameters = load_parameters(args.parameters)
    setup_logging()

    if args.cpu:
        disable_gpu()

    if args.memory_growth:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.num_models > 1 and args.uncertainty_method != "ensemble":
        logger.warning("Number of models is greater than 1 but uncertainty "
                       "method is not ensemble. Setting it to ensemble.")
        args.uncertainty_method = "ensemble"

    elif args.num_models == 1 and args.uncertainty_method == "ensemble":
        logger.warning("Uncertainty method is ensemble but number of models is"
                       " 1. Setting number of models to 5.")
        args.num_models = 5

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

    # initializations
    cache = ResultCache(["{step}.npz"], prefix=args.save_dir)
    args.codec = TFJPEG(codec=args.codec)
    strategy = tf.distribute.MirroredStrategy()
    logger.info(
        'Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Prepare model with mirrored strategy.
    with strategy.scope():
        model = JPEGDoubleCompression(**args.parameters, **vars(args))
        optimizer = tf.keras.optimizers.Adam(args.lr)
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        model._model.compile(optimizer, loss=loss_criterion,
                             metrics=["accuracy"])

    # load presampled validation data
    if args.use_presampled:
        val_n_patches = 20
        loaded_val_data = np.load(
            os.path.join(args.use_presampled, 'native12k_20k_val.npy'))[
                          :val_n_patches * args.n_val_images]
    else:
        val_n_patches = 1
        loaded_val_data = None

    # Initialize the relevant dataset
    if args.load_model:
        data = DoubleCompressionDataset(
            load="y",
            n_images=1024,
            v_images=args.n_val_images,
            preloaded_rgb_val_data=loaded_val_data,
            val_n_patches=val_n_patches,
            val_rgb_patch_size=args.patch_size,
            calc_pywt_residual="pywt" in args.parameters["residual_type"],
            **vars(args)
        )
        model.load_model(os.path.abspath(args.load_model))
    else:
        # load presampled training data
        if args.use_presampled:
            train_n_patches = 25
            loaded_train_data = np.load(
                os.path.join(args.use_presampled, 'native12k_qM.npy'))[
                         :train_n_patches * args.n_train_images]
        else:
            loaded_train_data = None
            train_n_patches = 1

        data = DoubleCompressionDataset(
            load="y",
            n_images=args.n_train_images,
            preloaded_rgb_train_data=loaded_train_data,
            train_n_patches=train_n_patches,
            v_images=args.n_val_images,
            preloaded_rgb_val_data=loaded_val_data,
            val_n_patches=val_n_patches,
            val_rgb_patch_size=args.patch_size,
            calc_pywt_residual="pywt" in args.parameters["residual_type"],
            **vars(args)
        )

        # Data pipeline prep
        train_data = data.get_training_pipeline().prefetch(tf.data.AUTOTUNE)
        val_data = data.get_validation_pipeline().prefetch(tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        # get callbacks using options
        save_freq = args.save_every * args.n_train_images // args.batch_size
        callbacks = get_callbacks(
            args.save_dir,
            model_name=model.model_filename,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent),
            verbose=args.verbose,
            update_freq=args.validation_freq
        )

        train_performance = model._model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=args.epochs,
            verbose=0,
            callbacks=callbacks,
            validation_freq=args.validation_freq,
        )

        # save the training performance
        history = train_performance.history
        cache.save(history, step="performance")

        fig = perf(history, alpha=0.01)
        fig.savefig(os.path.join(args.save_dir, "training_progress.png"))

    if args.calibrate:
        model.set_temp(data)

    logger.info("Started Testing")
    accuracies = validate(
        model=model,
        data=data,
        cache=cache,
        num_runs=args.n_runs
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
