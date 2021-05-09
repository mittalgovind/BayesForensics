#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import argparse
import sys
import os
import json

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
    qf_plot,
)

# necessary here, as slurm executes a copy
sys.path.append(os.path.abspath("/"))


def main():
    args = parse_args()
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

    qf_train = (
        int(args.qf_train.split(",")[0]), int(args.qf_train.split(",")[1]))
    qf_test = (
        int(args.qf_test.split(",")[0]), int(args.qf_test.split(",")[1]))
    cache = ResultCache(["{step}.npz"], prefix=args.save_dir)

    # TODO (Govind) Change to the new standard parameters from sensor branch.
    if args.parameters:
        f = open(args.parameters, 'r')
    else:
        f = open('config/jpeg_double/default_params.json', 'r')

    try:
        parameters = json.load(f)
        f.close()
        logger.info(
            'Model configuration loaded successfully from {}.'.format(f))
        args.parameters = parameters
    except RuntimeError:
        logger.error("Cannot load parameter configuration.")
        sys.exit()

    print(args.parameters)

    calc_pywt_residual = True if "pywt" in args.parameters[
        "residual_type"] else False

    # load the dataset
    data = DoubleCompressionDataset(
        data_directory=args.data_dir,
        load="y",
        n_images=args.n_train_images if not args.load_model else 0,
        v_images=args.n_val_images,
        randomize=args.seed,
        val_rgb_patch_size=args.patch_size,
        calc_pywt_residual=calc_pywt_residual,
        qf_train=qf_train,
        qf_test=qf_test,
        codec=JPEG(codec=args.codec),
        presample_epochs=args.presample,
    )

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

    if args.load_model:
        model.load_model(os.path.abspath(args.load_model))
    else:
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        optimizer = tf.keras.optimizers.Adam(args.lr)

        model._model.compile(optimizer, loss=loss_criterion,
                             metrics=["accuracy"])
        save_freq = args.save_every * args.n_train_images // args.batch_size
        callbacks = get_callbacks(
            args.save_dir,
            model_name=model.model_filename,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent),
            verbose=args.verbosity
        ),
        train_performance = model._model.fit(
            x=data.get_training_generator(args.batch_size, args.patch_size),
            validation_data=data.get_validation_generator(args.batch_size),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbosity,
            callbacks=callbacks,
            steps_per_epoch=args.n_train_images // args.batch_size,
            validation_steps=args.n_vfmoal_images // args.batch_size
        )

        # save the training performance
        fig = perf(train_performance.history)
        fig.savefig(os.path.join(args.save_dir, "training_progress.pdf"))

    # TODO Add calibration
    # if args.calibrate:
    #     model.set_temp(data)

    logger.info("Started Testing")
    accuracies = validate(
        model=model,
        data=data,
        batch_size=args.batch_size,
        cache=cache
    )
    qf_plot(qf_test, accuracies, args.save_dir)


if __name__ == "__main__":
    main()

"""
 ACTIONS
 - train
 - validate 
 - hyperopt
 - calibrate
"""
