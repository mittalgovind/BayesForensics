#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import sys
import os
import json

# External libraries
import numpy as np
import tensorflow as tf
from loguru import logger

# Internal libraries
from helpers.results_data import ResultCache
from helpers.plots import perf
from helpers.utils import setup_logging
from helpers.tf_helpers import disable_gpu
from models.bayes import DeepEnsemble
from workflows.bayes_scaling_factor import (
    # train_single,
    # train_ensemble,
    ScalingFactorDataset,
    run_tests,
    parse_args,
    ScalingFactor,
    sf_plot,
)

# TODO (Marcelo) Do you still need this?
# necessary here, as slurm executes a copy
sys.path.append(os.path.abspath("/"))


def main():
    setup_logging()
    args = parse_args()

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

    cache = ResultCache(["{step}_{sampling_method}.npz"], prefix=args.save_dir)

    flags = {
        "lr": args.lr,
        "save_dir": args.save_dir,
        "adversarial": args.adversarial,
        "epsilon": args.epsilon,
        "save_every": args.save_every,
    }

    data = ScalingFactorDataset(
        data_directory=args.data_dir,
        load="y",
        n_images=args.n_train_images,
        v_images=args.n_val_images,
        randomize=args.seed,
        scales=args.scales,
        patch_size=args.patch_size,
        sampling_method=args.sampling_method,
        n_classes=args.n_classes,
        codec=args.codec if args.jpeg_compression else None,
        jpeg_quality=args.jpeg_quality
    )

    # TODO (Govind) Change to the new standard parameters from sensor branch.
    if args.parameters:
        f = open(args.parameters, 'r')
    else:
        f = open('config/scaling_factor/default_params.json', 'r')

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

    if args.uncertainty_method == "ensemble":
        model = DeepEnsemble([
            ScalingFactor(
                method=args.uncertainty_method,
                n_classes=args.n_classes,
                patch_size=128,
                dropout=0.1,
            ) for _ in range(5)]
        )

        train_function = train_ensemble

    else:
        '''
        model = SFP(
            args.uncertainty_method,
            c_filters=(32, 32, 32, 32),
            d_filters=(32, 16, args.n_classes),
            kernel=5,
            activation="leaky_relu",
            trainable_residual=True,
            drop=0.1,
            append_rgb=False,
        )
        '''
        model = ScalingFactor(
            method=args.uncertainty_method,
            n_classes=args.n_classes,
            patch_size=128,
            dropout=0.1,
        )

        train_function = train_single

    if args.cont_model_path:
        # train for an epoch so that model is built
        _ = train_function(model, 1, data, args.batch_size, cache=None,
                           codec=codec, **flags)
        model.load_model(os.path.abspath(args.cont_model_path))

    if not args.only_eval:
        train_performance = train_function(model, args.epochs, data,
                                           args.batch_size, cache, codec,
                                           **flags)

        # save the training performance
        if args.uncertainty_method == "ensemble":
            for performance in train_performance:
                perf(performance)
        else:
            perf(train_performance)

    if args.calibrate:
        temp_model = BayarStammCalibrated(model, batch_size=args.batch_size)
        temp_model.set_temp(data)
        temperature = temp_model.temperature
    else:
        temperature = 1.0

    logger.info("Started Testing")

    tests_summary = run_tests(
        model,
        args.uncertainty_method,
        args.sampling_method,
        data,
        methods,
        scales,
        classes,
        args.n_val_images,
        patch_size,
        n_runs,
        cache,
        temperature,
        codec
    )

    sf_plot(tests_summary, classes, args.sampling_method, args.save_dir)


if __name__ == "__main__":
    main()
