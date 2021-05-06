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
from models.jpeg import JPEG
from helpers.dataset import Dataset
from helpers.results_data import ResultCache
from helpers.plots import perf
from helpers.utils import setup_logging
from helpers.tf_helpers import disable_gpu
from models.bayes import DeepEnsemble
from workflows.bayes_scaling_factor import (
    train_single,
    train_ensemble,
    run_tests,
    parse_args,
    SFP,
    BayarStammSFP,
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
    if os.path.isdir(os.path.abspath(args.save_dir)) and not args.overwrite:
        raise IsADirectoryError(
            "Output directory exists!"
            " Use --overwrite or provide another directory name."
        )

    scales = (
        float(args.scales.split(",")[0]), float(args.scales.split(",")[1]))
    methods = ["nearest", "bilinear", "bicubic", "lanczos3", "random"]
    cache = ResultCache(["{step}_{sampling_method}.npz"], prefix=args.save_dir)
    classes = np.linspace(*scales, num=args.n_classes)

    flags = {
        "lr": args.lr,
        "patch_size": args.patch_size,
        "scales": scales,
        "sampling_method": args.sampling_method,
        "classes": classes,
        "save_dir": args.save_dir,
        "methods": methods,
        "adversarial": args.adversarial,
        "epsilon": args.epsilon,
        "save_every": args.save_every,
    }

    data = Dataset(
        data_directory=args.data_dir,
        load="y",
        n_images=args.n_train_images,
        v_images=args.n_val_images,
        randomize=69,
    )

    if args.jpeg_compression:
        codec = JPEG(quality=args.jpeg_quality, codec="libjpeg")
    else:
        codec = None

    if args.uncertainty_method == "ensemble":
        model = DeepEnsemble([
            BayarStammSFP(
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
        model = BayarStammSFP(
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
