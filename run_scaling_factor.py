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
from workflows.bayes_scaling_factor import (
    ScalingFactorDataset,
    validate,
    parse_args,
    ScalingFactor,
    sf_plot, plot_conf_matrix,
    load_parameters
)


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

    cache = ResultCache(["{step}}.npz"], prefix=args.save_dir)

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

    model = ScalingFactor(
        uncertainty_method=args.uncertainty_method,
        n_classes=args.n_classes,
        patch_size=args.patch_size,
        **args.parameters
    )

    # TODO CLEANUP
    # if args.uncertainty_method == "ensemble":
    #     model = DeepEnsemble([model for _ in range(5)])
    #     train_function = train_ensemble
    #
    # else:
    #     '''
    #     model = SFP(
    #         args.uncertainty_method,
    #         c_filters=(32, 32, 32, 32),
    #         d_filters=(32, 16, args.n_classes),
    #         kernel=5,
    #         activation="leaky_relu",
    #         trainable_residual=True,
    #         drop=0.1,
    #         append_rgb=False,
    #     )
    #     '''
    #     train_function = train_single

    #
    # if args.uncertainty_method == "ensemble":
    #     for performance in train_performance:
    #         perf(performance)
    # else:
    #     perf(train_performance)

    if args.load_model:
        model.load_model(os.path.abspath(args.load_model))
    else:
        optimizer = tf.keras.optimizers.Adam(args.lr)
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        model._model.compile(optimizer, loss=loss_criterion,
                             metrics=["accuracy"])
        save_freq = args.save_every * args.n_train_images // args.batch_size
        callbacks = get_callbacks(
            args.save_dir,
            model_name=model.model_filename,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent),
            verbose=args.verbose
        )

        train_performance = model._model.fit(
            x=data.get_training_generator(args.batch_size, args.patch_size),
            validation_data=data.get_validation_generator(args.batch_size),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
            callbacks=callbacks,
            steps_per_epoch=args.n_train_images // args.batch_size,
            validation_steps=args.n_val_images // args.batch_size,
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
