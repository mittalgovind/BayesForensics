#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os

# External libraries
import numpy as np
import tensorflow as tf
from loguru import logger
import matplotlib.pyplot as plt

# Internal libraries
from helpers.results_data import ResultCache
from helpers.plots import perf
from helpers.utils import setup_logging
from helpers.tf_helpers import disable_gpu, get_callbacks
from workflows.scaling_factor import (
    ScalingFactorDataset,
    distributed_validate,
    parse_args,
    ScalingFactor,
    sf_plot,
    load_parameters,
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

    if args.num_models > 1 and args.uncertainty_method != "ensemble":
        logger.warning("Number of models is greater than 1 but uncertainty "
                       "method is not ensemble. Setting it to ensemble mode.")
        args.uncertainty_method = "ensemble"

    elif args.num_models == 1 and args.uncertainty_method == "ensemble":
        logger.warning("Uncertainty method is ensemble but number of models is"
                       " 1. Setting number of models to 5.")
        args.num_models = 5

    # Check for saving directory
    if os.path.isdir(os.path.abspath(args.save_dir)):
        if not args.overwrite:
            raise IsADirectoryError(
                "Output directory exists!"
                " Use --overwrite or provide another directory name."
            )
        else:
            logger.warning("Overwriting output directory.")
    else:
        os.makedirs(args.save_dir)

    # initializations
    args.codec = args.codec if args.jpeg_compression else None
    cache = ResultCache(["{step}.npz"], prefix=args.save_dir)
    strategy = tf.distribute.MirroredStrategy()
    logger.info(
        'Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # Prepare model with mirrored strategy.
    with strategy.scope():
        model = ScalingFactor(**vars(args), **args.parameters)
        optimizer = tf.keras.optimizers.Adam(args.lr)
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        model._model.compile(optimizer, loss=loss_criterion,
                             metrics=["accuracy"])

    # load presampled validation data
    if args.use_presampled:
        if args.imagenet_val:
            val_n_patches = 1
            # loaded_val_data = np.zeros((1024, 128, 128, 3), np.float32)
            loaded_val_data = np.load(
                "./data/rgb/imagenet_128_4974.npy")[
                :args.n_val_images
            ]
        else:
            val_n_patches = 20
            loaded_val_data = np.load(
                os.path.join(args.use_presampled, 'native12k_20k_val.npy'))[
                              :val_n_patches * args.n_val_images]
    else:
        loaded_val_data = None
        val_n_patches = 1

    if args.load_model:
        data = ScalingFactorDataset(
            load="y",
            n_images=0,
            v_images=args.n_val_images,
            preloaded_rgb_val_data=loaded_val_data,
            val_n_patches=val_n_patches,
            val_rgb_patch_size=args.patch_size,
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

        data = ScalingFactorDataset(
            load="y",
            n_images=args.n_train_images,
            preloaded_rgb_train_data=loaded_train_data,
            train_n_patches=train_n_patches,
            v_images=args.n_val_images,
            preloaded_rgb_val_data=loaded_val_data,
            val_n_patches=val_n_patches,
            val_rgb_patch_size=args.patch_size,
            **vars(args)
        )

        # Data pipeline prep
        train_data = data.get_training_pipeline(
            gamma=args.gamma, brighten=args.brighten, rotate=args.rotate
        ).prefetch(tf.data.AUTOTUNE)
        val_data = data.get_validation_pipeline().prefetch(tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        # get callbacks using options
        steps_per_epoch = args.n_train_images // args.batch_size
        save_freq = args.save_every * steps_per_epoch
        callbacks = get_callbacks(
            args.save_dir,
            model_name=model.model_filename,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent),
            verbose=args.verbose,
            update_freq=args.validation_freq,
            steps_per_epoch=steps_per_epoch
        )

        # Start training
        train_performance = model._model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=args.epochs,
            verbose=0,
            callbacks=callbacks,
            validation_freq=args.validation_freq,
            # -1 because of an error of val_loss being nan for last class
            validation_steps=(args.n_classes - 1) * len(data.test_methods),
        )
        # save the training performance
        history = train_performance.history
        cache.save(history, step="performance")

        fig = perf(history, alpha=0.1)
        fig.savefig(os.path.join(args.save_dir, "training_progress.png"))

    if args.calibrate:
        logger.info("Started Calibration")
        model.set_temp(data=data, save_dir=args.save_dir)
    
    logger.info("Started Testing (1/3)")
    train_scales = (
        float(args.scales.split(",")[0]),
        float(args.scales.split(",")[1])
    )
    train_classes = tf.linspace(*train_scales, num=args.n_classes)
    tests_summary, conf_matrix = distributed_validate(
        model=model, data=data, batch_size=args.batch_size, cache=cache,
        uncertainty_method=args.uncertainty_method, test_classes=train_classes,
        strategy=strategy, num_runs=args.num_runs, prefix='normal_range'
    )
    
    performance = cache.load(step='normal_range_performance')
    tests_summary = performance["tests_summary"]
    conf_matrix = performance["conf_matrices"]

    sf_plot(tests_summary, conf_matrix, data.classes.numpy(), train_classes,
            args.sampling_method, args.save_dir, prefix='normal_range')

    logger.info("Started Testing (2/3)")

    train_scales = (
        float(args.scales.split(",")[0]),
        float(args.scales.split(",")[1])
    )
    train_classes = tf.linspace(*train_scales, num=args.n_classes * 5)
    tests_summary, conf_matrix = distributed_validate(
        model=model, data=data, batch_size=args.batch_size, cache=cache,
        uncertainty_method=args.uncertainty_method, test_classes=train_classes,
        strategy=strategy, num_runs=args.num_runs, prefix='in_range'
    )

    performance = cache.load(step='in_range_performance')
    tests_summary = performance["tests_summary"]
    conf_matrix = performance["conf_matrices"]

    sf_plot(tests_summary, conf_matrix, data.classes.numpy(), train_classes,
            args.sampling_method, args.save_dir, prefix='in_range')
    
    logger.info("Started Testing (3/3)")

    test_scales = args.test_scales.split(",")
    if len(test_scales) == 2:
        test_scales = (float(test_scales[0]), float(test_scales[1]))
    elif len(test_scales == 3):
        test_scales = (float(test_scales[0]), float(test_scales[1]),
                       float(test_scales[2]))
    else:
        raise ValueError("Test scales should be comma-separated pair/triplet.")
    test_classes = tf.linspace(*test_scales, num=args.test_n_classes)
    tests_summary, conf_matrix = distributed_validate(
         model=model, data=data, batch_size=args.batch_size, cache=cache,
         uncertainty_method=args.uncertainty_method, test_classes=test_classes,
         strategy=strategy, num_runs=args.num_runs, prefix='out_of_range'
    )
    
    performance = cache.load(step='out_of_range_performance')
    tests_summary = performance["tests_summary"]
    conf_matrix = performance["conf_matrices"]

    sf_plot(tests_summary, conf_matrix, data.classes.numpy(), test_classes,
            args.sampling_method, args.save_dir, prefix='out_of_range')


if __name__ == "__main__":
    main()
