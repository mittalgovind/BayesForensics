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
    validate,
    JPEGDoubleCompression,
    qf_plot,
)

setup_logging()

# necessary here, as slurm executes a copy
sys.path.append(os.path.abspath("/"))


# TODO (put in notion) Refactor Workflows to Pipelines
# TODO Refactor train_* scripts to run_* scripts
# TODO put together every run scripts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a bayesian NN on different methods for downsampling"
    )
    parser.add_argument(
        "-um",
        "--uncertainty-method",
        action="store",
        default="mc-dropout",
        type=str,
        help="Uncertainty method."
             " Can be 'vanilla', 'mc-dropout', 'temp-scaling', 'mc-temp',"
             " 'flipout', or 'reparameterization'.",
    )
    parser.add_argument(
        "--qf-train",
        dest="qf_train",
        action="store",
        default="75,100",
        type=str,
        help="Comma separated values for lower and upper bound of Quality "
             "factor used for training, e.g. '75,100'",
    )
    parser.add_argument(
        "--qf-test",
        dest="qf_test",
        action="store",
        default="60,100",
        type=str,
        help="Comma separated values for lower and upper bound of Quality "
             "factor used for testing, e.g. '60,100'",
    )
    parser.add_argument(
        "--codec",
        action="store",
        default="soft",
        type=str,
        help="Type of codec. Possible choices - libjpeg, soft, sin, harmonic. (default: soft)",
    )
    parser.add_argument(
        "--patch-size",
        dest="patch_size",
        action="store",
        default=64,
        type=int,
        help="Square patch size to be sampled from images (default: 64)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        action="store",
        default=10000,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        dest="batch_size",
        action="store",
        default=64,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-nt",
        "--train-images",
        dest="n_train_images",
        action="store",
        default=1024,
        type=int,
        help="Number of images for training set",
    )
    parser.add_argument(
        "-nv",
        "--validation-images",
        dest="n_val_images",
        action="store",
        default=256,
        type=int,
        help="Number of images for validation set",
    )
    parser.add_argument(
        "-nc",
        "--calibration-images",
        dest="n_val_images",
        action="store",
        default=256,
        type=int,
        help="Number of images for calibration set",
    )
    parser.add_argument(
        "-nr",
        "--num-runs",
        dest="n_runs",
        action="store",
        default=50,
        type=int,
        help="Number of test runs per image in validation set",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output",
        help="Output save directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/govind/Workspace/neural-imaging-dev/data/rgb/native12k",
        help="Data directory for getting images from.",
    )
    parser.add_argument(
        "-se",
        "--save-every",
        action="store",
        default=100,
        type=int,
        help="Number of epochs to log after.",
    )
    parser.add_argument(
        "-lr", "--lr", action="store", default=5e-4, type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to a trained model.",
        default=None,
    )
    parser.add_argument(
        "--parameters",
        type=str,
        default=None,
        help="path to a parameters JSON file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output folder, if exists.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Calibrate model using temperature scaling.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Output tensorboard logs to the save directory.",
    )
    parser.add_argument(
        "--memory-growth",
        action="store_true",
        default=False,
        help="Enable TF memory growth for GPU.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Disables GPU utilization.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="Controls verbosity of training.",
    )
    parser.add_argument(
        "--patience-percent",
        type=float,
        default=0.1,
        help="Percentage of total epochs to use as patience. (def : 0.1 or 10%).",
    )
    return parser.parse_args()


def main():
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

    qf_train = (
        int(args.qf_train.split(",")[0]), int(args.qf_train.split(",")[1]))
    qf_test = (
        int(args.qf_test.split(",")[0]), int(args.qf_test.split(",")[1]))
    cache = ResultCache(["{step}.npz"], prefix=args.save_dir)

    flags = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patch_size": args.patch_size,
        "save_dir": args.save_dir,
        "save_every": args.save_every,
        "n_runs": args.n_runs,
    }

    # TODO (Govind) Change to the new standard parameters from sensor branch.
    if args.parameters is None:
        # TODO change to a good config after hyperopt
        # TODO put this in a default config file.
        args.parameters = {
            "conv_layers": 4,
            "dense_layers": 2,
            "dense_units": 512,
            "filters": 64,
            "kernel": 5,
            "pool_size": 1,
            "residual_type": 'trainable',
            "dense_multiplier": 0.5,
            "filter_multiplier": 1,
        }
    else:
        try:
            f = open(args.parameters, 'r')
            parameters = json.load(f)
            f.close()
            logger.info(
                'Model configuration loaded successfully from {}.'.format(
                    args.parameters))
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
        n_images=args.n_train_images,
        v_images=args.n_val_images,
        randomize=69,
        val_rgb_patch_size=args.patch_size,
        calc_pywt_residual=calc_pywt_residual,
        qf_train=qf_train,
        qf_test=qf_test,
        codec=JPEG(codec=args.codec),
    )

    # Build a model
    model = JPEGDoubleCompression(
        method=args.uncertainty_method,
        patch_size=args.patch_size,
        **args.parameters
    )

    if args.load_model:
        model.load_model(os.path.abspath(args.load_model))
    # TODO there is still some hard-coding left, like codec below.
    else:
        loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        optimizer = tf.keras.optimizers.Adam(args.lr)

        model._model.compile(optimizer, loss=loss_criterion,
                             metrics=["accuracy"])
        save_freq = args.save_every * args.n_train_images // args.batch_size
        callbacks = get_callbacks(
            args.save_dir,
            save_freq=save_freq,
            tensorboard=args.tensorboard,
            patience=int(args.epochs * args.patience_percent)
        ),
        train_performance = model._model.fit(
            x=data.get_training_generator(args.batch_size, args.patch_size),
            validation_data=data.get_validation_generator(args.batch_size),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbosity,
            callbacks=callbacks,
            steps_per_epoch=args.n_train_images // args.batch_size,
            validation_steps=args.n_val_images // args.batch_size
        )

        # save the training performance
        fig = perf(train_performance.history)
        fig.savefig(os.path.join(args.save_dir, "training_progress.pdf"))

    # TODO Add calibration
    # TODO add a dataset for calibration specifically (extend class Dataset)
    # TODO include calibration to the BayesBaseModel
    # if args.calibrate:
    #     model.set_temp(data)

    logger.info("Started Testing")
    accuracies = validate(
        model=model,
        data=data,
        cache=cache,
        **flags
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
