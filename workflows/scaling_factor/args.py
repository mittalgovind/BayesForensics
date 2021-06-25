#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import argparse
import json

# External libraries
from loguru import logger


# Internal libraries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a bayesian NN on different methods for downsampling"
    )
    parser.add_argument(
        "-sm",
        "--sampling-method",
        dest="sampling_method",
        action="store",
        default="nearest",
        type=str,
        help="Sampling method."
             "Can be 'nearest', 'bilinear', 'bicubic', 'lanczos3' or 'random'",
    )
    parser.add_argument(
        "-um",
        "--uncertainty-method",
        action="store",
        default="vanilla",
        type=str,
        help="Uncertainty method. Can be 'vanilla', 'mc-dropout','flipout',"
             " 'reparameterization'.",
    )
    parser.add_argument(
        "-s",
        "--scales",
        dest="scales",
        action="store",
        default="0.25,1.0",
        type=str,
        help="Comma separated values for lower and upper bound of scales,"
             " e.g. '0.25,1.0'",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        action="store",
        default=100,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--patch-size",
        dest="patch_size",
        action="store",
        default=128,
        type=int,
        help="Square patch size to be sampled from images (default: 64)",
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
        "-c",
        "--classes",
        dest="n_classes",
        action="store",
        default=31,
        type=int,
        help="Number of classes in the range defined by scales argument",
    )
    parser.add_argument(
        "-nt",
        "--train-images",
        dest="n_train_images",
        action="store",
        default=10240,
        type=int,
        help="Number of images for training set",
    )
    parser.add_argument(
        "-nv",
        "--validation-images",
        dest="n_val_images",
        action="store",
        default=1024,
        type=int,
        help="Number of images for validation set",
    )
    parser.add_argument(
        "-nr",
        "--num-runs",
        dest="num_runs",
        action="store",
        default=10,
        type=int,
        help="Number of test runs per image in validation set",
    )
    parser.add_argument(
        "--seed",
        default=69,
        type=int,
        help="Seed used for randomization. (default: 69)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output",
        help="Output save directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for getting images from.",
    )
    parser.add_argument(
        "--parameters", type=str, default=None,
        help="path to a parameters JSON file."
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
        "-lr", "--lr", action="store", default=1e-3, type=float,
        help="Learning_rate"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to a trained model.",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output folder, if exists.",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        dest="epsilon",
        action="store",
        default=0.01,
        help="Epsilon value used when generating adversarial training examples.",
    )
    parser.add_argument(
        "--validation-freq",
        default=1,
        type=int,
        help="Number of training epochs to run before a new validation run.",
    )
    parser.add_argument(
        "--jpeg-compression",
        default=False,
        action="store_true",
        dest="jpeg_compression",
        help="Use JPEG compression",
    )
    parser.add_argument(
        "--codec",
        action="store",
        default="soft",
        type=str,
        help="Type of codec. Possible choices - libjpeg, soft, sin, harmonic."
             " (default: libjpeg)",
    )
    parser.add_argument(
        "--jpeg-quality",
        "-jq",
        default=95,
        action="store",
        type=int,
        dest="jpeg_quality",
        help="Quality factor for jpeg compression.",
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
        "--xla",
        action="store_true",
        default=False,
        help="Enables GPU utilization.",
    )
    parser.add_argument(
        "--gamma",
        action="store_true",
        default=False,
        help="Data augmentation using gamma correction.",
    )
    parser.add_argument(
        "--brighten",
        action="store_true",
        default=False,
        help="Data augmentation using brightening",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        default=False,
        help="Data augmentation using rotation",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Controls verbosity of training.",
    )
    parser.add_argument(
        "--patience-percent",
        type=float,
        default=1.0,
        help="Percentage of total epochs to use as patience. (def : 1 or None).",
    )
    parser.add_argument(
        "--n-models",
        dest="num_models",
        default=1,
        type=int,
        help="Number of models to use for ensemble."
    )
    parser.add_argument(
        "--use-presampled",
        default=None,
        type=str,
        help="Uses presampled data. Pass the path to npy file.",
    )
    parser.add_argument(
        "--test-scales",
        dest="test_scales",
        action="store",
        default="0.25,1.0",
        type=str,
        help="Comma separated values for lower and upper bound of scales,"
             " e.g. '0.25,1.0'",
    )
    parser.add_argument(
        "--test-classes",
        dest="test_n_classes",
        action="store",
        default=31,
        type=int,
        help="Number of classes in the range defined by scales argument",
    )
    return parser.parse_args()


def load_parameters(parameters):
    """Load parameters from the config file"""
    # TODO (Govind) Change to the new standard parameters from sensor branch.
    if parameters:
        f = open(parameters, 'r')
    else:
        f = open('config/scaling_factor/default.json', 'r')

    try:
        parameters = json.load(f)
        f.close()
        logger.info(
            'Model configuration loaded successfully from {}.'.format(f))
        logger.info("Model parameters : {}".format(parameters))
    except RuntimeError:
        logger.error("Cannot load parameter configuration.")

    return parameters
