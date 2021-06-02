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
        "-um",
        "--uncertainty-method",
        action="store",
        default="vanilla",
        type=str,
        help="Uncertainty method."
             " Can be 'vanilla', 'mc-dropout', 'temp-scaling', 'mc-temp',"
             " 'flipout', 'ensemble', or 'reparameterization'.",
    )
    parser.add_argument(
        "--qf-train",
        dest="qf_train",
        action="store",
        default="75,100",
        type=str,
        help="Comma separated values for lower and upper bound of Quality "
             "factor and step used for training, e.g. '75,100' or '75,95,5'",
    )
    parser.add_argument(
        "--qf-test",
        dest="qf_test",
        action="store",
        default="60,100",
        type=str,
        help="Comma separated values for lower and upper bound of Quality "
             "factor and step used for training, e.g. '60,100' or '60,95,5'",
    )
    parser.add_argument(
        "--codec",
        action="store",
        default="soft",
        type=str,
        help="Type of codec. Possible choices - libjpeg, soft, sin, harmonic."
             " (default: soft)",
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
        "--seed",
        default=69,
        type=int,
        help="Seed used for randomization. (default: 69)",
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
        "-lr", "--lr", action="store", default=1e-3, type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--validation-freq",
        default=50,
        type=int,
        help="Number of training epochs to run before a new validation run.",
    )

    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to a trained model.",
        default=None,
    )
    parser.add_argument(
        "--parameters", type=str, default=None,
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
        "--verbose",
        type=int,
        default=2,
        help="Controls verbosity of training.",
    )
    parser.add_argument(
        "--presample",
        type=int,
        default=0,
        help="Number of patches to presample per training image (default :0).",
    )
    parser.add_argument(
        "--patience-percent",
        type=float,
        default=1,
        help="Percentage of total epochs to use as patience (def : 1 or None)."
    )
    parser.add_argument(
        "--use-presampled",
        default=None,
        type=str,
        help="Uses presampled data. Pass the path to npy file.",
    )
    parser.add_argument(
        "--n-models",
        dest="num_models",
        default=1,
        type=int,
        help="Number of models to use for ensemble."
    )

    return parser.parse_args()


def load_parameters(parameters):
    """Load parameters from the config file"""
    # TODO (Govind) Change to the new standard parameters from sensor branch.
    if parameters:
        f = open(parameters, 'r')
    else:
        f = open('config/jpeg_double/default_params.json', 'r')

    try:
        parameters = json.load(f)
        f.close()
        logger.info(
            'Model configuration loaded successfully from {}.'.format(f))
        logger.info("Model parameters : {}".format(parameters))
    except RuntimeError:
        logger.error("Cannot load parameter configuration.")

    return parameters
