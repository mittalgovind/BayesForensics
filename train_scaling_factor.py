#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import argparse
import sys
import os

# necessary here, as slurm executes a copy
sys.path.append(os.path.abspath("/"))

# External libraries
import numpy as np

# Internal libraries

from helpers.dataset import Dataset
from helpers.results_data import ResultCache
from helpers.tf_helpers import disable_gpu
from workflows.bayes_scaling_factor import SFPDeepEnsemble
from workflows.bayes_scaling_factor import (
    train,
    run_tests,
    SFP,
    BayarStammSFP,
    BayarStammCalibrated,
)
from models.bayes.temp_scaling import TemperatureScaling


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
        help="Sampling method. Can be 'nearest', 'bilinear', 'bicubic', 'lanczos3' or 'random'",
    )
    parser.add_argument(
        "-s",
        "--scales",
        dest="scales",
        action="store",
        default="0.25,1.0",
        type=str,
        help="Comma separated values for lower and upper bound of scales, e.g. '0.25,1.0'",
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
        "-nr",
        "--num-runs",
        dest="n_runs",
        action="store",
        default=50,
        type=int,
        help="Number of test runs per image in validation set",
    )
    parser.add_argument(
        "-um",
        "--uncertainty-method",
        action="store",
        default="mc-dropout",
        type=str,
        help="Uncertainty method. Can be 'mc-dropout', 'flipout', 'vanilla'",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Calibrate model using temperature scaling.",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output", help="Output save directory"
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
        "-lr", "--lr", action="store", default=1e-3, type=float, help="Learning_rate"
    )
    parser.add_argument(
        "--only-eval",
        default=False,
        action="store_true",
        help="Only evaluate passed model",
    )
    parser.add_argument(
        "--cont-model-path",
        type=str,
        help="Path to a partially trained model",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output folder, if exists.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Change json to npz
    if os.path.isdir(os.path.abspath(args.save_dir)) and not args.overwrite:
        raise IsADirectoryError(
            "Output directory exists! Use --overwrite or provide another directory name."
        )

    scales = (float(args.scales.split(",")[0]), float(args.scales.split(",")[1]))
    patch_size = 128
    methods = ["nearest", "bilinear", "bicubic", "lanczos3", "random"]
    cache = ResultCache(["{step}_{sampling_method}.npz"], prefix=args.save_dir)
    classes = np.linspace(*scales, num=args.n_classes)
    flags = {
        "lr": args.lr,
        "patch_size": patch_size,
        "scales": scales,
        "sampling_method": args.sampling_method,
        "classes": classes,
        "save_dir": args.save_dir,
        "methods": methods,
        "save_every": args.save_every,
    }
    n_runs = args.n_runs

    data = Dataset(
        data_directory=args.data_dir,
        load="y",
        n_images=args.n_train_images,
        v_images=args.n_val_images,
        randomize=69,
    )

    if args.uncertainty_method == "ensemble":
        model = SFPDeepEnsemble(
            SFP(
                args.uncertainty_method,
                c_filters=(32, 32, 32, 32),
                d_filters=(32, 16, args.n_classes),
                kernel=5,
                activation="leaky_relu",
                trainable_residual=True,
                drop=0.1,
                append_rgb=False,
            ),
            5,
        )
        model.train(args.epochs, data, args.batch_size, cache, **flags)
        n_runs = 1

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
    else:
        model = BayarStammSFP(
            method=args.uncertainty_method,
            bayesian=False,
            n_classes=args.n_classes,
            patch_size=128,
        )

    if args.cont_model_path:
        # train for an epoch so that model is built
        model = train(model, 1, data, args.batch_size, cache=None, **flags)
        model.load_model(os.path.abspath(args.cont_model_path))

    if not args.only_eval:
        model = train(model, args.epochs, data, args.batch_size, cache, **flags)

    if args.calibrate:
        temp_model = BayarStammCalibrated(model, batch_size=args.batch_size)
        temp_model.set_temp(data)
        temperature = temp_model.temperature
    else:
        temperature = 1.0

    for method in methods:
        run_tests(
            model,
            method,
            data,
            methods,
            classes,
            args.n_val_images,
            patch_size,
            n_runs,
            cache,
            temperature,
        )


if __name__ == "__main__":
    main()
