#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import argparse

# External libraries
import tensorflow as tf
import numpy as np

# Internal libraries

from trainer import train, run_tests
from model import SFP
from helpers.dataset import Dataset


def main():
    parser = argparse.ArgumentParser(
        description='Train a bayesian NN on different methods for downsampling')
    parser.add_argument('-sm', '--sampling-method', dest='sampling_method',
                        action='store', default='nearest', type=str,
                        help="Sampling method. Can be 'nearest', 'bilinear', 'bicubic', 'lanczos3' or 'random'")
    parser.add_argument('-s', '--scales', dest='scales', action='store',
                        default='0.25,1.0', type=str,
                        help="Comma separated values for lower and upper bound of scales, e.g. '0.25,1.0'")
    parser.add_argument('-e', '--epochs', dest='epochs', action='store',
                        default=10000, type=int,
                        help="Number of epochs")
    parser.add_argument('-bs', '--batch-size', dest='batch_size',
                        action='store', default=64, type=int,
                        help="Batch size")
    parser.add_argument('-c', '--classes', dest='n_classes', action='store',
                        default=31, type=int,
                        help="Number of classes in the range defined by scales argument")
    parser.add_argument('-nt', '--train-images', dest='n_train_images',
                        action='store', default=1000, type=int,
                        help="Number of images for training set")
    parser.add_argument('-nv', '--validation-images', dest='n_val_images',
                        action='store', default=250, type=int,
                        help="Number of images for validation set")
    parser.add_argument('-nr', '--num-runs', dest='n_runs', action='store',
                        default=50, type=int,
                        help="Number of test runs per image in validation set")
    parser.add_argument('-um', '--uncertainty-method',
                        action='store', default='mc-dropout', type=str,
                        help="Uncertainty method. Can be 'mc-dropout' or 'flipout'")
    parser.add_argument('--save-dir', type=str, default='./output',
                        help='Output save directory')
    parser.add_argument('--data-dir', type=str,
                        default='/home/govind/Workspace/neural-imaging-dev/data/rgb/native12k',
                        help='Data directory for getting images from.')
    args = parser.parse_args()

    # Make argument for directory to store results
    # Change json to npz

    scales = (float(args.scales.split(',')[0]), float(args.scales.split(',')[1]))
    patch_size = 128
    methods = ['nearest', 'bilinear', 'bicubic', 'lanczos3']

    data = Dataset(data_directory=args.data_dir, load='y',
                   n_images=args.n_train_images, v_images=args.n_val_images,
                   randomize=69)
    classes = np.linspace(*scales, num=args.n_classes)

    model = SFP(args.uncertainty_method, c_filters=(32, 32, 32, 32),
                d_filters=(32, 16, args.n_classes), kernel=5,
                activation='leaky_relu', trainable_residual=True,
                drop=0.1, append_rgb=False)

    flags = {'lr': 1e-3, 'patch_size': patch_size, 'scales': scales,
             'sampling_method': args.sampling_method, 'classes': classes,
             'save_dir': args.save_dir}
    model = train(model, args.epochs, data, args.batch_size, **flags)

    # run_tests(model, args.sampling_method, data, methods, classes,
    #           args.n_val_images, patch_size, num_runs=args.n_runs)


if __name__ == '__main__':
    main()
