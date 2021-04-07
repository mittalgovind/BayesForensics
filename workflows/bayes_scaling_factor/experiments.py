# Standard Libraries
import argparse

# External libraries
import tensorflow as tf
import numpy as np
import pickle as pkl
import seaborn as sns
import pandas as pd
from scipy.special import softmax
import matplotlib.ticker as ticker

# Internal libraries
from sfp_ensemble import SFPDeepEnsemble
from flow import SFP, BayarStammSFP
from helpers.dataset import Dataset
from helpers.uncertainty import variation_ratio, predictive_entropy, mutual_information, get_pred
from helpers.plots import sub
from models.jpeg import JPEG


####### Change to use os.join and .npz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiments for scaling factor prediction with uncertainty estimates."
    )

    parser.add_argument("--experiment", "-e", dest="experiment", type=str,
                        action="store", default=None,
                        help=("Experiment to be performed." +
                              "Can be one of: 'in_scales', 'out_scales' or 'jpeg'"))

    parser.add_argument("--models-dir", "-md", dest="models_dir", type=str,
                        action="store", default=None,
                        help="Directory where the models are stored.")

    parser.add_argument("--model-type", "-mt", dest="model_type", type=str,
                        action="store", default=None,
                        help=("Type of the models to run experiments with." +
                              "Can be one of: 'ensemble', 'mc'."))

    parser.add_argument("--results-dir", "-rd", dest="results_dir", type=str,
                        action="store", default=None,
                        help="Directory where the results of the experiments will be stored.")

    parser.add_argument("--num-samples", "-n", dest="num_samples", type=int,
                        action="store", default=None,
                        help="Number of images used in experiments.")

    parser.add_argument("--model-method", "-mm", dest="model_method", type=str,
                        action="store", default=None,
                        help="Interpolation method the model was trained on.")

    return parser.parse_args()


def get_results(model, weights_path, classes, methods, dataset, num_samples):
    outputs = []
    model_loaded = False
    for sf in classes:
        for method in methods:
            batch = dataset.next_validation_batch(0, num_samples)
            resized = tf.image.resize(
                batch,
                [int(sf * 128), int(sf * 128)],
                method=method
            )

            if not model_loaded:
                logits = model(resized, training=False)

                if model == 'ensemble':
                    model.load_weights(weights_path)
                else:
                    model.load_model(weights_path)

                model_loaded = True

            if model == 'ensemble':
                logits = model(resized, training=False)

            else:
                logits = tf.convert_to_tensor([model(resized, training=True) for i in range(5)])

            for i in range(logits.shape[1]):
                outputs.append({'logits': logits[:, i:i + 1, :],
                                'sf': sf,
                                'method': method,
                                'var_ratio': variation_ratio(logits[:, i:i + 1, :]),
                                'pred_ent': predictive_entropy(logits[:, i:i + 1, :]),
                                'mut_info': mutual_information(logits[:, i:i + 1, :])})

    return outputs


def get_results_jpeg(model, model_type, method, weights_path, classes, dataset, num_samples, qualities):
    outputs = []
    model_loaded = False

    for sf in classes:
        for quality in qualities:
            codec = JPEG(quality=quality, codec="libjpeg")

            batch = dataset.next_validation_batch(0, num_samples)
            resized = tf.image.resize(
                batch,
                [int(sf * 128), int(sf * 128)],
                method=method
            )
            resized = codec.process(resized)

            if not model_loaded:
                logits = model(resized, training=False)

                if model_type == 'ensemble':
                    model.load_weights(weights_path)
                else:
                    model.load_model(weights_path)

                model_loaded = True

            if model_type == 'ensemble':
                logits = model(resized, training=False)

            else:
                logits = tf.convert_to_tensor([model(resized, training=True) for i in range(5)])

            for i in range(logits.shape[1]):
                outputs.append({'logits': logits[:, i:i + 1, :],
                                'sf': sf,
                                'quality': quality,
                                'var_ratio': variation_ratio(logits[:, i:i + 1, :]),
                                'pred_ent': predictive_entropy(logits[:, i:i + 1, :]),
                                'mut_info': mutual_information(logits[:, i:i + 1, :])})

    return outputs


def in_scales_experiment(model_type, models_dir, results_dir, num_samples):
    data = Dataset(
        data_directory='data/rgb/native12k',
        load='y',
        n_images=1024,
        v_images=num_samples,
        randomize=69
    )

    if model_type == 'ensemble':
        model = SFPDeepEnsemble(
            SFP(
                'vanilla',
                c_filters=(32, 32, 32, 32),
                d_filters=(32, 16, 31),
                kernel=5,
                activation="leaky_relu",
                trainable_residual=True,
                drop=0.1,
                append_rgb=False
            ),
            5
        )
    else:
        model = BayarStammSFP(
            method=model_type,
            n_classes=31,
            patch_size=128
        )

    scales = (0.25, 1.0)
    classes = np.linspace(*scales, num=31)
    methods = ['random', 'nearest', 'bilinear', 'bicubic', 'lanczos3']

    for model_method in methods:
        results = get_results(model, f'{models_dir}/{model_method}', classes, methods[1:], data, num_samples)

        with open(f'{results_dir}/{model_method}.pkl', 'wb') as f:
            pkl.dump(results, f)


def out_scales_experiment(model_type, models_dir, results_dir, num_samples):
    data = Dataset(
        data_directory='data/rgb/native12k',
        load='y',
        n_images=1024,
        v_images=num_samples,
        randomize=69
    )

    if model_type == 'ensemble':
        model = SFPDeepEnsemble(
            SFP(
                'vanilla',
                c_filters=(32, 32, 32, 32),
                d_filters=(32, 16, 31),
                kernel=5,
                activation="leaky_relu",
                trainable_residual=True,
                drop=0.1,
                append_rgb=False
            ),
            5
        )
    else:
        model = BayarStammSFP(
            method=model_type,
            n_classes=31,
            patch_size=128
        )

    scales = (0.25, 1.0)
    lower_scales = (0.1, 0.245)
    higher_scales = (1.1, 5.0)
    classes = np.linspace(*scales, num=61)
    higher = np.linspace(*higher_scales, num=20)
    lower = np.linspace(*lower_scales, num=10)
    classes = np.concatenate((classes, higher, lower))
    methods = ['random', 'nearest', 'bilinear', 'bicubic', 'lanczos3']

    for model_method in methods:
        results = get_results(model, f'{models_dir}/{model_method}', classes, methods[1:], data, num_samples)

        with open(f'{results_dir}/{model_method}.pkl', 'wb') as f:
            pkl.dump(results, f)


def jpeg_experiment(model_type, model_method, models_dir, results_dir, num_samples):
    data = Dataset(
        data_directory='data/rgb/native12k',
        load='y',
        n_images=1024,
        v_images=num_samples,
        randomize=69
    )

    if model_type == 'ensemble':
        model = SFPDeepEnsemble(
            SFP(
                'vanilla',
                c_filters=(32, 32, 32, 32),
                d_filters=(32, 16, 31),
                kernel=5,
                activation="leaky_relu",
                trainable_residual=True,
                drop=0.1,
                append_rgb=False
            ),
            5
        )
    else:
        model = BayarStammSFP(
            method=model_type,
            n_classes=31,
            patch_size=128
        )

    for m in model.models:
        m.create_model()

    scales = (0.25, 1.0)
    classes = np.linspace(*scales, num=31)
    methods = ['random', 'nearest', 'bilinear', 'bicubic', 'lanczos3']
    qualities = [100, 90, 80, 70]

    results = get_results_jpeg(model, model_type, model_method, f'{models_dir}/{model_method}', classes, data,
                               num_samples, qualities)

    with open(f'{results_dir}/{model_method}.pkl', 'wb') as f:
        pkl.dump(results, f)


def main():
    args = parse_args()

    if args.experiment == "in_scales":
        in_scales_experiment(model_type=args.model_type,
                             models_dir=args.models_dir,
                             results_dir=args.results_dir,
                             num_samples=args.num_samples)

    elif args.experiment == "out_scales":
        out_scales_experiment(model_type=args.model_type,
                              models_dir=args.models_dir,
                              results_dir=args.results_dir,
                              num_samples=args.num_samples)

    elif args.experiment == "jpeg":
        jpeg_experiment(model_type=args.model_type,
                        model_method=args.model_method,
                        models_dir=args.models_dir,
                        results_dir=args.results_dir,
                        num_samples=args.num_samples)


if __name__ == "__main__":
    main()
