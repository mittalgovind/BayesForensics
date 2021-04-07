# Standard Libraries
import argparse
import os

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graphs for experiments for scaling factor prediction with uncertainty estimates."
    )

    parser.add_argument("--figs", "-f", dest="figs", type=str,
                        action="store", default=None,
                        help=("Figures to be generated." +
                              "Can be one of: 'quant', 'acc_unc', 'unc_method', 'quant_jpeg' or 'unc_sf'"))

    parser.add_argument("--results-dir", "-rd", dest="results_dir", type=str,
                        action="store", default=None,
                        help="Directory where the results of the experiments are stored.")

    parser.add_argument("--num-samples", "-n", dest="num_samples", type=int,
                        action="store", default=None,
                        help="Number of images used in experiments.")

    parser.add_argument("--figs-dir", "-fd", dest="figs_dir", type=str,
                        action="store", default=None,
                        help="Directory where the figures will be stored")

    parser.add_argument("--model-method", "-mm", dest="model_method", type=str,
                        action="store", default=None,
                        help="Interpolation method used for training.")

    return parser.parse_args()


def sort_uncertainty_by_value(data, uncertainty, step=0.1):
    sorted_results = {}
    for elm in data:
        for s in np.arange(step, 5, step):
            if elm[uncertainty] < s:
                if s not in sorted_results.keys():
                    sorted_results[s] = []
                sorted_results[s].append(elm)
                break

    return sorted_results


def get_uncertainties_by_method(experiment_list):
    var_ratio = {'nearest': [], 'bilinear': [], 'bicubic': [], 'lanczos3': []}
    pred_ent = {'nearest': [], 'bilinear': [], 'bicubic': [], 'lanczos3': []}
    mut_info = {'nearest': [], 'bilinear': [], 'bicubic': [], 'lanczos3': []}
    for elm in experiment_list:
        var_ratio[elm['method']].append(elm['var_ratio'][0])
        pred_ent[elm['method']].append(elm['pred_ent'][0])
        mut_info[elm['method']].append(elm['mut_info'][0])

    return {'var_ratio': var_ratio, 'pred_ent': pred_ent, 'mut_info': mut_info}


def get_accuracy(experiments):
    scales = (0.25, 1.0)
    classes = np.linspace(*scales, num=31)

    acc = 0
    for elm in experiments:
        pred = get_pred(elm['logits'])[0]
        if np.isclose(classes[pred], elm['sf']):
            acc += 1
    acc = acc / len(experiments)

    return acc


def acc_vs_uncertainty_graph(experiments):
    fig, axes = sub(3, ncols=3)
    uncs = ['var_ratio', 'pred_ent', 'mut_info']
    for i in range(len(uncs)):
        unc_buckets = sort_uncertainty_by_value(experiments, uncs[i], step=0.05)
        unc_buckets_acc = {}
        for bucket in list(unc_buckets.keys()):
            unc_buckets_acc[bucket] = get_accuracy(unc_buckets[bucket])

        sorted_buckets = {k: unc_buckets_acc[k] for k in sorted(unc_buckets_acc)}

        x = np.array(list(sorted_buckets.keys()))
        y = list(sorted_buckets.values())

        axes[i].plot(x, y)

        axes[i].set_ylim([0, 1])

        axes[i].plot([max(x), 0], [0, 1], ls="--", c=".3")

        axes[i].yaxis.set_ticks(np.arange(0, 1.05, 0.05))

        axes[i].set_xlabel(uncs[i])

        if i == 0:
            axes[i].set_ylabel('Accuracy')

        else:
            axes[i].get_yaxis().set_visible(False)

    return fig, axes


def plot_acc_graphs(results_dir, figs_dir):
    methods = ['random', 'nearest', 'bilinear', 'bicubic', 'lanczos3']
    for method in methods:
        with open(os.path.join(results_dir, f'{method}.pkl'), 'rb') as f:
            exp_list = pkl.load(f)

        fig, axes = acc_vs_uncertainty_graph(exp_list)

        fig.suptitle(f'Accuracy vs. Uncertainty - Method: {method}')

        fig.savefig(os.path.join(figs_dir, f'acc_vs_unc_{method}.png'))

        print(f'{method} done.')


def graph_uncertainties_by_method(results_dir, figs_dir):
    fig, axes = sub(15, ncols=3)
    methods = ['random', 'nearest', 'bilinear', 'bicubic', 'lanczos3']

    for i in range(len(methods)):
        with open(os.path.join(results_dir, f'{methods[i]}.pkl'), 'rb') as f:
            exp_list = pkl.load(f)
        uncertainties = get_uncertainties_by_method(exp_list)

        sns.boxplot(data=list(uncertainties['var_ratio'].values()),
                    ax=axes[3 * i + 0],
                    showfliers=False,
                    color='steelblue')

        axes[3 * i + 0].set_ylabel(f' Trained on {methods[i]}')

        sns.boxplot(data=list(uncertainties['pred_ent'].values()),
                    ax=axes[3 * i + 1],
                    showfliers=False,
                    color='steelblue')

        sns.boxplot(data=list(np.abs(list(uncertainties['mut_info'].values()))),
                    ax=axes[3 * i + 2],
                    showfliers=False,
                    color='steelblue')

        if i == 0:
            axes[3 * i + 0].set_title('Variation Ratio')
            axes[3 * i + 1].set_title('Predictive Entropy')
            axes[3 * i + 2].set_title('Mutual Information')

        if i > 0:
            axes[3 * i + 0].artists[i - 1].set_facecolor('firebrick')
            axes[3 * i + 1].artists[i - 1].set_facecolor('firebrick')
            axes[3 * i + 2].artists[i - 1].set_facecolor('firebrick')

        if i == 4:
            axes[3 * i + 0].set_xticklabels(methods[1:])
            axes[3 * i + 1].set_xticklabels(methods[1:])
            axes[3 * i + 2].set_xticklabels(methods[1:])
        else:
            axes[3 * i + 0].get_xaxis().set_visible(False)
            axes[3 * i + 1].get_xaxis().set_visible(False)
            axes[3 * i + 2].get_xaxis().set_visible(False)

    fig.suptitle('Uncertainties by scaling method in training and testing')
    fig.savefig(os.path.join(figs_dir, 'uncertainties_by_method.png'))


def get_quantiles(experiment_list, method, uncertainty, step=0.1):
    unc_list = []
    unc_ind = []
    for i in range(len(experiment_list)):
        elm = experiment_list[i]
        if elm['method'] == method:
            unc_list.append(elm[uncertainty])
            unc_ind.append(i)

    quantile_steps = np.arange(step, 1, step)
    quantiles = np.quantile(unc_list, quantile_steps)

    indices = [(np.abs(unc_list - q)).argmin() for q in quantiles]
    indices = [unc_ind[i] for i in indices]

    return indices


def get_different_method_from_indices(experiment_list, indices, method, num_samples):
    new_indices = []
    found = False
    offset = num_samples
    for ind in indices:
        while not found:
            if ind + offset > len(experiment_list):
                offset = -num_samples * 3

            if ind + offset < 0:
                offset = 0

            elm = experiment_list[ind + offset]

            if elm['method'] == method and np.isclose(elm['sf'], experiment_list[ind]['sf']):
                found = True
                break

            if elm['sf'] > experiment_list[ind]['sf']:
                offset = -num_samples * 3
            else:
                offset += num_samples

        new_indices.append(ind + offset)

    return new_indices


def graph_quantiles(experiments_source, image_output, training_method, num_samples):
    methods = ['nearest', 'bilinear', 'bicubic', 'lanczos3']

    with open(experiments_source, 'rb') as f:
        exp_list = pkl.load(f)
    indices = get_quantiles(exp_list, training_method, 'mut_info')

    scales = (0.25, 1.0)
    classes = np.linspace(*scales, num=31)
    classes = [f'{x:.2f}' for x in classes]

    distribs = get_uncertainties_by_method(exp_list)

    method_ind = methods.index(training_method)
    other_methods = [x for i, x in enumerate(methods) if i != method_ind]

    full_indices = [indices,
                    get_different_method_from_indices(exp_list, indices, other_methods[0], num_samples),
                    get_different_method_from_indices(exp_list, indices, other_methods[1], num_samples),
                    get_different_method_from_indices(exp_list, indices, other_methods[2], num_samples)]

    full_indices = np.array(full_indices).flatten('F').tolist()

    fig, axes = sub(48, ncols=4, figwidth=12)
    for i in range(len(full_indices)):
        sm_logits = softmax(np.squeeze(exp_list[full_indices[i]]['logits']), axis=1)
        data = pd.DataFrame(sm_logits, columns=classes)
        sns.boxplot(data=data,
                    ax=axes[i],
                    showfliers=False)
        axes[i].set_ylim(top=1.1)
        axes[i].axvline(classes.index(f"{exp_list[full_indices[i]]['sf']:.2f}"))

        if i == 0:
            axes[i].set_title(f'Tested on {training_method}')
        elif 1 <= i <= 3:
            axes[i].set_title(f'Tested on {other_methods[i - 1]}')

        if i % 4 == 0:
            axes[i].set_ylabel(
                f'Image number {(full_indices[i] % 256):3d}\nScaling factor: {exp_list[full_indices[i]]["sf"]:.2f}')

    for i in range(len(methods)):
        sns.kdeplot(data=distribs['var_ratio'][methods[i]],
                    ax=axes[-12 + i], fill=True)
        axes[-12 + i].set_xlim(left=0)

        sns.kdeplot(data=distribs['pred_ent'][methods[i]],
                    ax=axes[-8 + i], fill=True)
        axes[-8 + i].set_xlim(left=0)

        sns.kdeplot(data=distribs['mut_info'][methods[i]],
                    ax=axes[-4 + i], fill=True)
        axes[-4 + i].set_xlim(left=0)

        if i > 0:
            axes[-12 + i].yaxis.label.set_visible(False)
            axes[-8 + i].yaxis.label.set_visible(False)
            axes[-4 + i].yaxis.label.set_visible(False)

        else:
            axes[-12 + i].yaxis.set_label_text('Variation Ratio')
            axes[-8 + i].yaxis.set_label_text('Predictive Entropy')
            axes[-4 + i].yaxis.set_label_text('Mutual Information')

    fig.suptitle(f'Uncertainty Quantiles - Trained on: {training_method}')
    fig.savefig(image_output)


def graph_all_quantiles(results_dir, figs_dir, num_samples):
    methods = ['nearest', 'bilinear', 'bicubic', 'lanczos3']
    for method in methods:
        graph_quantiles(os.path.join(results_dir, f'{method}.pkl'),
                        os.path.join(figs_dir, f'{method}_softmax_quantiles.png'),
                        method, num_samples)


def mean_uncertainty_by_sf(experiments, uncertainty):
    uncertainties = {}
    for elm in experiments:
        if elm['sf'] not in uncertainties.keys():
            uncertainties[elm['sf']] = []
        uncertainties[elm['sf']].append(elm[uncertainty])

    mean_uncertainties = {}
    for k in list(uncertainties.keys()):
        mean_uncertainties[k] = np.median(uncertainties[k])

    return mean_uncertainties


def graph_uncertainties_by_sf(results_dir, figs_dir):
    methods = ['random', 'nearest', 'bilinear', 'bicubic', 'lanczos3']
    uncs = ['Variation Ratio', 'Predictive Entropy', 'Mutual Information']

    for method in methods:
        with open(os.path.join(results_dir, f'{method}.pkl'), 'rb') as f:
            exp_list = pkl.load(f)

        fig, axes = sub(3, ncols=1, figwidth=12)

        uncertainties = [mean_uncertainty_by_sf(exp_list, 'var_ratio'),
                         mean_uncertainty_by_sf(exp_list, 'pred_ent'),
                         mean_uncertainty_by_sf(exp_list, 'mut_info')]

        for i in range(len(uncertainties)):
            sorted_unc = {k: uncertainties[i][k] for k in sorted(uncertainties[i])}

            x = list(sorted_unc.keys())
            y = list(sorted_unc.values())

            axes[i].plot(x, y)

            axes[i].axvspan(0.25, 1, facecolor='r', alpha=0.2)

            axes[i].xaxis.set_ticks(np.arange(0, 5.1, 0.2))
            axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            axes[i].set_ylabel(uncs[i])

            if i < 2:
                axes[i].get_xaxis().set_visible(False)

            else:
                axes[i].set_xlabel('Scaling factor')

        fig.suptitle(f'Median Uncertainties by Scaling Factor - Model: {method}')
        fig.savefig(os.path.join(figs_dir, f'unc_vs_sf_{method}.png'))

        print(f'{method} done.')


def get_jpeg_quantiles(experiment_list, quality, uncertainty, step=0.1):
    unc_list = []
    unc_ind = []
    for i in range(len(experiment_list)):
        elm = experiment_list[i]
        if elm['quality'] == quality:
            unc_list.append(elm[uncertainty])
            unc_ind.append(i)

    quantile_steps = np.arange(step, 1, step)
    quantiles = np.quantile(unc_list, quantile_steps)

    indices = [(np.abs(unc_list - q)).argmin() for q in quantiles]
    indices = [unc_ind[i] for i in indices]

    return indices


def get_uncertainties_by_quality(experiment_list):
    var_ratio = {100: [], 90: [], 80: [], 70: []}
    pred_ent = {100: [], 90: [], 80: [], 70: []}
    mut_info = {100: [], 90: [], 80: [], 70: []}
    for elm in experiment_list:
        var_ratio[elm['quality']].append(elm['var_ratio'][0])
        pred_ent[elm['quality']].append(elm['pred_ent'][0])
        mut_info[elm['quality']].append(elm['mut_info'][0])

    return {'var_ratio': var_ratio, 'pred_ent': pred_ent, 'mut_info': mut_info}


def get_different_quality_from_indices(experiment_list, indices, quality, num_samples):
    new_indices = []
    found = False
    offset = num_samples
    for ind in indices:
        while not found:
            if ind + offset > len(experiment_list):
                offset = -num_samples * 3

            if ind + offset < 0:
                offset = 0

            elm = experiment_list[ind + offset]

            if elm['quality'] == quality and np.isclose(elm['sf'], experiment_list[ind]['sf']):
                found = True
                break

            if elm['sf'] > experiment_list[ind]['sf']:
                offset = -num_samples * 3
            else:
                offset += num_samples

        new_indices.append(ind + offset)

    return new_indices


def graph_quantiles_jpeg(results_dir, figs_dir, method, num_samples):
    with open(f'{results_dir}/{method}.pkl', 'rb') as f:
        exp_list = pkl.load(f)

    qualities = [100, 90, 80, 70]

    indices = get_jpeg_quantiles(exp_list, qualities[0], 'mut_info')

    scales = (0.25, 1.0)
    classes = np.linspace(*scales, num=31)
    classes = [f'{x:.2f}' for x in classes]

    distribs = get_uncertainties_by_quality(exp_list)

    full_indices = [indices,
                    get_different_quality_from_indices(exp_list, indices, qualities[1], num_samples),
                    get_different_quality_from_indices(exp_list, indices, qualities[2], num_samples),
                    get_different_quality_from_indices(exp_list, indices, qualities[3], num_samples)]

    full_indices = np.array(full_indices).flatten('F').tolist()

    fig, axes = sub(48, ncols=4, figwidth=12)
    for i in range(len(full_indices)):
        print(exp_list[full_indices[i]]['logits'])
        sm_logits = softmax(np.squeeze(exp_list[full_indices[i]]['logits']), axis=1)
        data = pd.DataFrame(sm_logits, columns=classes)
        sns.boxplot(data=data,
                    ax=axes[i],
                    showfliers=False)
        axes[i].set_ylim(top=1.1)
        axes[i].axvline(classes.index(f"{exp_list[full_indices[i]]['sf']:.2f}"))

        if i == 0:
            axes[i].set_title(f'Tested on {qualities[0]}')
        elif 1 <= i <= 3:
            axes[i].set_title(f'Tested on {qualities[i]}')

        if i % 4 == 0:
            axes[i].set_ylabel(
                f'Image number {(full_indices[i] % 256):3d}\nScaling factor: {exp_list[full_indices[i]]["sf"]:.2f}')

    for i in range(len(qualities)):
        sns.kdeplot(data=distribs['var_ratio'][qualities[i]],
                    ax=axes[-12 + i], fill=True)
        axes[-12 + i].set_xlim(left=0)

        sns.kdeplot(data=distribs['pred_ent'][qualities[i]],
                    ax=axes[-8 + i], fill=True)
        axes[-8 + i].set_xlim(left=0)

        sns.kdeplot(data=distribs['mut_info'][qualities[i]],
                    ax=axes[-4 + i], fill=True)
        axes[-4 + i].set_xlim(left=0)

        if i > 0:
            axes[-12 + i].yaxis.label.set_visible(False)
            axes[-8 + i].yaxis.label.set_visible(False)
            axes[-4 + i].yaxis.label.set_visible(False)

        else:
            axes[-12 + i].yaxis.set_label_text('Variation Ratio')
            axes[-8 + i].yaxis.set_label_text('Predictive Entropy')
            axes[-4 + i].yaxis.set_label_text('Mutual Information')

    fig.suptitle(f'Uncertainty Quantiles - Trained on: {method}')
    fig.savefig(os.path.join(figs_dir, f'{method}_softmax_quantiles.png'))


def main():
    args = parse_args()

    if args.figs == "quant":
        graph_all_quantiles(results_dir=args.results_dir,
                            figs_dir=args.figs_dir,
                            num_samples=args.num_samples)

    elif args.figs == "acc_unc":
        plot_acc_graphs(results_dir=args.results_dir,
                        figs_dir=args.figs_dir)

    elif args.figs == "unc_sf":
        graph_uncertainties_by_sf(results_dir=args.results_dir,
                                  figs_dir=args.figs_dir)

    elif args.figs == "unc_method":
        graph_uncertainties_by_method(results_dir=args.results_dir,
                                      figs_dir=args.figs_dir)

    elif args.figs == "quant_jpeg":
        graph_quantiles_jpeg(results_dir=args.results_dir,
                             figs_dir=args.figs_dir,
                             method=args.model_method,
                             num_samples=args.num_samples)


if __name__ == "__main__":
    main()
