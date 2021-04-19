from helpers.uncertainty import variation_ratio, predictive_entropy, mutual_information, get_pred
import tensorflow as tf
import numpy as np
import pandas as pd
from helpers.plots import sub
import seaborn as sns
import matplotlib.cm as cm
import os


def get_uncertainties(data, classes):
    results = []
    classes = (classes * 128).astype(int)

    for elm in data['runs']:
        correct = np.where(classes == elm['sf'])[0][0]

        logits = tf.convert_to_tensor([elm['logits']])

        pred = get_pred(logits)
        vr = variation_ratio(logits)
        pe = predictive_entropy(logits)
        mi = mutual_information(logits)

        for i in range(len(pred)):
            results.append({
                'method': elm['method'],
                'correct': correct,
                'pred': pred[i][0],
                'variation_ratio': vr[i],
                'predictive_entropy': pe[i],
                'mutual_information': mi[i]
            })

    return results


def sf_plot(data, classes, training_method, save_dir):
    data = get_uncertainties(data, classes)

    classes = [f'{x:.2f}' for x in classes]
    methods = ['nearest', 'bilinear', 'bicubic', 'lanczos3']
    method_titles = ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos 3']
    measure_titles = ['Accuracy', 'Variation Ratio', 'Predictive Entropy', 'Mutual Information']

    acc = [np.zeros((len(classes), len(classes))) for _ in methods]
    vr = [np.zeros((len(classes), len(classes))) for _ in methods]
    pe = [np.zeros((len(classes), len(classes))) for _ in methods]
    mi = [np.zeros((len(classes), len(classes))) for _ in methods]

    for elm in data:
        ind = methods.index(elm['method'])

        acc[ind][elm['correct'], elm['pred']] += 1 / (len(data) / len(classes) / len(methods))
        vr[ind][elm['correct'], elm['pred']] += elm['variation_ratio'] / (len(data) / len(classes) / len(methods))
        pe[ind][elm['correct'], elm['pred']] += elm['predictive_entropy'] / (len(data) / len(classes) / len(methods))
        mi[ind][elm['correct'], elm['pred']] += np.abs(elm['mutual_information']) / (
                    len(data) / len(classes) / len(methods))

    acc_fig, acc_axes = sub(4, ncols=2, figwidth=12)
    vr_fig, vr_axes = sub(4, ncols=2, figwidth=12)
    pe_fig, pe_axes = sub(4, ncols=2, figwidth=12)
    mi_fig, mi_axes = sub(4, ncols=2, figwidth=12)

    for i in range(len(methods)):
        sns.heatmap(pd.DataFrame(acc[i], columns=classes, index=classes), ax=acc_axes[i], cbar=False, cmap='Greys')
        sns.heatmap(pd.DataFrame(vr[i], columns=classes, index=classes), ax=vr_axes[i], cbar=False, cmap='Greys')
        sns.heatmap(pd.DataFrame(pe[i], columns=classes, index=classes), ax=pe_axes[i], cbar=False, cmap='Greys')
        sns.heatmap(pd.DataFrame(mi[i], columns=classes, index=classes), ax=mi_axes[i], cbar=False, cmap='Greys')

    for axes in [acc_axes, vr_axes, pe_axes, mi_axes]:
        for i in range(len(axes)):
            if i < 2:
                axes[i].xaxis.tick_top()
                axes[i].xaxis.set_label_position('top')

            if (i % 2) == 1:
                axes[i].yaxis.tick_right()
                axes[i].yaxis.set_label_position('right')

            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation='horizontal')
            axes[i].set_title(f'Tested using {method_titles[i]}')

    figs = [acc_fig, vr_fig, pe_fig, mi_fig]
    for i in range(len(figs)):
        figs[i].suptitle(f'{measure_titles[i]} for model trained on {training_method}', size=24)

    acc_fig.savefig(os.path.join(save_dir, 'acc_matrix.pdf'))
    vr_fig.savefig(os.path.join(save_dir, 'vr_matrix.pdf'))
    pe_fig.savefig(os.path.join(save_dir, 'pe_matrix.pdf'))
    mi_fig.savefig(os.path.join(save_dir, 'mi_matrix.pdf'))
