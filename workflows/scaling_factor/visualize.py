import os

import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd

from helpers.plots import sub, confusion_matrix
from helpers.uncertainty import (
    variation_ratio,
    predictive_entropy,
    mutual_information,
    get_pred,
)


def get_uncertainties(summary, classes):
    results = []

    for method in summary.keys():
        for sf in summary[method].keys():
            correct = np.where(classes == sf)[0][0]

            if len(summary[method][sf].shape) == 2:
                logits = tf.convert_to_tensor([summary[method][sf]])
            else:
                logits = summary[method][sf]

            pred = get_pred(logits)
            vr = variation_ratio(logits)
            pe = predictive_entropy(logits)
            mi = mutual_information(logits)

            for i in range(len(pred)):
                results.append(
                    {
                        "method": method,
                        "correct": correct,
                        "pred": pred[i][0],
                        "variation_ratio": vr[i],
                        "predictive_entropy": pe[i],
                        "mutual_information": mi[i],
                    }
                )

    return results


def uncertainty_graph(data, unc_measure, classes, methods, sampling_method, ax=None, **kwargs):
    if sampling_method == "random":
        indices = list(range(len(methods)))
    else:
        indices = [methods.index(sampling_method)]
    
    uncertainties = {}
    for m in indices:
        for sf in range(len(data[m])):
            if f"{sf}" not in uncertainties.keys():
                uncertainties[f"{sf}"] = []
            
            for i in range(data[m][sf].shape[1]):
                logits = np.array([data[m][sf][i, :]])
                unc = unc_measure(logits)[0]
                uncertainties[f"{sf}"].append(unc)
    
    uncertainties_df = pd.DataFrame(uncertainties)
    
    x = classes
    y = []
    low = []
    high = []
    for col in uncertainties_df.columns:
        mean = np.mean(uncertainties_df[col])
        y.append(mean)
        std = np.std(uncertainties_df[col])
        low.append(np.percentile(uncertainties_df[col], 5))
        high.append(np.percentile(uncertainties_df[col], 95))
    
    if ax is None:
        ax = sns.lineplot(x=x, y=y, **kwargs)
    else:
        ax = sns.lineplot(x=x, y=y, ax=ax, **kwargs)
    
    ax.fill_between(x, low, high, alpha=0.3)
    
    return ax


def sf_plot(summary, conf_matrix, classes, test_classes, training_method, save_dir, prefix=None):
    text_classes = [f"{x:.2f}" for x in classes]
    text_test_classes = [f"{x:.2f}" for x in test_classes]
    methods = ["nearest", "bilinear", "bicubic", "lanczos3"]
    method_titles = ["Nearest", "Bilinear", "Bicubic", "Lanczos 3"]
    measure_titles = [
        "Accuracy",
        "Variation Ratio",
        "Predictive Entropy",
        "Mutual Information",
    ]

    acc_fig, acc_axes = sub(4, ncols=2, figwidth=12)
    for i in range(len(methods)):
        confusion_matrix(
            conf_matrix[i].T,
            classes_x=text_classes,
            classes_y=text_test_classes,
            axes=acc_axes[i],
            title="Tested using {} - Accuracy = {:.2f}%".format(
                method_titles[i], 100*conf_matrix[i].diagonal().mean()),
            cbar=False,
            cmap="Greys",
        )
    if prefix:
        acc_fig.savefig(os.path.join(save_dir, f"{prefix}_acc_matrix.pdf"))
    else:
        acc_fig.savefig(os.path.join(save_dir, f"acc_matrix.pdf"))
    
    unc_fig, unc_axes = sub(3, figwidth=16, ncols=1)
    unc_functions = [variation_ratio, predictive_entropy, mutual_information]
    unc_titles = ['Variation Ratio', 'Predictive Entropy', 'Mutual Information']

    for i in range(len(unc_axes)):
        uncertainty_graph(
            summary,
            unc_functions[i],
            test_classes,
            methods,
            training_method,
            ax=unc_axes[i]
        )
        unc_axes[i].set_title(unc_titles[i])
        
    if prefix:
        unc_fig.savefig(os.path.join(save_dir, f"{prefix}_uncertainties.pdf"))
    else:
        unc_fig.savefig(os.path.join(save_dir, "uncertainties.pdf"))
    
