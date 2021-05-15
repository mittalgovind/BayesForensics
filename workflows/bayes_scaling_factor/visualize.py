from helpers.uncertainty import (
    variation_ratio,
    predictive_entropy,
    mutual_information,
    get_pred,
)
import tensorflow as tf
import numpy as np
from helpers.plots import sub, confusion_matrix, image
import seaborn as sns
import os


def get_uncertainties(summary, classes):
    results = []
    classes = (classes * 128).astype(int)

    for method, method_value in summary.iter_items():
        for sf, logits in method_value.iter_items():
            correct = np.where(classes == elm["sf"])[0][0]

            if len(elm["logits"].shape) == 2:
                logits = tf.convert_to_tensor([elm["logits"]])
            else:
                logits = elm["logits"]

            pred = get_pred(logits)
            vr = variation_ratio(logits)
            pe = predictive_entropy(logits)
            mi = mutual_information(logits)

            for i in range(len(pred)):
                results.append(
                    {
                        "method": elm["method"],
                        "correct": correct,
                        "pred": pred[i][0],
                        "variation_ratio": vr[i],
                        "predictive_entropy": pe[i],
                        "mutual_information": mi[i],
                    }
                )

    return results


def sf_plot(summary, classes, training_method, save_dir):
    summary = get_uncertainties(summary, classes)

    text_classes = [f"{x:.2f}" for x in classes]
    methods = ["nearest", "bilinear", "bicubic", "lanczos3"]
    method_titles = ["Nearest", "Bilinear", "Bicubic", "Lanczos 3"]
    measure_titles = [
        "Accuracy",
        "Variation Ratio",
        "Predictive Entropy",
        "Mutual Information",
    ]

    acc = [np.zeros((len(classes), len(classes))) for _ in methods]

    vr = [{} for _ in methods]
    pe = [{} for _ in methods]
    mi = [{} for _ in methods]

    for elm in summary:
        ind = methods.index(elm["method"])

        acc[ind][elm["correct"], elm["pred"]] += 1 / (
                len(summary) / len(classes) / len(methods)
        )

        err = np.abs(elm["correct"] - elm["pred"])
        if err not in vr[ind].keys():
            vr[ind][err] = []
            pe[ind][err] = []
            mi[ind][err] = []

        vr[ind][err].append(elm["variation_ratio"])
        pe[ind][err].append(elm["predictive_entropy"])
        mi[ind][err].append(elm["mutual_information"])

    for i in range(len(methods)):
        for j in vr[i].keys():
            vr[i][j] = np.mean(vr[i][j])
            pe[i][j] = np.mean(pe[i][j])
            mi[i][j] = np.mean(mi[i][j])

    acc_fig, acc_axes = sub(4, ncols=2, figwidth=12)
    vr_fig, vr_axes = sub(4, ncols=2, figwidth=12)
    pe_fig, pe_axes = sub(4, ncols=2, figwidth=12)
    mi_fig, mi_axes = sub(4, ncols=2, figwidth=12)

    for i in range(len(methods)):
        confusion_matrix(
            acc[i],
            classes=text_classes,
            axes=acc_axes[i],
            title=f"Tested using {method_titles[i]}",
            cbar=False,
            cmap="Greys",
        )

        sns.lineplot(vr[i].keys(), vr[i].values(), ax=vr_axes[i])
        sns.lineplot(pe[i].keys(), pe[i].values(), ax=pe_axes[i])
        sns.lineplot(mi[i].keys(), mi[i].values(), ax=mi_axes[i])

    for axes in [acc_axes, vr_axes, pe_axes, mi_axes]:
        for i in range(len(axes)):
            if i < 2:
                axes[i].xaxis.tick_top()
                axes[i].xaxis.set_label_position("top")

            if (i % 2) == 1:
                axes[i].yaxis.tick_right()
                axes[i].yaxis.set_label_position("right")

            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation="horizontal")
            axes[i].set_title(f"Tested using {method_titles[i]}")

    figs = [acc_fig, vr_fig, pe_fig, mi_fig]
    for i in range(len(figs)):
        figs[i].suptitle(
            f"{measure_titles[i]} for model trained on {training_method}", size=24
        )

    acc_fig.savefig(os.path.join(save_dir, "acc_matrix.pdf"))
    vr_fig.savefig(os.path.join(save_dir, "vr_matrix.pdf"))
    pe_fig.savefig(os.path.join(save_dir, "pe_matrix.pdf"))
    mi_fig.savefig(os.path.join(save_dir, "mi_matrix.pdf"))


def plot_conf_matrix(conf_matrix, methods, classes, save_dir):
    acc_fig, acc_axes = sub(4, ncols=2, figwidth=12)
    text_classes = [f"{x:.2f}" for x in classes]
    method_titles = ["Nearest", "Bilinear", "Bicubic", "Lanczos 3"]
    for i in range(len(methods)):
        confusion_matrix(
            conf_matrix[i],
            classes=text_classes,
            axes=acc_axes[i],
            title=f"Tested using {method_titles[i]}",
            cbar=False,
            cmap="Greys",
        )
    acc_fig.savefig(os.path.join(save_dir, "acc_matrix.pdf"))