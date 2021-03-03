#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os

# External libraries
import numpy as np

# Internal libraries
import helpers.plots as plots


def qf_plot(qf, accuracies, save_dir):
    """plotting function for confusion matrix between quality factors"""
    n_factors = len(qf)
    m_accuracy = np.mean(accuracies[np.tri(qf[1] - qf[0], dtype=np.bool)])

    fig, axes = plots.sub(1)
    plots.image(accuracies, f"accuracy={m_accuracy:.2f} : []", axes=axes[0])

    axes[0].invert_yaxis()

    axes[0].set_xticks(range(0, n_factors, 5))
    axes[0].set_xticklabels(qf[::5])
    axes[0].set_yticks(range(0, len(qf), 5))
    axes[0].set_yticklabels(qf[::5])

    axes[0].set_ylabel("$Q_2$")
    axes[0].set_xlabel("$Q_1$")
    axes[0].plot([0, n_factors - 1], [0, n_factors - 1], "r:")

    fig.savefig(os.path.join(save_dir, "conf_matrix.pdf"))
