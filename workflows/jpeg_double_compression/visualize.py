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
    q_factors = np.arange(*qf)
    n_factors = len(q_factors)
    step = q_factors[2] if len(qf) > 2 and qf[2] != 1 else 5
    if n_factors < 0:
        raise ValueError("Specify correct range of QF.")

    fig, axes = plots.sub(1)
    plots.image(
        accuracies, f"accuracy={accuracies.mean():.2f} : []", axes=axes[0], cmap="seismic"
    )

    axes[0].set_xticks(range(0, n_factors, step))
    axes[0].set_xticklabels(q_factors[::step])

    axes[0].set_yticks(range(0, n_factors, step))
    axes[0].set_yticklabels(q_factors[::step])

    axes[0].set_ylabel("$Q_1$")
    axes[0].set_xlabel("$Q_2$")
    axes[0].plot([0, n_factors - 1], [0, n_factors - 1], "r:")

    fig.savefig(os.path.join(save_dir, "conf_matrix.png"))
