#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os

# External libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Internal libraries
import helpers.plots as plots


def qf_plot(qf, accuracies, save_dir):
    """plotting function for confusion matrix between quality factors"""
    qf = np.array(qf)
    n_factors = len(qf)
    if qf[1] - qf[0] > 1:
        step = 1
    else:
        step = 5
    if n_factors < 0:
        raise ValueError("Specify correct range of QF.")

    ax = plt.subplot()
    total = np.sum(accuracies) / (len(accuracies) ** 2 - len(accuracies))
    plots.image(
        accuracies, f"accuracy={total:.2f} : []",
        axes=ax, cmap="seismic"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.05)

    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.pyplot.cm.seismic,
                                   norm=mpl.colors.Normalize(vmin=0, vmax=1),
                                   spacing='proportional')
    ax.set_xticks(range(0, n_factors, step))
    ax.set_xticklabels(qf[::step])

    ax.set_yticks(range(0, n_factors, step))
    ax.set_yticklabels(qf[::step])

    ax.set_ylabel("$Q_1$")
    ax.set_xlabel("$Q_2$")
    ax.plot([0, n_factors - 1], [0, n_factors - 1], "r:")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "conf_matrix.png"))
