#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import matplotlib
from matplotlib import figure
from matplotlib.backends import backend_agg
import seaborn as sns
import tensorflow as tf
import numpy as np

# Internal libraries

matplotlib.use('Agg')
IMAGE_SHAPE = [28, 28, 1]

# TODO (Govind) change this file and make generic

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.

    Args:
      names: A Python `iterable` of `str` variable names.
        qm_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior means of weight varibles.
      qs_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior standard deviations of weight varibles.
      fname: Python `str` filename to save the plot to.
    """
    fig = figure.Figure(figsize=(6, 3 * len(qs_vals)))
    canvas = backend_agg.FigureCanvasAgg(fig)
    colors = sns.color_palette(n_colors=len(qs_vals))
    for i, (n, qm, qs) in enumerate(zip(names, qm_vals, qs_vals)):
        ax = fig.add_subplot(6, 2, 2 * i + 1)
        sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n,
                     color=colors[i])
        ax.set_title('weight means')
        ax.set_xlim([-1.5, 1.5])
        ax.legend()

        ax = fig.add_subplot(6, 2, 2 * i + 2)
        sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax, color=colors[i])
        ax.set_title('weight stddevs')
        ax.set_xlim([0, 1.])

    fig.tight_layout()
    canvas.print_figure(fname, format='png')
    print('saved {}'.format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=''):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.

    Args:
      input_vals: A `float`-like Numpy `array` of shape
        `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
      probs: A `float`-like Numpy array of shape `[num_monte_carlo,
        num_heldout, num_classes]` containing Monte Carlo samples of
        class probabilities for each heldout sample.
      fname: Python `str` filename to save the plot to.
      n: Python `int` number of datapoints to vizualize.
      title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(9, 3 * n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3 * i + 1)
        ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]),
                  interpolation='None')

        ax = fig.add_subplot(n, 3, 3 * i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title('posterior samples')

        ax = fig.add_subplot(n, 3, 3 * i + 3)
        sns.barplot(np.arange(10), tf.reduce_mean(probs[:, i, :], axis=0),
                    ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title('predictive probs')
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format='png')
    print('saved {}'.format(fname))
