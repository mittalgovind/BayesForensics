# -*- coding: utf-8 -*-
"""
High-level plotting and visualization functions.

Caution
-------
The functions create unmanaged matplotlib figures to prevent RAM leaks. If managed figures are required, they need
to be passed from outside, e.g.,:

plots.images(batch, ..., fig=plt.figure())

Overview
--------
- image             - show a single image (axes-level)
- images            - show a batch of images (figure-level)
- sub               - generate a figure divided into sub-plots (ALWAYS returns a plain list of axes)
- progress          - plot training progress with moving averages (single metric)
- perf              - plot training progress with moving averages (various metrics)
- detection         - binary detection metrics (positive/negative/reference stats)
- roc               - ROC curve with basic stats (tpr, auc)
- intervals         - plots mean values with shaded percentile ranges
- correlation       - scatter plot with correlation stats
- scatter_hex       - 2d density plot with hex binning

"""
import os
import imageio
import numpy as np
import scipy.stats as sps

from skimage.transform import resize

from loguru import logger

from helpers import stats, utils, fsutil

__INTERACTIVE = False


def configure(profile=None):

    if profile == 'tex':
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        rc('axes', titlesize=14)
        rc('axes', labelsize=14)
        rc('xtick', labelsize=8)
        rc('ytick', labelsize=8)
        rc('legend', fontsize=10)
        rc('figure', titlesize=14)
    if profile == "big":
        from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=False)
        rc('axes', titlesize=14)
        rc('axes', labelsize=14)
        rc('xtick', labelsize=12)
        rc('ytick', labelsize=12)
        rc('legend', fontsize=12)
        rc('figure', titlesize=14)
    else:
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)
    # import seaborn as sns
    # sns.set('paper', font_scale=2, style="darkgrid")
    # sns.set_context("paper")


def get_figure(**kwargs):
    if __INTERACTIVE:
        import matplotlib.pyplot as plt
        return plt.figure(**kwargs)
    else:
        from matplotlib.figure import Figure
        return Figure(**kwargs)

def set_interactive(status=False):
    global __INTERACTIVE
    __INTERACTIVE = status


def thumbnails(images, ncols=0, columnwise=False):
    """
    Return a numpy array with image thumbnails.
    """
    
    if type(images) is np.ndarray:
        
        n_images = images.shape[0]        
        n_channels = images.shape[-1]
        img_size = images.shape[1:]
        
        if len(img_size) == 2:
            img_size += (1,)
                        
    elif type(images) is list or type(images) is tuple:
        
        n_images = len(images)
        n_channels = images[0].shape[-1]
        img_size = list(images[0].shape)
        
        if len(img_size) == 2:
            img_size += (1,)
    
    ncols = ncols if ncols > 0 else n_images
    images_x = ncols or int(np.ceil(np.sqrt(n_images)))
    images_y = int(np.ceil(n_images / images_x))
    size = (images_y, images_x)
        
    # Allocate space for the thumbnails
    output = np.zeros((size[0] * img_size[0], size[1] * img_size[1], img_size[2]))
        
    for r in range(n_images):
        bx = int(r % images_x)
        by = int(np.floor(r / images_x))
        if columnwise:
            by = int(r % images_y)
            bx = int(np.floor(r / images_y))
        current = images[r].squeeze()
        if current.shape[0] != img_size[0] or current.shape[1] != img_size[1]:
            current = resize(current, img_size[:-1], anti_aliasing=True)
        if len(current.shape) == 2:
            current = np.expand_dims(current, axis=2)
        output[by*img_size[0]:(by+1)*img_size[0], bx*img_size[1]:(bx+1)*img_size[1], :] = current
        
    return output
    

def _imarray(img, n_images, fetch_hook, titles, figwidth=4, cmap='gray', ncols=0, fig=None, rowlabels=None):
    """
    Function for plotting arrays of images. Not intended to be used directly. See 'images' for typical use cases.
    """
    if n_images > 128:
        raise RuntimeError('The number of subplots exceeds reasonable limits ({})!'.format(n_images))                            
    
    if ncols == 0:
        ncols = int(np.ceil(np.sqrt(n_images)))
    elif ncols < 0:
        ncols = n_images // abs(ncols)

    subplot_x = ncols
    subplot_y = int(np.ceil(n_images / subplot_x))

    if rowlabels is not None and len(rowlabels) != subplot_y:
        raise ValueError('The number of rows does not match the provided labels!')
            
    if titles is not None and type(titles) is str:
        titles = [titles for x in range(n_images)]
        
    if titles is not None and len(titles) != n_images:
        raise ValueError('Provided titles ({}) do not match the number of images ({})!'.format(len(titles), n_images))

    fig = fig or get_figure(figsize=(figwidth * subplot_x, figwidth * subplot_y))

    for n in range(n_images):
        ax = fig.add_subplot(subplot_y, subplot_x, n + 1)
        image(fetch_hook(img, n), titles[n] if titles is not None else None, axes=ax, cmap=cmap)
        if rowlabels is not None and n % subplot_x == 0:
            ax.set_ylabel(rowlabels[n // subplot_x])

    return fig
    

def images(imgs, titles=None, figwidth=4, cmap='gray', ncols=0, fig=None, rowlabels=None):
    """
    Plot a series of images (in various structures). Not thoroughly tested, but should work with:

    - np.ndarray of size (h,w,3) or (h,w)
    - lists or tuples of np.ndarray of size (h,w,3) or (h,w)    
    - np.ndarray of size (h,w,channels) -> channels shown separately
    - np.ndarray of size (1, h, w, channels)
    - np.ndarray of size (N, h, w, 3) and (N, h, w, 1)
    
    CAUTION: This function creates new figures without a canvas manager - this prevents memory leaks but makes it more
    difficult to plot figures in interactive notebooks. In case of trouble, you can supply a target figure created by
    a managed matplotlib interface - use fig=plt.figure().

    :param imgs: input image structure (see details above)
    :param titles: a single string or a list of strings matching the number of images in the structure
    :param figwidth: width of a single image in the figure
    :param cmap: color map
    :param ncols: number of columns or: 0 for sqrt(#images) cols; use negative to set the number of rows
    :param fig: specify the target figure for plotting
    """
        
    if type(imgs) is list or type(imgs) is tuple:
        
        n_images = len(imgs)
        
        def fetch_example(image, n):
            return image[n]        
                    
        return _imarray(imgs, n_images, fetch_example, titles, figwidth, cmap, ncols, fig, rowlabels)
            
    elif type(imgs) in [np.ndarray, imageio.core.util.Image]:
        
        if imgs.ndim == 2 or (imgs.ndim == 3 and imgs.shape[-1] == 3):
            
            fig = fig or get_figure(tight_layout=True, figsize=(figwidth, figwidth))
            image(imgs, titles, axes=fig.gca(), cmap=cmap)
            
            return fig

        elif imgs.ndim == 3 and imgs.shape[-1] != 3:
                        
            def fetch_example(im, n):
                return im[..., n]
            
            n_images = imgs.shape[-1]

            if n_images > 100:
                imgs = np.moveaxis(imgs, 0, -1)
                n_images = imgs.shape[-1]
                                        
        elif imgs.ndim == 4 and (imgs.shape[-1] == 3 or imgs.shape[-1] == 1):
            
            n_images = imgs.shape[0]
            
            def fetch_example(im, n):
                return im[n]
            
        elif imgs.ndim == 4 and imgs.shape[0] == 1:

            n_images = imgs.shape[-1]
            
            def fetch_example(im, n):
                return im[..., n]

        else:
            raise ValueError('Unsupported array dimensions {}!'.format(imgs.shape))
            
        return _imarray(imgs, n_images, fetch_example, titles, figwidth, cmap, ncols, fig, rowlabels)
            
    else:
        raise ValueError('Unsupported array type {}!'.format(type(imgs)))
                
    return fig


def image(x, label=None, *, axes=None, cmap='gray', vrange=None):
    """
    Plot a single image, hide ticks & add a formatted title with patterns replaced as follows:
    - '()' -> '(height x width)'
    - '[]' -> '[min - max]'
    - '{}' -> '(height x width) / [min - max]'
    - '<>' -> 'avg ± std'
    """
    
    label = label if label is not None else '{}'
    
    x = x.squeeze()
    
    if any(ptn in label for ptn in ['{}', '()', '[]', '<>']):
        label = label.replace('{}', '() / []')
        label = label.replace('()', '({}x{})'.format(*x.shape[0:2]))
        label = label.replace('[]', '[{:.2f} - {:.2f}]'.format(np.min(x), np.max(x)))
        label = label.replace('<>', '{:.2f} ± {:.2f}'.format(np.mean(x), np.std(x)))
        
    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    if vrange is None:
        axes.imshow(x, cmap=cmap)
    elif isinstance(vrange, tuple) and len(vrange) == 2:
        axes.imshow(x, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    else:
        axes.imshow(x, cmap=cmap, vmin=0, vmax=1)

    if len(label) > 0:
        axes.set_title(label)
    axes.set_xticks([])
    axes.set_yticks([])

    if 'fig' in locals():
        return fig


def sub(n_plots, figwidth=6, figheight=None, ncols=-1, fig=None, transpose=False):
    """
    Create a figure and split it into subplots. Provides more consistent behavior than matplotlib. Key features:
    - the returned axes are always a list
    - automatically choose number of rows/columns based on the total number of plots
    - the extra subplots will be turned off
    - axes traversal order can be changed (column/row-wise)

    :param n_plots:
    :param figwidth:
    :param figheight:
    :param ncols:
    :param fig:
    :param transpose:
    :return:
    """

    if ncols == 0:
        ncols = int(np.ceil(np.sqrt(n_plots)))
    elif ncols < 0:
        ncols = n_plots // abs(ncols)

    figheight = figheight or figwidth

    subplot_x = ncols or int(np.ceil(np.sqrt(n_plots)))
    subplot_y = int(np.ceil(n_plots / subplot_x))

    if transpose:
        subplot_x, subplot_y = subplot_y, subplot_x

    fig = fig or get_figure(tight_layout=True, figsize=(figwidth * subplot_x, subplot_y * (figheight or figwidth * (subplot_y / subplot_x))))
    axes = fig.subplots(nrows=subplot_y, ncols=subplot_x)
    axes_flat = []

    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax in axes:
        
        if hasattr(ax, '__iter__'):
            for a in ax:
                if len(axes_flat) < n_plots:
                    axes_flat.append(a)
                else:
                    a.remove()
        else:
            if len(axes_flat) < n_plots:
                axes_flat.append(ax)
            else:
                ax.remove()

    if transpose:
        from itertools import product
        axes_flat = [axes_flat[j * subplot_x + i] for i, j in product(range(subplot_x), range(subplot_y))]
    
    return fig, axes_flat


def progress(k, v, results=('training', 'validation'), log='auto', axes=None, start=0, alpha=0.8, color=None, title=None):
    active = False
    markers = '.os^'[:len(results)]

    for ri, r in enumerate(results):
        if r not in v or len(v[r]) == 0:
            continue
        n_hist = len(v[r]) // 2
        active = True
        xr = start + np.linspace(0, 100, len(v[r]))
        if title is not None:
            axes.set_title(title)
        ma = stats.ma_exp(v[r], alpha)
        opacity = 0.1 + 0.9 * np.exp((1 - len(xr))/100)
        axes.plot(xr, v[r], color or f'C{ri}{markers[ri]}', alpha=opacity)
        axes.plot(xr, ma, color or f'C{ri}-', label=f'{k}:{r} ({utils.format_number(ma[-1])})')
        if (log == 'auto' and np.std(v[r][-n_hist:])/(max(v[r]) - min(v[r])) < 0.02) or (isinstance(log, bool) and log):
            axes.set_yscale('log')
        axes.set_xlabel(f'training progress [% of {len(xr)} steps]')

    if active:
        axes.legend()


def perf(training_progress, results=None, metrics=None, figwidth=5, log='auto', fig=None, alpha=0.25):
    """
    Plots training performance stats organized into a dictionary with the following structure:
     - {metric}/{training,validation} -> [values]
     - {metric} -> [values]

    :param training_progress: dictionary with training progress
    :param results: tuple or string, specifies which results to show, e.g., ('training', 'validation') or 'training'
    :param figwidth: width of a single subplot
    :param log: whether to use log scale
    :param fig: handle to matplotlib figure
    :param alpha: parameter for the exponential moving average
    :return: figure handle
    """

    if isinstance(results, str):
        results = (results, )

    # If the data is not formatted as {metric: {training: [values], validation: [values]}} but rather {metric: [values]}
    # convert to the expected structure

    if any(not isinstance(v, dict) for v in training_progress.values()):
        auto = {k: {'auto': v} for k, v in training_progress.items() if utils.is_vector(v)}
        training_progress = {k: v for k, v in training_progress.items() if isinstance(v, dict)}
        training_progress.update(auto)

    # Find the number of metrics with available data
    active = []

    for i, (k, v) in enumerate(training_progress.items()):
        # Check if all training metrics have all requested sets of results
        if results is None or all(r in v and len(v[r]) > 0 for r in results):
            active.append(k)

    if len(active) == 0:
        raise ValueError('No valid plots! Missing training/validation data? Use results=["training"] to select.')

    fig, axes = sub(len(active), ncols=-1, fig=fig)
    fig.set_size_inches((len(active) * figwidth, figwidth * 0.75))

    if metrics is not None:
        active = [x for x in active if x in metrics]

    for i, k in enumerate(active):
        v = training_progress[k]
        progress(k, v, results or v.keys(), log, axes[i], alpha=alpha)
    
    return fig


def boxplots(samples, labels, axes=None, guides=None, xlabel=None):
    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    bp = axes.boxplot([x.ravel() for x in samples], vert=False, notch='True', patch_artist=True)

    for ci, o in enumerate(bp['boxes']):
        o.set(facecolor=f'C{ci}', alpha=0.5)

    for flier in bp['fliers']:
        flier.set(marker='.', color=f'C{ci}', alpha=0.5)

    for ti, t in enumerate(guides):
        axes.plot([t, t], [0.5, len(samples)+0.5], 'k:' if ti == 0 else 'k--')

    if xlabel is not None:
        axes.set_xlabel(xlabel)

    axes.set_yticklabels(labels)


def hist(samples, bins, labels, xlabel=None, guides=0, axes=None, alpha=0.4, scale=False, colors=None, guide=None, kde=False):
    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    if isinstance(samples, np.ndarray) and utils.is_vector(samples):
        samples = [samples]
        labels = [labels]

    s_min = np.min([np.min(s) for s in samples])
    s_max = np.max([np.max(s) for s in samples])
    
    if s_min == s_max:
        delta = 10 ** (np.log10(s_max) - 1)
        s_min -= delta
        s_max += delta
        
    cc = np.linspace(s_min, s_max, bins)
    h_bins = stats.bin_edges(cc)

    h_max_global = 0

    for si, spls in enumerate(samples):

        p50 = np.percentile(spls, 50)
        p99 = np.percentile(spls, 99)
        p01 = np.percentile(spls, 1)

        if labels is not None:
            label = labels[si]
            label = label.replace('()', f'{p50:.2f}')
        else:
            label = None

        color = colors[si] if colors is not None else None

        h1 = axes.hist(spls.ravel(), h_bins, color=color, alpha=alpha, density=True, label=label)

        h_max = 1.05 * np.max(h1[0])
        if h_max > h_max_global:
            h_max_global = h_max
        color = h1[-1][0].get_facecolor()

        if kde:
            try:
                e_color = (color[0] * 0.75, color[1] * 0.75, color[2] * 0.75, 0.7)
                d_bins = np.linspace(h_bins[0], h_bins[-1], 200)
                kde_pos = sps.gaussian_kde(spls.ravel())
                axes.plot(d_bins, kde_pos.pdf(d_bins), color=e_color, linewidth=2)
            except np.linalg.LinAlgError as e:
                logger.warning(f'Plotting error (KDE): {e}')

        if guides & 1:
            axes.plot([p01, p01], [0, h_max], ':', color=color)

        if guides & 2:
            axes.plot([p50, p50], [0, h_max], ':', color=color)

        if guides & 4:
            axes.plot([p99, p99], [0, h_max], ':', color=color)

    axes.set_yticks([])

    if labels is not None:
        axes.legend()

    if scale:
        margin = (cc[-1] - cc[0]) / 100
        axes.set_xlim([cc[0] - margin, cc[-1] + margin])
        axes.set_ylim([0, h_max_global * 1.01])

    if guide is not None:
        axes.plot([guide * 1.01, guide * 1.01], [0, h_max_global], 'k:', alpha=0.5)

    if xlabel is not None:
        axes.set_xlabel(xlabel)

    if 'fig' in locals():
        return fig


def detection(positive, negative, bins=100, axes=None, title='()', scale=True, reference=None, guides=2, kde=True):
    """
    Plot histograms of positive & negative detection scores.

    :param positive: positive detection scores (numpy array)
    :param negative: positive detection scores (numpy array)
    :param bins: number of histogram bins
    :param axes: matplotlib axes' handle
    :param title: plot title, '()' will be replaced with accuracy and tpr stats
    :param scale: boolean flag to auto select x limits
    :param reference: additional scores to be plotted as a reference (shown in gray)
    :param guides: draw lines as guides: 0 (no lines), 1 (best accuracy threshold), 2 (threshold + percentiles)
    :return: figure handle (if created here)
    """

    cc_min = np.min([positive.min(), negative.min()])
    cc_max = np.max([positive.max(), negative.max()])

    if reference is not None:
        cc_max = np.max([cc_max, reference.max()])
        cc_min = np.min([cc_min, reference.min()])

    cc = np.linspace(cc_min, cc_max, bins)
    bin_accuracy, thr_id = stats.detection_accuracy(positive, negative, cc, return_index=True)
    tpr = stats.true_positive_rate(positive, negative)

    v_no_match_min = np.percentile(negative, 99)
    v_do_match_max = np.percentile(positive, 1)

    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    # From bin centers, convert to bin edges for histogram computation
    h_bins = stats.bin_edges(cc)
    
    h1 = axes.hist(positive.ravel(), h_bins, color='g', alpha=0.4, density=True, label='positive')
    h2 = axes.hist(negative.ravel(), h_bins, color='r', alpha=0.4, density=True, label='negative')
    if reference is not None:
        h3 = axes.hist(reference.ravel(), h_bins, color='tab:blue', alpha=0.4, density=True, label='reference')

    if kde:
        dd = np.linspace(cc_min, cc_max, max(100, 10 * bins))
        d_bins = stats.bin_edges(dd)

        kde_pos = sps.gaussian_kde(positive.ravel())
        axes.plot(d_bins, kde_pos.pdf(d_bins), color='g')

        kde_neg = sps.gaussian_kde(negative.ravel())
        axes.plot(d_bins, kde_neg.pdf(d_bins), color='r')

        if reference is not None:
            try:
                kde_ref = sps.gaussian_kde(reference.ravel())
                axes.plot(d_bins, kde_ref.pdf(d_bins), color='tab:blue')
            except np.linalg.LinAlgError as e:
                logger.warning(f'Plotting error (KDE): {e}')
                
    h_max = max(np.max(h1[0]), np.max(h2[0]))
    h_max = min(h_max, 5 * np.max(h1[0][1:]))
    h_max = 1.05 * h_max

    if utils.is_number(guides):
        if guides == 2:
            axes.plot([v_do_match_max, v_do_match_max], [0, h_max], 'g--')
            axes.plot([v_no_match_min, v_no_match_min], [0, h_max], 'r--')
            axes.plot([cc[thr_id], cc[thr_id]], [0, max(h1[0][thr_id], 0.05 * h_max)], 'k:')
        elif guides == 1:
            axes.plot([cc[thr_id], cc[thr_id]], [0, h_max], 'k:')
    else:
        for g in guides:
            axes.plot([g, g], [0, h_max], 'k:')

    axes.set_title(title.replace('()', f'acc. {bin_accuracy:.2f}, tpr @ 1\\%far={tpr:.2f}'))
    
    if scale:
        axes.set_xlim([1.1 * cc_min if cc_min < 0 else 0.9 * cc_min, 1.1 * cc_max])
    
    axes.set_ylim([0, h_max])
    axes.legend()

    if 'fig' in locals():
        return fig


def roc(matching, non_matching, bins=1000, axes=None, label=None, plot_guides=True):

    tpr, fpr = stats.roc(matching, non_matching, bins)
    tpr_at_1pp_fpr = stats.true_positive_rate(matching, non_matching, 0.01)
    auc = stats.auc(matching, non_matching)

    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    label = f'{label} : tpr={tpr_at_1pp_fpr:.2f} auc={auc:.2f}' if label is not None else None
    axes.plot(fpr, tpr, '-', label=label)
    if plot_guides:
        axes.plot([0, 1], [0, 1], 'k--', alpha=0.2)
        axes.plot([0.01, 0.01], [0, tpr_at_1pp_fpr], 'k:', alpha=0.25)
        axes.plot([0.01, 1], [tpr_at_1pp_fpr, tpr_at_1pp_fpr], 'k:', alpha=0.25)
    axes.set_xlim([-0.02, 1.02])
    axes.set_ylim([-0.02, 1.02])
    axes.set_xlabel('false positive rate')
    axes.set_ylabel('true positive rate')
    if label is not None: axes.legend()

    if 'fig' in locals():
        return fig


def intervals_bulk(x, y, p=10):
    # fig, axes = plt.subplots(nrows=1, ncols=len(y), sharex=True)
    fig, axes = sub(len(y), ncols=-1)
    fig.set_size_inches((6 * len(y), 3))
    
    xl = sorted(x.keys())[0]
    xv = x[xl]
    
    for i, (k, v) in enumerate(y.items()):
        axes[i].plot(xv, np.percentile(v, 50, axis=0))
        axes[i].fill_between(xv, np.percentile(v, p, axis=0), np.percentile(v, 100-p, axis=0), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',)
        axes[i].set_ylabel(k)
        axes[i].set_xlabel(xl)

    return fig


def intervals(x, y, p=10, xlabel=None, ylabel=None, style='.-', axes=None, label=None):
    h = axes.plot(x, np.percentile(y, 50, axis=0), style, label=label)
    color = h[0].get_color()
    axes.fill_between(x, np.percentile(y, p, axis=0), np.percentile(y, 100-p, axis=0), alpha=0.2,
                      edgecolor=color, facecolor=color)
    if ylabel is not None: axes.set_ylabel(ylabel)
    if xlabel is not None: axes.set_xlabel(xlabel)


def correlation(x, y, xlabel=None, ylabel=None, title=None, axes=None, alpha=0.1, guide=False, color=None):

    title = '{} : '.format(title) if title is not None else ''

    cc = stats.corrcoeff(x.ravel(), y.ravel())
    r2 = stats.rsquared(x.ravel(), y.ravel())

    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    axes.plot(x.ravel(), y.ravel(), '.', alpha=alpha, color=color)
    axes.set_title('{}corr {:.2f} / R2 {:.2f}'.format(title, cc, r2))

    if guide:
        p1 = min(np.min(x), np.min(y))
        p2 = max(np.max(x), np.max(y))
        axes.plot([p1, p2], [p1, p2], 'k--', alpha=0.3)
        span_x = np.max(x) - np.min(x)
        span_y = np.max(y) - np.min(y)
        axes.set_xlim([np.min(x) - span_x * 0.05, np.max(x) + span_x * 0.05])
        axes.set_ylim([np.min(y) - span_y * 0.05, np.max(y) + span_y * 0.05])

    if xlabel is not None: axes.set_xlabel(xlabel)
    if ylabel is not None: axes.set_ylabel(ylabel)

    if 'fig' in locals():
        return fig    


def scatter_hex(x, y, xlabel=None, ylabel=None, axes=None, marginals=True, bins=30):
    axes.hexbin(x, y, gridsize=50, bins=bins, cmap='Blues')
    axes.set_xticks([])
    axes.set_yticks([])

    if xlabel is not None: axes.set_xlabel(xlabel)
    if ylabel is not None: axes.set_ylabel(ylabel)

    if marginals:
        # X marginal
        x_hist, x_bins = np.histogram(x.reshape((-1, )), bins=bins)
        x_bins = np.convolve(x_bins, [0.5, 0.5], mode='valid')
        x_hist = x_hist / x_hist.max()
        yy = axes.get_ylim()
        axes.bar(x_bins, bottom=yy[1], height=0.1 * np.abs(yy[1] - yy[0]) * x_hist, zorder=-1, clip_on=False, alpha=0.5, width=x_bins[1] - x_bins[0])
        axes.set_ylim(yy)

        # Y marginal
        y_hist, y_bins = np.histogram(y.reshape((-1, )), bins=bins)
        y_bins = np.convolve(y_bins, [0.5, 0.5], mode='valid')
        y_hist = y_hist / y_hist.max()
        xx = axes.get_xlim()
        axes.barh(y_bins, left=xx[1], width=0.1 * np.abs(xx[1] - xx[0]) * y_hist, zorder=-1, clip_on=False, alpha=0.5, height=y_bins[1] - y_bins[0])
        axes.set_xlim(xx)


def to_tikz(fig, filename=None):
    import tikzplotlib
    from helpers import results_data as rd

    tikzplotlib.clean_figure(fig=fig)

    if filename is not None:
            
        if filename.endswith('.tex'):
            tikzplotlib.save(filename, figure=fig)

        elif filename.endswith('.pdf'):
            tex = tikzplotlib.get_tikz_code(figure=fig)
            rd.render_tex(tex, format='file', filename=filename)

        elif filename.endswith('.*'):
            tikzplotlib.save(filename.replace('.*', '.tex'), figure=fig)
            tex = tikzplotlib.get_tikz_code(figure=fig)
            rd.render_tex(tex, format='file', filename=filename.replace('.*', '.pdf'))

        elif filename == '-':
            return tikzplotlib.get_tikz_code(figure=fig)
        
        else:
            raise ValueError('File format not recognized!')

    else:
        tex = tikzplotlib.get_tikz_code(figure=fig)
        return rd.render_tex(tex)
        

