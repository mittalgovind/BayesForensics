#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse

import numpy as np
import scipy.fftpack as sfft

from helpers import coreutils

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fft_log_norm(x, boost=10, perc=0):
    x = x.squeeze()
    if x.ndim != 3:
        raise ValueError('Only single images can be accepted as input.')
    y = np.zeros_like(x)
    for i in range(x.shape[-1]):
        y[:, :, i] = np.abs(sfft.fft2(x[:, :, i]))
        y[:, :, i] = sfft.fftshift(y[:, :, i])
        y[:, :, i] = np.log(boost + y[:, :, i])
        y[:, :, i] = nm(y[:, :, i], perc)
    return y


def nm(x, perc=0):
    if np.all(x == 0):
        return x
    mn = x.min() if perc == 0 else np.percentile(x, perc)
    mx = x.max() if perc == 0 else np.percentile(x, 100 - perc)
    return ((x - mn) / (mx - mn)).clip(0, 1)


def compare_nips(model_a_dirname, model_b_dirname, camera=None, image=None, patch_size=128, root_dirname='./data', output_dir=None, model_a_args=None, model_b_args=None):
    """
    Display a comparison of two variants of a neural imaging pipeline.
    :param camera: camera name (e.g., 'Nikon D90')
    :param model_a_dirname: directory with the first variant of the model
    :param model_b_dirname: directory with the second variant of the model
    :param ps: patch size (patch will be taken from the middle)
    :param image_id: index of the test image
    :param root_dir: root data directory
    :param output_dir: set an output directory if the figure should be saved (matplotlib2tikz will be used)
    """
    # Lazy imports to minimize delay for invalid command line parameters
    import re
    import inspect
    import imageio as io
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from models import pipelines, tfmodel
    from helpers import raw_api, loading

    supported_cameras = coreutils.listdir(os.path.join(root_dirname, 'models', 'nip'), '.*')
    supported_pipelines = pipelines.supported_models

    if patch_size < 4 or patch_size > 2048:
        raise ValueError('Patch size seems to be invalid!')

    if camera is not None and camera not in supported_cameras:
        raise ValueError('Camera data not found ({})! Available cameras: {}'.format(camera, ', '.join(supported_cameras)))

    # Find available Bayer stacks for the camera

    # Construct the NIP models
    if os.path.isdir(model_a_dirname):
        # Restore a NIP model from a training log
        model_a = tfmodel.restore(model_a_dirname, pipelines)
    else:
        # Construct the NIP model from class name (and optional arguments)
        if model_a_args is None:
            model_a = getattr(pipelines, model_a_dirname)()
        else:
            model_a = getattr(pipelines, model_a_dirname)(**model_a_args)
        model_a.load_model(os.path.join(root_dirname, model_a.model_code))

    if os.path.isdir(model_b_dirname):
        # Restore a NIP model from a training log
        model_b = tfmodel.restore(model_b_dirname, pipelines)
    else:
        # Construct the NIP model from class name (and optional arguments)
        if model_b_args is None:
            model_b = getattr(pipelines, model_b_dirname)()
        else:
            model_b = getattr(pipelines, model_b_dirname)(**model_b_args)
        model_b.load_model(os.path.join(root_dirname, model_b.model_code))

    print('ISP-A: {}'.format(model_a.summary()))
    print('ISP-B: {}'.format(model_b.summary()))

    # Load sample data

    if isinstance(image, int) and camera is not None:

        data_dirname = os.path.join(root_dirname, 'raw', 'training_data', camera)
        files = coreutils.listdir(data_dirname, '.*\.npy')
        files = files[image:image+1]
        data = loading.load_images(files, data_dirname)
        sample_x, sample_y = data['x'], data['y']

        with open('config/cameras.json') as f:
            cameras = json.load(f)
            cfa, srgb = cameras[camera]['cfa'], np.array(cameras[camera]['srgb'])

    elif image is not None:
        print('Loading a RAW image {}'.format(image))
        sample_x, cfa, srgb, _ = raw_api.unpack(image, expand=True)
        sample_y = raw_api.process(image, brightness=None, expand=True)

    if isinstance(model_a, pipelines.ClassicISP):
        print('Configuring ISP-A to CFA: {} & sRGB {}'.format(cfa, srgb.round(2).tolist()))
        model_a.set_cfa_pattern(cfa)
        model_a.set_srgb_conversion(srgb)

    if isinstance(model_b, pipelines.ClassicISP):
        print('Configuring ISP-B to CFA: {} & sRGB {}'.format(cfa, srgb.round(2).tolist()))
        model_b.set_cfa_pattern(cfa)
        model_b.set_srgb_conversion(srgb)

    # Develop images
    sample_ya = model_a.process(sample_x).numpy()
    sample_yb = model_b.process(sample_x).numpy()

    if patch_size > 0:
        print('Cropping a {p}x{p} patch from the middle'.format(p=patch_size))
        xx = (sample_x.shape[2] - patch_size // 2) // 2
        yy = (sample_x.shape[1] - patch_size // 2) // 2
        sample_x = sample_x[:, yy:yy+patch_size, xx:xx+patch_size, :]
        sample_y = sample_y[:, 2*yy:2*(yy+patch_size), 2*xx:2*(xx+patch_size), :]
        sample_ya = sample_ya[:, 2*yy:2*(yy+patch_size), 2*xx:2*(xx+patch_size), :]
        sample_yb = sample_yb[:, 2*yy:2*(yy+patch_size), 2*xx:2*(xx+patch_size), :]

    # Plot images
    fig = compare_images_ab_ref(sample_y, sample_ya, sample_yb)

    if output_dir is not None:
        from tikzplotlib import save as tikz_save
        dcomp = [x for x in coreutils.splitall(model_b_dirname) if re.match('(ln-.*|[0-9]{3})', x)]
        tikz_save('{}/examples-{}-{}-{}-{}-{}.tex'.format(output_dir, camera, pipeline, image_id, dcomp[0], dcomp[1]), figureheight='8cm', figurewidth='8cm', strict=False)
    else:
        fig.tight_layout()
        fig.show(fig)

    fig.suptitle('{}, A={}, B={}'.format(image, model_a.model_code, model_b.model_code))
    plt.show()
    plt.close(fig)


def compare_images_ab_ref(img_ref, img_a, img_b, labels=None):
    from helpers import plotting, metrics

    import matplotlib.pyplot as plt

    labels = labels or ['target', '', '']

    img_a = img_a.squeeze()
    img_b = img_b.squeeze()
    img_ref = img_ref.squeeze()

    fig, axes = plotting.sub(9, fig=plt.figure())
    fig.tight_layout()

    plotting.quickshow(img_ref, '(T) {}'.format(labels[0]), axes=axes[0])

    label_a = '(A) {}: {:.1f} dB / {:.3f}'.format(labels[1], metrics.psnr(img_ref, img_a), metrics.ssim(img_ref, img_a))
    plotting.quickshow(img_a, label_a, axes=axes[1])

    label_b = '(B) {}: {:.1f} dB / {:.3f}'.format(labels[2], metrics.psnr(img_ref, img_b), metrics.ssim(img_ref, img_b))
    plotting.quickshow(img_b, label_b, axes=axes[3])

    # A hack to allow image axes to zoom together
    axes[1].get_shared_x_axes().join(axes[0], axes[1])
    axes[3].get_shared_x_axes().join(axes[0], axes[3])
    axes[1].get_shared_y_axes().join(axes[0], axes[1])
    axes[3].get_shared_y_axes().join(axes[0], axes[3])

    # Compute and plot difference images
    diff_a = np.abs(img_a - img_ref)
    diff_a_mean = diff_a.mean()
    diff_a = nm(diff_a, 0.1)

    diff_b = np.abs(img_b - img_ref)
    diff_b_mean = diff_b.mean()
    diff_b = nm(diff_b, 0.1)

    diff_ab = np.abs(img_b - img_a)
    diff_ab_mean = diff_ab.mean()
    diff_ab = nm(diff_ab, 0.1)

    plotting.quickshow(diff_a, 'T - A: mean abs {:.3f}'.format(diff_a_mean), axes=axes[2])
    plotting.quickshow(diff_b, 'T - B: mean abs {:.3f}'.format(diff_b_mean), axes=axes[6])
    plotting.quickshow(diff_ab, 'A - B: mean abs {:.3f}'.format(diff_ab_mean), axes=axes[4])

    # Compute and plot spectra
    fft_a = fft_log_norm(diff_a)
    fft_b = fft_log_norm(diff_b)

    # fft_ab = nm(np.abs(fft_a - fft_b))
    fft_ab = nm(np.abs(fft_log_norm(img_b) - fft_log_norm(img_a)), 0.01)
    plotting.quickshow(fft_a, 'FFT(T - A)', axes=axes[5])
    plotting.quickshow(fft_b, 'FFT(T - B)', axes=axes[7])
    plotting.quickshow(fft_ab, 'FFT(A) - FFT(B)', axes=axes[8])

    return fig


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('-c', '--cam', dest='camera', action='store', help='camera')
    parser.add_argument('-i', '--image', dest='image', action='store', 
                        help='RAW image path or training image id')
    parser.add_argument('-p', '--patch', dest='patch', action='store', default=128, type=int,
                        help='patch size')
    parser.add_argument('-a', dest='model_a_dir', action='store', default='./data/models/nip',
                        help='path to first model (TF checkpoint dir)')
    parser.add_argument('-b', dest='model_b_dir', action='store', default='./data/models/nip',
                        help='path to second model (TF checkpoint dir)')
    parser.add_argument('--dir', dest='dir', action='store', default='./data/',
                        help='root directory with images and training data')
    parser.add_argument('--out', dest='out', action='store', default=None,
                        help='output directory for TikZ output (if set, the figure is not displayed)')

    args = parser.parse_args()

    compare_nips(args.model_a_dir, args.model_b_dir, args.camera, args.image,
                 args.patch, args.dir, args.out)


if __name__ == "__main__":
    main()

