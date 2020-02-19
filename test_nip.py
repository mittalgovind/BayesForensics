#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import logging
import argparse
import numpy as np

from helpers import coreutils, dataset, metrics, plotting

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

supported_pipelines = ['UNet', 'DNet', 'INet']


def develop_image(camera, pipeline, ps=128, n_images=5, patches=2, root_dir='./data'):
    """
    Display a patch developed by a neural imaging pipeline.
    """

    supported_cameras = coreutils.listdir(os.path.join(root_dir,  'models', 'nip'), '.*')

    if ps < 4 or ps > 2048:
        raise ValueError('Patch size seems to be invalid!')

    if pipeline not in supported_pipelines:
        raise ValueError('Unsupported pipeline model ({})! Available models: {}'.format(pipeline, ', '.join(supported_pipelines)))

    if camera not in supported_cameras:
        raise ValueError('Camera data not found ({})! Available cameras: {}'.format(camera, ', '.join(supported_cameras)))

    # Lazy imports to minimize delay for invalid command line parameters
    import numpy as np
    import imageio as io
    import matplotlib.pylab as plt
    import tensorflow as tf
    from models import pipelines

    root_dirname = os.path.join(root_dir, 'models', 'nip')
    data_dirname = os.path.join(root_dir, 'raw', 'training_data', camera)

    # Construct the NIP model
    model = getattr(pipelines, pipeline)()
    log.info('Using NIP: {}'.format(model.summary()))
    log.info('Loading weights from: {}'.format(os.path.join(root_dirname, camera)))
    model.load_model(os.path.join(root_dirname, camera))

    # Load sample data
    data = dataset.IPDataset(data_dirname, n_images=0, v_images=n_images, val_rgb_patch_size=ps, val_n_patches=patches)
    sample_x, sample_y = data.next_validation_batch(0, data.count_validation)
    sample_Y = model.process(sample_x).numpy()

    psnrs = metrics.psnr(sample_y, sample_Y)

    # Plot images
    fig, axes = plotting.sub(2, ncols=1)
    plotting.quickshow(plotting.thumbnails(sample_Y, n_images, True), '{}, average PSNR={:.1f} dB'.format(type(model).__name__, float(psnrs.mean())), axes=axes[0])
    plotting.quickshow(plotting.thumbnails(sample_y, n_images, True), 'Target RGB images', axes=axes[1])

    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--nip', dest='nip', action='store', help='imaging pipeline ({})'.format(supported_pipelines))
    parser.add_argument('--images', dest='images', action='store', default=8, type=int,
                        help='number of test images')
    parser.add_argument('--patches', dest='patches', action='store', default=3, type=int,
                        help='number of patches per image')
    parser.add_argument('--patch', dest='patch', action='store', default=128, type=int,
                        help='patch size')
    parser.add_argument('--dir', dest='dir', action='store', default='./data',
                        help='root directory with images and training data')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    develop_image(args.camera, args.nip, args.patch, args.images, args.patches, args.dir)


if __name__ == "__main__":
    main()
