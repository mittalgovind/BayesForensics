#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path

from helpers import coreutils, dataset, results_data
from training.validation import validate_fan
from compression import codec

from workflows import camera_to_browser

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Test manipulation detection (FAN) on RGB images')
    group = parser.add_argument_group('General settings')
    group.add_argument('--patch', dest='patch', action='store', default=64, type=int,
                        help='patch size')
    group.add_argument('--images', dest='images', action='store', default=-1, type=int,
                        help='number of validation images (defaults to -1 - use all in the directory)')
    group.add_argument('--patches', dest='patches', action='store', default=1, type=int,
                        help='number of validation patches')
    group.add_argument('--data', dest='data', action='store', default='./data/rgb/native12k',
                        help='directory with test RGB images')

    group = parser.add_argument_group('Training session selection')
    group.add_argument('--dir', dest='dir', action='store', default='./data/m/7-raw',
                        help='directory with training sessions')
    group.add_argument('--re', dest='re', action='store', default=None,
                        help='regular expression to filter training sessions')

    group = parser.add_argument_group('Override training settings')
    group.add_argument('--jpeg', dest='jpeg', action='store', default=None, type=int,
                        help='Override JPEG quality level (distribution channel)')
    group.add_argument('--codec', dest='jpeg_codec', action='store', default=None, type=str,
                        help='Override JPEG codec settings (libjpeg, soft, sin)')
    group.add_argument('--dcn', dest='dcn_model', action='store', default=None,
                        help='Coverride DCN model directory')
    group.add_argument('--manip', dest='manipulations', action='store', default=None,
                       help='Included manipulations, e.g., : {}'.format('sharpen,jpeg,resample,gaussian'))

    args = parser.parse_args()

    # Split manipulations
    if args.manipulations is not None:
        args.manipulations = args.manipulations.strip().split(',')

    json_files = sorted(str(f) for f in Path(args.dir).glob('**/training.json'))

    if len(json_files) == 0:
        sys.exit(0)

    # Load training / validation data
    data = dataset.IPDataset(args.data, n_images=0, v_images=args.images, load='y', val_rgb_patch_size=2 * args.patch, val_n_patches=args.patches)

    print('Data: {}'.format(data.summary()))
    print('Found {} candidate training sessions ({})'.format(len(json_files), args.dir))

    for filename in json_files:
        if args.re is None or re.findall(args.re, filename):

            with open(filename) as f:
                training_log = json.load(f)

            # Setup manipulations
            if args.manipulations is None:
                manipulations = coreutils.getkey(training_log, 'manipulations')
                if 'native' in manipulations: manipulations.remove('native')
            else:
                print('info: overriding manipulation list with {}'.format(args.manipulations))
                manipulations = args.manipulations

            # TODO This is good
            try:
                accuracy = coreutils.getkey(training_log, 'forensics/performance/accuracy/validation')[-1]
            except:
                accuracy = np.nan
            distribution = coreutils.getkey(training_log, 'distribution') 

            if args.jpeg is not None:
                print('info: overriding JPEG quality with {}'.format(args.jpeg))
                distribution['compression_params']['quality'] = args.jpeg

            if args.jpeg_codec is not None:
                print('info: overriding JPEG codec with {}'.format(args.jpeg_codec))
                distribution['compression_params']['codec'] = args.jpeg_codec

            if args.dcn_model is not None:
                print('info: overriding DCN model with {}'.format(args.dcn_model))
                distribution['compression_params']['dirname'] = args.dcn_model

            print('\n[{}]'.format(os.path.split(filename)[0]))
            flow = camera_to_browser.Camera2Browser('ONet', manipulations, distribution, {}, patch_size=args.patch)
            flow.fan.load_model(os.path.join(os.path.split(filename)[0], 'models'))
            print(flow.summary())

            _, conf = validate_fan(flow, data)
            
            print('Accuracy validated/expected: {:.4f} / {:.4f}'.format(np.mean(np.diag(conf)), accuracy))
            print(results_data.confusion_to_text((100*conf).round(0), flow._forensics_classes, filename, 'txt'))


if __name__ == "__main__":
    main()
