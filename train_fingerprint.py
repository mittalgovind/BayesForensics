import sys
import json
import argparse

from models import pipelines, spn, jpeg

from helpers import dataset, plots, tf_helpers, metrics, results_data, imdiff, utils

from workflows import sensor_fingerprint as sf
from workflows.sensor_fingerprint import visualize as vis
# from workflows.sensor_fingerprint import attack

from loguru import logger

utils.setup_logging('finger', level='DEBUG')

tf_helpers.disable_warnings()
tf_helpers.log_status()

__ACTIONS = ['train', 'validate', 'threats']


def action_defaults(action):
    action = utils.match_option(action, __ACTIONS)
    if action == 'train':
        return {"epochs": 1000, "batch_size": 50, "patch_size": 128, "decay": -1}
    elif action == 'validate':
        return {"n_reps": 10, "estimation_images": 0}
    elif action == 'threats':
        return {"n_reps": 10, "residual_images": 1}


def factory(spec):
    if isinstance(spec, str):
        instance = eval(spec)
    elif isinstance(spec, dict):
        cls = eval(spec['class'])
        args = spec['args']
        instance = cls(**args)
    else:
        raise ValueError('Model definition not supported!')

    return instance


def batch_training(config=None, dry=True):

    logger.debug(f'Loading JSON: {config}')

    with open(config) as file:
        flows = json.load(file)

    for flow_id, fc in enumerate(flows):

        prefix = f'(Config {flow_id+1}/{len(flows)})'

        logger.debug(f'{prefix} {len(fc)} keys -> {list(fc.keys())}')

        # Read the current configuration & re-use parameters from the previous one
        if flow_id == 0:
            if 'data' not in fc:
                logger.error('Dataset not defined in the first config!')
                sys.exit(1)
            last_flow = dict(**fc)
            run_id = 0
        else:
            if len(fc) == 0:
                run_id += 1
            else:
                run_id = 0
            if 'data' in fc:
                logger.error('Dataset MUST be defined ONLY in the first config!')
                sys.exit(1)
            new_flow = dict(**last_flow)
            new_flow.update(fc)
            fc = dict(**new_flow)
            last_flow = dict(**fc)

        # Read parameters and create necessary objects
        label = fc['label'].format(flow_id=flow_id, run_id=run_id, **fc)

        # Actions to be performed
        actions = set(fc['actions'].split(','))
        logger.info(f'{prefix} {label}: actions={actions}')

        if any(action not in __ACTIONS for action in actions):
            logger.error(f'{prefix}: Some actions are not supported: {actions.difference(__ACTIONS)}')
            sys.exit(1)

        # Create the workflow
        if not dry:
            sensor = factory(fc['sensor'])
            detector = factory(fc['detector'])
            isp = pipelines.ClassicISP.restore(camera=fc['camera'])
            channel = factory(fc['channel'])
            channel_strength = factory(fc['channel_strength'])
            alphas = factory(fc['alphas'])
            weights = factory(fc['weights'])

            f = sf.SensorFingerprint(isp, sensor, detector, channel, alphas,
                                     channel_strength=channel_strength,
                                     learning_rate=fc['learning_rate'],
                                     fingerprint_demosaicing=fc['demosaicing'],
                                     label=label,
                                     root_dir=fc['root_dir'])

            preexisting_status = f.model_status()
            logger.info(f'{prefix} {label}: {f.summary()}')

        # Load dataset
        if len(actions) > 0 and not dry and flow_id == 0:
            data = dataset.Dataset(fc['camera'], **fc['data'])
            logger.info(f'{prefix} Loaded dataset: {data.summary()}')
        else:
            logger.info(f'{prefix} Configured dataset: camera={fc["camera"]} args={fc["data"]}')

        if 'train' in actions:
            t = action_defaults('train')
            t.update(fc['train'])
            logger.debug(f'(dry={dry}) Training arguments: {t}')
            if not dry:
                sf.train_all(f, data, epochs=t['epochs'], batch_size=t['batch_size'], patch_size=t['patch_size'],
                             restart=False, decay=t['decay'], weights=weights)

                if not all(preexisting_status.values()):
                    vis.training_progress(f, save=True)
                    vis.embedding_patterns(f, data, save=True)
                else:
                    logger.warning(f'{[prefix]}: The entire model seems to be already ready - skipping visualization')

        if 'validate' in actions:
            v = action_defaults('validate')
            v.update(fc['validate'])
            logger.debug(f'(dry={dry}) Validation arguments: {v}')
            if not dry and not all(preexisting_status.values()):
                sf.validate(f, data, batch_size=10, n_reps=v['n_reps'], estimation_images=v['estimation_images'], save=True)  # (50,90)
                vis.validation(f, save=True)
            else:
                logger.warning(f'{prefix} skipping validation (dry run or model was ready before)')

        if 'threats' in actions:
            v = action_defaults('threats')
            v.update(fc['threats'])
            logger.debug(f'(dry={dry}) Threat assessment arguments: {v}')
            if not dry and not all(preexisting_status.values()):
                sf.assess_security(f, data, residual_images=v['residual_images'], save=True)
                vis.security(f, 'tm*', save=True)
            else:
                logger.warning(f'{prefix} skipping security assessment (dry run or model was ready before)')


def main():
    parser = argparse.ArgumentParser(description='NIP & FAN optimization for manipulation detection')

    group = parser.add_argument_group('general parameters')
    group.add_argument('-f', '--config', dest='config', action='store', required=True,
                        help='JSON file with fingerprint workflow definitions')
    group.add_argument('--dry', dest='dry_run', action='store_true',
                       help='Dry run (show configurations)')
    # group.add_argument('--cam', dest='cameras', action='append',
    #                     help='add cameras for evaluation (repeat if needed)')
    # group.add_argument('--manip', dest='manipulations', action='store', default='sharpen,resample,gaussian,jpeg',
    #                    help='comma-sep. list of manipulations (:strength), e.g., : {}'.format('sharpen:1,jpeg:80,resample,gaussian'))
    # group.add_argument('--fan', dest='fan_args', default=None,
    #                     help='Set hyper-parameters for the FAN model (JSON string)')

    # Directories
    # group = parser.add_argument_group('directories')
    # group.add_argument('--dir', dest='root_dir', action='store', default='./data/m/playground/',
    #                     help='the root directory for storing results')
    # group.add_argument('--nip-dir', dest='nip_directory', action='store', default='./data/models/nip/',
    #                     help='the root directory for storing results')
    #
    # # Training parameters
    # group = parser.add_argument_group('training parameters')
    # group.add_argument('--loss', dest='loss_metric', action='store', default='L2',
    #                     help='loss metric for the NIP (L2, L1, SSIM)')
    # group.add_argument('--split', dest='split', action='store', default='120:30:4',
    #                     help='data split (#training:#validation:#validation_patches): e.g., 120:30:4')
    # group.add_argument('--ln', dest='lambdas_nip', action='append',
    #                     help='set custom regularization strength for the NIP (repeat for multiple values)')
    # group.add_argument('--lc', dest='lambdas_dcn', action='append',
    #                     help='set custom regularization strength for the DCN (repeat for multiple values)')
    # group.add_argument('--train', dest='trainables', action='append',
    #                     help='add trainable elements (nip, dcn)')
    # group.add_argument('--patch', dest='patch', action='store', default=256, type=int,
    #                     help='RGB patch size for NIP output (default 256)')
    #
    # # Training scope and progress
    # group = parser.add_argument_group('training scope')
    # group.add_argument('--scratch', dest='from_scratch', action='store_true', default=False,
    #                     help='train NIP from scratch (ignore pre-trained model)')
    # group.add_argument('--start', dest='start', action='store', default=0, type=int,
    #                     help='first iteration (default 0)')
    # group.add_argument('--end', dest='end', action='store', default=10, type=int,
    #                     help='last iteration (exclusive, default 10)')
    # group.add_argument('--epochs', dest='epochs', action='store', default=1001, type=int,
    #                     help='number of epochs (default 1001)')
    #
    # # Distribution channel
    # group = parser.add_argument_group('distribution channel')
    # group.add_argument('--jpeg', dest='jpeg_quality', action='store', default=None, type=str,
    #                     help='JPEG quality level (distribution channel)')
    # group.add_argument('--jpeg_mode', dest='jpeg_mode', action='store', default='soft',
    #                     help='JPEG approximation mode: sin, soft, harmonic')
    # group.add_argument('--dcn', dest='dcn_model', action='store', default=None,
    #                     help='DCN compression model path')
    # group.add_argument('--ds', dest='downsampling', action='store', default='pool',
    #                     help='Distribution channel sub-sampling: pool/bilinear/none')

    args = parser.parse_args()

    # Parse FAN args
    # try:
    #     args.fan_args = json.loads(args.fan_args.replace('\'', '"')) if args.fan_args is not None else {}
    # except json.decoder.JSONDecodeError:
    #     print('WARNING', 'JSON parsing error for: ', args.hyperparams_args.replace('\'', '"'))
    #     sys.exit(2)

    # Split manipulations
    # args.manipulations = args.manipulations.strip().split(',')

    batch_training(args.config, args.dry_run)


if __name__ == "__main__":
    main()
