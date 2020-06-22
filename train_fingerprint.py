import sys
import json
import argparse

import numpy as np
import tensorflow as tf

from helpers.utils import factory

from helpers import dataset, tf_helpers, metrics, results_data, utils

from workflows import sensor_fingerprint as sf
from workflows.sensor_fingerprint import visualize as vis

from loguru import logger

utils.setup_logging('finger', level='DEBUG')

tf_helpers.disable_warnings()
tf_helpers.log_status()

__ACTIONS = ['train', 'retrain', 'validate', 'threats']


def action_defaults(action):
    action = utils.match_option(action, __ACTIONS)
    if action == 'train':
        return {"epochs": 1000, "batch_size": 50, "patch_size": 128, "decay": -1}
    elif action == 'validate':
        return {"n_reps": 100, "estimation_images": 0}
    elif action == 'threats':
        return {"n_reps": 50, "residual_images": 1}


def batch_training(config=None, dry_run=True, repeat=1, start_rep=0, actions=None, overwrite=False, active_configs=None):

    if overwrite:
        results_data.set_overwrite_mode('backup')
    else:
        results_data.set_overwrite_mode('warning')

    logger.info(f'Using overwrite mode="{results_data.get_overwrite_mode()}"')
    logger.info(f'Requested run_ids: {list(range(start_rep, repeat))}')
    logger.debug(f'Loading JSON: {config}')
    if active_configs is not None and len(active_configs) > 0:
        logger.warning(f'{len(active_configs)} active configurations: {active_configs}')

    with open(config) as file:
        flows = json.load(file)

    actions_override = actions is not None

    for flow_id, fc in enumerate(flows):

        # Read the current configuration & re-use parameters from the previous one
        if flow_id == 0:
            if 'data' not in fc:
                logger.error('Dataset not defined in the first config!')
                sys.exit(1)
            last_flow = dict(**fc)
        else:
            if 'data' in fc:
                logger.error('Dataset MUST be defined ONLY in the first config!')
                sys.exit(1)
            new_flow = dict(**last_flow)
            new_flow.update(fc)
            fc = dict(**new_flow)
            last_flow = dict(**fc)

        prefix = f'(Config {flow_id + 1}/{len(flows)})'

        # Actions to be performed
        if not actions_override:
            actions = set(fc['actions'].split(','))
        logger.info(f'{prefix}: actions={actions}')

        if any(action not in __ACTIONS for action in actions):
            logger.error(f'{prefix}: Some actions are not supported: {actions.difference(__ACTIONS)}')
            sys.exit(1)

        # Load dataset
        if len(actions) > 0 and not dry_run and flow_id == 0:
            data = dataset.Dataset(fc['camera'], **fc['data'])
            logger.info(f'Loaded dataset: {data.summary()}')
        else:
            logger.info(f'Configured dataset: camera={fc["camera"]} args={fc["data"]}')

        if active_configs is not None and len(active_configs) > 0:
            if flow_id not in active_configs:
                logger.info(f'{prefix} skipping configuration, as requested...')

        for run_id in range(start_rep, repeat):
            prefix = f'(Config {flow_id+1}/{len(flows)} run={run_id+1}/{repeat})'
            logger.debug(f'{prefix} {len(fc)} keys -> {list(fc.keys())}')

            # Generate a label for the current model based on current settings
            label = fc['label'].format(flow_id=flow_id, run_id=run_id, **fc)

            # Create the workflow
            if not dry_run:
                tf.keras.backend.clear_session()
                flow_settings = {**fc}
                flow_settings['label'] = label

                f = sf.SensorFingerprint.restore(flow_settings)

                # # Sanity check for the ISP
                sample_x, sample_y = data.next_validation_batch(0, 1)
                sample_Y = f.isp.process(sample_x).numpy()
                sample_ssim = np.mean(metrics.ssim(sample_y, sample_Y))

                if sample_ssim < 0.97:
                    logger.error(f'The ISP seems to work incorrectly: ssim={sample_ssim:.3f}!')
                    sys.exit(1)

                preexisting_model = all(f.model_status())
                logger.info(f'{prefix} {label}: {f.summary()}')

            if 'train' in actions or 'retrain' in actions:
                weights = factory(fc['weights'])
                t = action_defaults('train')
                t.update(fc['train'])
                logger.debug(f'(dry={dry_run}) Training arguments: {t}')

                if 'retrain' in actions:
                    f.delete_models()

                if dry_run:
                    logger.warning(f'{prefix} skipping training (dry run)')
                else:
                    sf.train_all(f, data, epochs=t['epochs'], batch_size=t['batch_size'], patch_size=t['patch_size'],
                                 restart=False, decay=t['decay'], weights=weights)
                    vis.training_progress(f, save=True)
                    vis.embedding_patterns(f, data, save=True)

            if 'validate' in actions:
                v = action_defaults('validate')
                v.update(fc['validate'])
                logger.debug(f'(dry={dry_run}) Validation arguments: {v}')

                if dry_run:
                    logger.warning(f'{prefix} skipping validation (dry run)')
                else:
                    sf.validate(f, data, batch_size=10, n_reps=v['n_reps'], estimation_images=v['estimation_images'], save=True)  # (50,90)
                    vis.validation(f, save=True)

            if 'threats' in actions:
                v = action_defaults('threats')
                v.update(fc['threats'])
                logger.debug(f'(dry={dry_run}) Threat assessment arguments: {v}')

                if dry_run:
                    logger.warning(f'{prefix} skipping security assessment (dry run)')
                else:
                    try:
                        sf.assess_security(f, data, residual_images=v['residual_images'], save=True)
                        vis.security(f, 'tm_all', save=True)
                    except Exception as e:
                        exception_type, exception_object, exception_traceback = sys.exc_info()
                        e_filename = exception_traceback.tb_frame.f_code.co_filename
                        e_line_number = exception_traceback.tb_lineno
                        logger.error(f'{e} @{e_filename}:{e_line_number}')



def main():
    parser = argparse.ArgumentParser(description='NIP & FAN optimization for manipulation detection')

    group = parser.add_argument_group('general parameters')
    group.add_argument('-c', '--config', dest='config', action='store', required=True,
                        help='JSON file with fingerprint workflow definitions')
    group.add_argument('--dry', dest='dry_run', action='store_true',
                       help='Dry run (show configurations)')
    group.add_argument('-r', '--repeat', dest='repeat', action='store', type=int, default=1,
                       help='Number of repetitions for each configuration - provides {run_id} for label formatting')
    group.add_argument('-s', '--start', dest='start_rep', action='store', type=int, default=0,
                       help='First repetition (defaults to 0)')
    group.add_argument('-a', '--actions', dest='actions', action='store',
                       help='Actions: train,validate,threats (overrides settings in the config file)')
    group.add_argument('-o', '--overwrite', dest='overwrite', action='store_true',
                       help='Force test results to be recalculated again (does not apply to training)')
    group.add_argument('-C', '--configs', dest='configs', action='store',
                       help='Force test results to be recalculated again (does not apply to training)')

    args = parser.parse_args()

    if args.actions is not None:
        args.actions = args.actions.split(',')

    if args.configs is not None:
        args.configs = [int(x) for x in args.configs.split(',')]

    batch_training(args.config, args.dry_run, args.repeat, args.start_rep, args.actions, args.overwrite, args.configs)


if __name__ == "__main__":
    main()
