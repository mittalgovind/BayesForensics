#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries
import os
import numpy as np

# External libraries
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from hyperopt import hp
from loguru import logger


def main(args):
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Initializing ray")
    ray.init(configure_logging=False, num_cpus=args.cpus, num_gpus=args.gpus)

    logger.info("Initializing ray search space")
    search_space, initial_best_config = create_search_space()

    logger.info("Initializing scheduler and search algorithms")
    # Use HyperBand scheduler to earlystop unpromising runs
    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration',
                                        metric="val_loss",
                                        mode="min")

    # Use bayesian optimisation provided by hyperopt
    search_alg = HyperOptSearch(space=search_space,
                                metric="val_loss",
                                mode="min",
                                points_to_evaluate=[initial_best_config])

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)

    logger.info("Initializing ray Trainable")
    data_train = np.load(
        os.path.join(args.root, 'native12k_qM.npy'))[
                 :25 * args.n_images]
    data_val = np.load(
        os.path.join(args.root, 'native12k_20k_val.npy'))[
               :20 * args.v_images]

    if args.days > 0:
        time_budget_s = int(args.days * 24 * 3600 - 30 * 60)
    else:
        time_budget_s = None

    trainer = Trainable(args.root, args.bs, args.lr, args.save_dir,
                        args.epochs, args.n_images, args.v_images,
                        args.memory_growth, args.verbose)

    logger.info("Starting hyperparameter tuning")
    analysis = tune.run(
        tune.with_parameters(trainer.train, data_train=data_train,
                             data_val=data_val),
        verbose=1,
        num_samples=args.num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        raise_on_failed_trial=False,
        resources_per_trial={"cpu": args.cpus, "gpu": args.gpus},
        resume=args.resume,
        local_dir=args.save_dir,
        log_to_file=True,
        time_budget_s=time_budget_s,
    )

    best_config = analysis.get_best_config(metric="val_loss", mode='min')
    logger.info(f'Best config: {best_config}')

    if best_config is None:
        logger.error(f'Optimization failed')
    else:
        logger.info("Saving best model config")
        with open(os.path.join(args.save_dir, 'config_on_two_gpus.json'),
                  'w') as f:
            import json
            json.dump(best_config, f, indent=4)

    logger.info("Training completed")
