#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University
# By: Govind (mittal@nyu.edu)

# Standard libraries
from abc import ABC, abstractmethod

# External libraries
import numpy as np
from tqdm.autonotebook import tqdm


class BayesBaseTrainer(ABC):
    """Base class for Bayesian training and inference pipeline.
    This pipeline is developed for Bayesian forensics on image data."""

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        output_path,
        batch_size=1,
        patch_size=128,
        print_interval=100,
        training=False,
        **kwargs,
    ):

        # required
        super().__init__(**kwargs)
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.output_path = output_path
        self.training = training

        # could be default
        self.batch_size = batch_size
        self.patch_size = patch_size

        # defaults or derived
        self.num_train_batches = dataloader.count_training // batch_size
        self.num_val_batches = dataloader.count_validation // batch_size
        self.training_loss_history = []
        self.validation_loss_history = []
        self.batch_counter = 0
        self.epoch_counter = 0

        self.print_interval = print_interval

    @abstractmethod
    def training_step(self, batch):
        """Compute loss for a training batch.

        Arguments
        ---------
        batch : dict
            batch from training dataloader

        Returns
        -------
        loss_value : tf.Tensor
            loss for input batch
        """

        raise NotImplemented

    @property
    def training_batch_iterator(self):
        """Construct progress bar validation batch iterator."""
        batch_iterator = tqdm(
            self.num_train_batches,
            total=len(self.num_train_batches),
            desc="Training",
            smoothing=0,
            leave=False,
        )

        return batch_iterator

    def training_print(self):
        """Print message for training batch."""
        if self.batch_counter % self.print_interval == 0:
            recent_history = self.training_loss_history[-100:]
            print(f"{self.batch_counter} training: {np.mean(recent_history):.4f}")

        return None

    def training_pass(self):
        """Runs through a complete epoch"""
        self.training = True
        for batch_idx in self.training_batch_iterator:
            batch = self.dataloader.next_training_batch(
                batch_idx, self.batch_size, self.patch_size
            )
            loss = self.training_step(batch)
            self.training_loss_history.append(loss)
            self.training_print()
            self.batch_counter += 1

        return None

    @abstractmethod
    def validation_step(self, batch):
        """Compute loss for a training batch.

        Arguments
        ---------
        batch : dict
            batch from validation dataloader

        Returns
        -------
        loss_value : torch.Tensor
            loss for input batch
        """

        raise NotImplementedError

    def validation_print(self):
        """Print message for validation."""
        if self.batch_counter % self.print_interval == 0:
            most_recent = self.validation_loss_history[-1]
            print(f"{self.batch_counter} validation: {most_recent:.4f}")

        return None

    @property
    def validation_batch_iterator(self):
        """Construct progress bar validation batch iterator."""
        batch_iterator = tqdm(
            self.num_val_batches,
            total=len(self.num_val_batches),
            desc="Validating",
            smoothing=0,
            leave=False,
        )

        return batch_iterator

    def validation_pass(self):
        """Run complete pass over validation data."""
        loss_history = []
        self.training = False
        for batch_idx in self.num_val_batches:
            batch = self.dataloader.next_validation_batch(
                batch_idx, self.batch_size, self.patch_size
            )
            loss_value = self.validation_step(batch)
            loss_history.append(loss_value.item())

        mean_loss = np.mean(loss_history)
        self.validation_loss_history.append(mean_loss)
        self.validation_print()

        # TODO (Govind) - Add LR handling and early stopping callback.

        return None

    def run(self, num_epochs):
        epoch_iterator = tqdm(
            range(num_epochs), total=num_epochs, desc="Epochs", smoothing=0, leave=True
        )

        for _ in epoch_iterator:
            self.training_pass()
            self.validation_pass()

            self.epoch_counter += 1

        # TODO (Govind) - Add early stopping and other callbacks?

        return None
