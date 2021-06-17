#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import tensorflow as tf
import tensorflow_probability as tfp


# Internal libraries


class FlipoutLoss(tf.keras.losses.Loss):
    def __init__(self, n_train_images, steps_per_epoch, kl_annealing=1,
                 **kwargs):
        super().__init__(name='FlipoutLoss', **kwargs)
        self.steps_per_epoch = steps_per_epoch
        self.n_train_images = n_train_images
        self.kl_annealing = kl_annealing
        self.class_loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.step = 0

    def call(self, labels, logits):
        crossentropy = self.class_loss_criterion(labels, logits)

        labels_distribution = tfp.distributions.Categorical(logits=logits)
        kl_reg = self.step / (self.kl_annealing * self.steps_per_epoch)
        self.step += 1

        log_likelihood = labels_distribution.log_prob(labels)
        neg_log_likelihood = -tf.reduce_mean(log_likelihood)
        kl = crossentropy / self.n_train_images * tf.minimum(1.0, kl_reg)

        return neg_log_likelihood + kl


class FixedLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate
