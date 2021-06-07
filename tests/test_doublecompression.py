import pytest

import tensorflow as tf
import numpy as np

from helpers.tf_jpeg import TFJPEG
from workflows.jpeg_double_compression import (
    JPEGDoubleCompression,
    DoubleCompressionDataset,
    load_parameters
)


QF1 = 50
QF2 = 60
PATCH_SIZE = 64
BATCH_SIZE = 10
NUM_MODELS = 2


@pytest.fixture
def dataset():
    return DoubleCompressionDataset(
        data_dir="native12k",
        load="y",
        n_images=20,
        v_images=20,
        seed=2468,
        codec=TFJPEG(codec="soft"),
        qf_train="50,70",
        qf_test="50,70",
        val_rgb_patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE
    )


@pytest.fixture
def model_mcd():
    return JPEGDoubleCompression(
        **(load_parameters()),
        uncertainty_method="mc-dropout"
    )


@pytest.fixture
def model_ens():
    return JPEGDoubleCompression(
        **(load_parameters()),
        uncertainty_method="ensemble",
        num_models=NUM_MODELS
    )


def test_dataset(dataset):
    for images, labels in dataset.get_validation_generator(QF1=QF1, QF2=QF2):
        assert tf.is_tensor(images)
        assert tf.is_tensor(labels)
        
        assert images.shape[0] == BATCH_SIZE * 2
        assert labels.shape[0] == BATCH_SIZE * 2
        
        assert images.shape[1] == PATCH_SIZE
        assert images.shape[2] == PATCH_SIZE
        
        for i in range(BATCH_SIZE):
            assert not tf.reduce_all(tf.equal(images[i], images[i + BATCH_SIZE]))

            
def test_mcdropout(dataset, model_mcd):
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    model = model_mcd
    model._model.compile(
        optimizer,
        loss=loss_criterion,
        metrics=["accuracy"]
    )

    for images, labels in dataset.get_validation_generator(QF1=QF1, QF2=QF2):
        preds = [model(images, training=False) for _ in range(10)]

        for i in range(len(preds) - 1):
            assert tf.is_tensor(preds[i])
            assert preds[i].shape == preds[i + 1].shape
            assert not tf.reduce_all(tf.equal(preds[i], preds[i + 1]))


def test_ensemble(dataset, model_ens):
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    model = model_ens
    model._model.compile(
        optimizer,
        loss=loss_criterion,
        metrics=["accuracy"]
    )

    for images, labels in dataset.get_validation_generator(QF1=QF1, QF2=QF2):
        preds = model(images, training=False)
        
        assert isinstance(preds, list)
        assert len(preds) == NUM_MODELS

        for i in range(NUM_MODELS - 1):
            assert tf.is_tensor(preds[i])
            assert preds[i].shape == preds[i + 1].shape
            assert not tf.reduce_all(tf.equal(preds[i], preds[i + 1]))
            