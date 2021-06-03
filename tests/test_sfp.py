import pytest

import tensorflow as tf
import numpy as np

from workflows.scaling_factor import (
    ScalingFactor,
    ScalingFactorDataset,
    validate,
    sf_plot
)


PATCH_SIZE = 128
SF = 0.25
NUM_MODELS = 2


@pytest.fixture
def dataset_nojpeg():
    return ScalingFactorDataset(
        data_dir="native12k",
        load="y",
        scales="0.25,1.00",
        sampling_method="nearest",
        n_classes=31,
        n_images=10,
        v_images=10,
        seed=2468,
        batch_size=10,
        patch_size=PATCH_SIZE,
    )


@pytest.fixture
def dataset_jpeg():
    return ScalingFactorDataset(
        data_dir="native12k",
        load="y",
        scales="0.25,1.00",
        sampling_method="nearest",
        n_classes=31,
        n_images=10,
        v_images=10,
        seed=2468,
        codec="libjpeg",
        jpeg_quality=60,
        batch_size=10,
        patch_size=PATCH_SIZE,
    )


@pytest.fixture
def model_mcd():
    return ScalingFactor(
        n_classes=31,
        uncertainty_method="mc-dropout",
        dense_dropout=0.5,
    )


@pytest.fixture
def model_ens():
    return ScalingFactor(
        n_classes=31,
        uncertainty_method="ensemble",
        num_models=NUM_MODELS,
    )


def test_dataset(dataset_nojpeg, dataset_jpeg):
    for data in [dataset_nojpeg, dataset_jpeg]:
        for images, labels in data.get_validation_generator(sf=SF):
            assert tf.is_tensor(images)
            assert tf.is_tensor(labels)

            assert images.shape[0] == labels.shape[0]
            assert images.shape[1] == int(SF * PATCH_SIZE)
            assert images.shape[2] == int(SF * PATCH_SIZE)

    print('Dataset tests passed.')

'''
def test_model(dataset_nojpeg, model_mcd, model_ens):
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

    for images, labels in dataset_nojpeg.get_validation_generator(sf=SF):
        preds = [model(images, training=False) for _ in range(10)]

        for i in range(len(preds) - 1):
            assert tf.is_tensor(preds[i])
            assert preds[i].shape == preds[i + 1].shape
            assert not tf.reduce_all(tf.equal(preds[i], preds[i + 1]))

    print('MC Dropout tests passed.')

    model = model_ens
    model._model.compile(
        optimizer,
        loss=loss_criterion,
        metrics=["accuracy"]
    )

    for images, labels in dataset_nojpeg.get_validation_generator(sf=SF):
        preds = model(images, training=False)

        assert tf.is_tensor(preds)
        assert preds.shape[0] == NUM_MODELS

        for i in range(NUM_MODELS - 1):
            assert preds[i].shape == preds[i + 1].shape
            assert not tf.reduce_all(tf.equal(preds[i], preds[i + 1]))

    print('Ensemble tests passed.')
'''
