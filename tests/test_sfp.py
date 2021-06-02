import pytest

import tensorflow as tf

from workflows.scaling_factor import (
    ScalingFactor,
    ScalingFactorDataset,
    validate,
    sf_plot
)


PATCH_SIZE = 128
SF = 0.25


@pytest.fixture
def dataset_nojpeg():
    return ScalingFactorDataset(
        data_dir="native12k",
        load="y",
        scales="0.25,1.00",
        sampling_method="nearest",
        n_classes=31,
        n_images=10,
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
        num_models=1,
    )


@pytest.fixture
def model_ens():
    return ScalingFactor(
        n_classes=31,
        uncertainty_method="ensemble",
        num_models=2,
    )


def test_dataset(dataset_nojpeg, dataset_jpeg):
    for data in [dataset_nojpeg, dataset_jpeg]:
        for images, labels in data.get_validation_generator(sf=SF):
            assert tf.is_tensor(images)
            assert tf.is_tensor(labels)
    
            assert images.shape[0] == labels.shape[0]
            assert images.shape[1] == int(SF * PATCH_SIZE)
            assert images.shape[2] == int(SF * PATCH_SIZE)
