import pytest

from copy import deepcopy

import tensorflow as tf

from models.bayes import DeepEnsemble
from workflows.scaling_factor.model import SFP
from helpers.dataset import Dataset


@pytest.fixture
def ensemble_model():
    return DeepEnsemble(
        SFP(
            "vanilla",
            c_filters=(32, 32, 32, 32),
            d_filters=(32, 16, 31),
            kernel=5,
            activation="leaky_relu",
            trainable_residual=True,
            drop=0.1,
            append_rgb=False,
        ),
        5,
    )


@pytest.fixture
def dataset():
    return Dataset(
        data_dir="/scratch/jms1595/neural-imaging-dev/data/rgb/native12k",
        load="y",
        n_images=10,
        v_images=10,
        seed=69,
    )


def test_run(ensemble_model, dataset, tmpdir):
    assert ensemble_model.n_models == len(ensemble_model.models)

    for model in ensemble_model.models:
        assert type(model) == SFP

    batch = dataset.next_training_batch(0, 10, 128)
    results_0 = ensemble_model(batch)
    assert tf.is_tensor(results_0)
    # assert results_0.shape = (31, ensemble_model.n_models)

    ensemble_model.save_weights(tmpdir)
    ensemble_model.load_weights(tmpdir)

    results_1 = ensemble_model(batch)
    assert tf.is_tensor(results_1)

    tf.debugging.assert_equal(results_0, results_1)
