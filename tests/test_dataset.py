import os
import numpy as np

import pytest

from helpers import dataset, utils

os.environ['DISABLE_TQDM'] = 'True'
utils.setup_logging(level='WARNING')


__ERRORS = {
    'count': 'Invalid data count',
    'shape': 'Invalid {} shape',
    'not_loaded': 'Some images don\'t seem to be loaded',
    'dtype': 'Invalid data type - expected float32',
    'btype': 'Invalid batch type',
    'imsize': 'Invalid image size'
}


def test_full_dataset():
    data = dataset.Dataset('D90', n_images=2, v_images=2, val_n_patches=4)

    assert 'x' in data.data['training'],  __ERRORS['not_loaded']
    assert 'x' in data.data['validation'],  __ERRORS['not_loaded']
    assert data.count_training == 2, __ERRORS['count']
    assert data.count_validation == 2 * 4, __ERRORS['count']
    assert data.loaded_data == 'raw+rgb', __ERRORS['not_loaded']
    assert data.rgb_patch_size == 128, __ERRORS['imsize']
    assert data.rgb_shape == (2868, 4310, 3), __ERRORS['imsize']

    batch = data.next_training_batch(0, 2)

    assert isinstance(batch, tuple), __ERRORS['btype']
    assert batch[0].dtype == np.float32, __ERRORS['dtype']
    assert batch[1].dtype == np.float32, __ERRORS['dtype']
    assert batch[0].shape == (2, 64, 64, 4), __ERRORS['shape'].format('RAW')
    assert batch[1].shape == (2, 128, 128, 3), __ERRORS['shape'].format('RGB')
    assert not any(np.sum(batch[1], axis=(1, 2, 3)) == 0), __ERRORS['not_loaded']

    batch = data.next_validation_batch(0, 8)
    assert batch[0].shape == (8, 64, 64, 4), __ERRORS['shape'].format('RAW')
    assert batch[1].shape == (8, 128, 128, 3), __ERRORS['shape'].format('RGB')


def test_raw_dataset():
    data = dataset.Dataset('D90', n_images=2, v_images=2, val_n_patches=4, load='x')
    assert data.loaded_data == 'raw', __ERRORS['not_loaded']

    with pytest.raises(ValueError):
        batch = data.next_training_batch(0, 2, 0, 'flat')

    batch = data.next_training_batch(0, 2, 0, None)
    assert isinstance(batch, np.ndarray), __ERRORS['btype']
    assert batch.shape == (2, 64, 64, 4), __ERRORS['shape'].format('RAW')

    batch = data.next_training_batch(0, 2, 96, None)
    assert batch.shape == (2, 48, 48, 4), __ERRORS['shape'].format('RAW')


def test_rgb_dataset():
    data = dataset.Dataset('D90', n_images=2, v_images=2, val_n_patches=4, load='y')
    assert data.loaded_data == 'rgb', __ERRORS['not_loaded']

    batch = data.next_training_batch(0, 2, 0, 'flat-aggressive')
    assert isinstance(batch, np.ndarray), __ERRORS['btype']
    assert batch.shape == (2, 128, 128, 3), __ERRORS['shape'].format('RGB')

    batch = data.next_training_batch(0, 2, 96, 'flat-aggressive')
    assert batch.shape == (2, 96, 96, 3), __ERRORS['shape'].format('RGB')


def test_preloaded_dataset():
    data = dataset.Dataset('D90', n_images=2, v_images=2, val_n_patches=4, presample_epochs=5, train_rgb_patch_size=256)
    assert data.count_training == 2 * 5, __ERRORS['count']
    assert data.count_validation == 2 * 4, __ERRORS['count']
    assert data.rgb_shape == (256, 256, 3), __ERRORS['shape'].format('RAW image')

    batch = data.next_training_batch(0, 10)
    assert batch[0].shape == (10, 64, 64, 4), __ERRORS['shape'].format('RAW')

    batch = data.next_training_batch(0, 3, 32, 'flat')
    assert batch[0].shape == (3, 16, 16, 4), __ERRORS['shape'].format('RAW')
