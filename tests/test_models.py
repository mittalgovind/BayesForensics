import os
import pytest

import numpy as np

from helpers import dataset, utils, metrics, tf_helpers

from models import compression, tfmodel, pipelines
from compression import codec

tf_helpers.disable_warnings()
tf_helpers.disable_gpu()
os.environ['DISABLE_TQDM'] = 'True'
utils.setup_logging(level='ERROR')


TEST_CAMERA = 'D90'
BATCH_SIZE = 5
SSIM_THRESHOLD = 0.9
SSIM_DCN_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def raw_data():
    return dataset.Dataset(TEST_CAMERA, n_images=0, v_images=BATCH_SIZE)


@pytest.fixture(scope="module")
def classic_isp_restored():
    return pipelines.ClassicISP.restore(camera='D90')


@pytest.fixture(scope="module")
def classic_isp_factory_dict():
    return utils.factory({
        'class': 'ClassicISP',
        'checkpoint': 'data/models/isp/ClassicISP_3x3_32-32-32-32-3R/'
    })


@pytest.fixture(scope="module")
def classic_isp_factory_dict_default():
    return utils.factory({
        'class': 'ClassicISP',
        'checkpoint': None
    })


@pytest.fixture(scope="module")
def dnet_factory():
    return utils.factory({
        'class': 'DNet',
        'checkpoint': 'data/models/nip/D90/dnet'
    })


@pytest.fixture(scope="module")
def dnet_restored():
    return pipelines.DNet.restore('data/models/nip/D90/dnet')


def test_restore_by_loading():
    # 1. Manual - create model instance and load weights
    dcn = compression.TwitterDCN(rounding="soft-codebook", n_features=16)
    dcn.load_model('data/models/dcn/baselines/16c')


def test_restore():
    # 2. Restore a pre-trained model of a known class
    dcn = compression.TwitterDCN.restore('data/models/dcn/baselines/16c/', key='codec')


def test_restore_generic():
    # 3. Restore a pre-trained model from a known Python module
    dcn = tfmodel.restore('data/models/dcn/baselines/16c/', compression, key='codec')


def test_restore_codec():
    # 4. Convenience function to restore specific model classes (here, compression)
    dcn = codec.restore('16c')


def test_baseline_dcs():
    for baseline in [16, 32, 64]:
        dcn = codec.restore(f'{baseline}c')
        assert dcn.latent_shape[-1] == baseline, f'DCN: Latent shape does not the expected number of channels ({baseline})!'


def test_dcn(raw_data):
    dcn = codec.restore('64c')

    batch_raw, batch_rgb = raw_data.next_validation_batch(0, BATCH_SIZE)
    ssims = metrics.ssim(batch_rgb, dcn.process(batch_rgb).numpy())

    assert np.all(ssims > SSIM_DCN_THRESHOLD), f'DCN: Got SSIM scores < {SSIM_DCN_THRESHOLD}'


@pytest.mark.parametrize('isp',
                         [pytest.lazy_fixture('classic_isp_factory_dict'),
                          pytest.lazy_fixture('classic_isp_factory_dict_default'),
                          pytest.lazy_fixture('classic_isp_restored'),
                          pytest.lazy_fixture('dnet_restored'),
                          pytest.lazy_fixture('dnet_factory')])
def test_isps(isp, raw_data):
    batch_raw, batch_rgb = raw_data.next_validation_batch(0, BATCH_SIZE)
    ssims = metrics.ssim(batch_rgb, isp.process(batch_raw).numpy())

    assert np.all(ssims > SSIM_THRESHOLD), f'DNet: Got SSIM scores < {SSIM_THRESHOLD}'
