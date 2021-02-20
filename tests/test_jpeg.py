import pytest

import numpy as np

from models import jpeg
from helpers import dataset, metrics, tf_helpers, utils, stats


tf_helpers.disable_warnings()
tf_helpers.disable_gpu()
utils.setup_logging(level="ERROR")


@pytest.fixture(scope="session")
def data():
    data = dataset.Dataset(
        "native12k", load="y", n_images=0, v_images=50, val_n_patches=4
    )
    return data


def test_compare_soft_libjpeg(data, correlation_threshold=0.97):
    jpg_soft = jpeg.JPEG(codec="soft")
    jpg_libjpeg = jpeg.JPEG(codec="libjpeg")

    batch_y = data.next_validation_batch(0, data.count_validation)
    q_levels = list(range(10, 100, 10))
    q_correlations = np.zeros((len(q_levels),))

    for qi, q in enumerate(q_levels):
        batch_s = jpg_soft.process(batch_y, q).numpy()
        batch_l = jpg_libjpeg.process(batch_y, q)
        ssim_s = metrics.ssim(batch_y, batch_s)
        ssim_l = metrics.ssim(batch_y, batch_l)
        q_correlations[qi] = stats.corrcoeff(ssim_s, ssim_l)

    assert len(q_correlations) == len(q_levels), "#(QF levels) != #(correlation scores)"
    assert np.all(
        q_correlations > correlation_threshold
    ), f"Got SSIM correlation scores < {correlation_threshold}"
