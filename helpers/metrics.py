import numpy as np
from skimage import metrics

def ssim(a, b):
    return metrics.structural_similarity(a, b, multichannel=True, data_range=1)

def psnr(a, b):
    return metrics.peak_signal_noise_ratio(a, b, data_range=1)

def mse(a, b):
    return metrics.mean_squared_error(a, b)

def mae(a, b):
    return np.mean(np.abs(a - b))

def batch(a, b, metric=ssim):
    assert len(a) == len(b), 'Image batches must be of the same length'
    return np.mean([metric(a[r], b[r]) for r in range(len(a))])
