# -*- coding: UTF-8 -*-
# Python implementation of the MLE engine for PRNU extraction
# Adapted from ISPL's numpy implementation by L. Bondi, P. Bestagini and N. Bonettini:
# https://github.com/polimi-ispl/prnu-python

from multiprocessing import Pool, cpu_count

import numpy as np
import pywt

from prnulib import commons


def estimate_prnu(imgs, as_rgb=True, expand=True, **kwargs):
    k = _extract_multiple_aligned((255 * imgs).astype(np.uint8), as_rgb=as_rgb,
                                  **kwargs)
    if expand:
        k = np.expand_dims(k, axis=0 if as_rgb else (0, -1))
    return k


def extract_residual(img, as_rgb=True, processes=None, **kwargs):
    if img.ndim == 3:
        return _extract_single((255 * img).astype(np.uint8), as_rgb=as_rgb,
                               **kwargs)

    elif img.ndim == 4:
        processes = processes or cpu_count()

        if processes == 1:
            return np.concatenate([np.expand_dims(
                _extract_single((255 * x).astype(np.uint8), as_rgb=as_rgb,
                                **kwargs), axis=0) for x in img], axis=0)
        else:
            with Pool(processes=processes) as pool:
                args_list = [((255 * x).astype(np.uint8)) for x in img]
                return np.concatenate([np.expand_dims(x, axis=0) for x in
                                       pool.map(_extract_single, args_list)],
                                      axis=0)


def _extract_single(im: np.ndarray, levels: int = 4, sigma: float = 4,
                    wdft_sigma: float = 0, as_rgb: bool = True) -> np.ndarray:
    """
    Extract noise residual from a single image
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :param wdft_sigma: estimated DFT noise power
    :return: noise residual
    """

    W = _noise_extract(im, levels, sigma)

    if not as_rgb:
        W = commons._rgb2gray(W)
        W = commons._zero_mean_total(W)
        W = commons._wiener_dft(W, W.std(ddof=1)).astype(np.float32)
    else:
        for c in range(im.shape[-1]):
            W[..., c] = commons._zero_mean_total(W[..., c])
            W[..., c] = commons._wiener_dft(W[..., c],
                                            W[..., c].std(ddof=1)).astype(
                np.float32)

    return W


def _noise_extract(im: np.ndarray, levels: int = 4,
                   sigma: float = 4) -> np.ndarray:
    """
    NoiseExtract as from Binghamton toolbox.

    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """
    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    if im.shape[0] < 128 or im.shape[1] < 128:
        levels = 3

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = commons._wiener_adaptive(
                    wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        W[:, :, ch] = wrec

    W = W.squeeze()
    W = W[:im.shape[0], :im.shape[1]]

    return W


def _noise_extract_compact(args):
    """
    Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations
    :param args: (im, levels, sigma), see noise_extract for usage
    :return: residual, multiplied by the image
    """
    w = _noise_extract(*args)
    im = args[0]
    return w * im / 255.0


def _extract_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5,
                              processes: int = None,
                              as_rgb: bool = False) -> np.ndarray:
    """
    Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented
    :param tqdm_str: tqdm description (see tqdm documentation)
    :param batch_size: number of parallel processed images
    :param processes: number of parallel processes
    :param imgs: list of images of size (H,W,Ch) and type np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: PRNU
    """
    assert (isinstance(imgs[0], np.ndarray))
    assert (imgs[0].ndim == 3)
    assert (imgs[0].dtype == np.uint8)

    h, w, ch = imgs[0].shape

    if h < 128 or w < 128:
        levels = 3

    processes = processes or cpu_count()

    if processes is None or processes > 1:
        args_list = [(im, levels, sigma) for im in imgs]
        pool = Pool(processes=processes)
        NN = np.sum(pool.map(commons._inten_sat_compact, args_list), axis=0)
        RPsum = np.sum(pool.map(_noise_extract_compact, args_list), axis=0)
        pool.close()

    else:
        RPsum = np.sum(
            (_noise_extract_compact((im, levels, sigma)) for im in imgs),
            axis=0)
        NN = np.sum(
            ((commons._inten_scale(im) * commons._saturation(im)) ** 2 for im
             in imgs), axis=0)

    K = RPsum / (NN + 1)

    if not as_rgb:
        K = commons._rgb2gray(K)
        K = commons._zero_mean_total(K)
        K = commons._wiener_dft(K, K.std(ddof=1)).astype(np.float32)
    else:
        for c in range(3):
            K[..., c] = commons._zero_mean_total(K[..., c])
            K[..., c] = commons._wiener_dft(K[..., c],
                                            K[..., c].std(ddof=1)).astype(
                np.float32)

    return K
