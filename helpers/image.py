# -*- coding: utf-8 -*-
"""
Helper functions for image processing.
"""
import numpy as np
from scipy import fftpack as sfft


def sliding_window(arr, window):
    if arr.ndim != 3:
        raise ValueError("The input array needs to be 3-D - (h,w,c)!")
    n_windows = (arr.shape[0] // window) * (arr.shape[1] // window)
    batch = np.zeros((n_windows, window, window, arr.shape[-1]), dtype=arr.dtype)
    window_id = 0
    for x in range(arr.shape[1] // window):
        for y in range(arr.shape[0] // window):
            batch[window_id] = arr[
                y * window : (y + 1) * window, x * window : (x + 1) * window, :
            ]
            window_id += 1
    return batch


def batch_gamma(batch_p, gamma=None):
    if gamma is None:
        gamma = np.array(
            np.random.uniform(low=0.25, high=3, size=(len(batch_p), 1, 1, 1)),
            dtype=np.float32,
        )
    elif type(gamma) is float:
        gamma = gamma * np.ones((len(batch_p), 1, 1, 1))

    return np.power(batch_p, 1 / gamma).clip(0, 1)


def crop_middle(image, patch=128):
    image = image.squeeze()

    xx = (image.shape[0] - patch) // 2
    yy = (image.shape[1] - patch) // 2

    if image.ndim == 2:
        return image[xx : (xx + patch), yy : (yy + patch)]
    elif image.ndim == 3:
        return image[xx : (xx + patch), yy : (yy + patch), :]
    else:
        raise ValueError("Invalid image size!")


def fft_log_norm(x, boost=10, perc=0):
    x = x.squeeze()
    if x.ndim == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    if x.ndim != 3:
        raise ValueError("Only single images can be accepted as input.")
    y = np.zeros_like(x)
    for i in range(x.shape[-1]):
        y[:, :, i] = np.abs(sfft.fft2(x[:, :, i]))
        y[:, :, i] = sfft.fftshift(y[:, :, i])
        y[:, :, i] = np.log(boost + y[:, :, i])
        y[:, :, i] = normalize(y[:, :, i], perc)
    return y


def cati(*args):
    """
    Concatenate arrays along the image dimension. Should handle various combinations of arrays / lists.
    """

    arrays = []

    for i, item in enumerate(args):

        if isinstance(item, np.ndarray):
            if item.ndim == 3:
                item = np.expand_dims(item, 0)

            if item.ndim != 4:
                raise ValueError(
                    f"Shape of element {i} ({item.shape}) is not supported!"
                )

        else:
            item = np.concatenate(
                [x if x.ndim == 4 else np.expand_dims(x, axis=0) for x in item]
            )
            if item.ndim != 4:
                item = item.squeeze()
            if item.ndim != 4:
                raise ValueError(
                    f"Shape of element {i} ({item.shape}) is not supported!"
                )

        arrays.append(item)

    out = np.concatenate(arrays, axis=0)
    return out if out.ndim == 4 else out.squeeze()


def catc(*args):
    return np.concatenate(args, axis=-1)


def normalize(x, perc=0, mode=None, keep_axis=None):
    """
    Normalize the input array to [0, 1]. Optionally, cut top and bottom outliers (based on percentiles).
    """

    if keep_axis is None:
        if mode is not None:
            if mode == "channel":
                axis = tuple(range(x.ndim - 1))
            elif mode == "batch":
                axis = tuple(range(1, x.ndim))
            else:
                raise ValueError(f"Unknown aggregation mode! {mode}")
        else:
            axis = None
    else:
        axis = set(range(x.ndim))
        discard_axis = keep_axis if keep_axis >= 0 else x.ndim + keep_axis
        if discard_axis not in axis:
            raise ValueError(
                f"Trying to drop a non-existent axis! axis={axis} drop={discard_axis}"
            )
        axis.discard(discard_axis)
        axis = tuple(axis)

    if perc == 0:
        mn = np.min(x, axis=axis, keepdims=True)
        mx = np.max(x, axis=axis, keepdims=True)
    else:
        mn = np.percentile(x, perc, axis=axis, keepdims=True)
        mx = np.percentile(x, 100 - perc, axis=axis, keepdims=True)

    return ((x - mn) / (mx - mn + 1e-9)).clip(0, 1)


def normalize_residual(x):
    return x / np.abs(x).max() / 2 + 0.5


# def encode_residuals(x):
#     x = x / np.abs(x).max()
#     y = np.zeros_like(x)
#     y[x > 0] = np.maximum(x, 0)
#     y[x < 0] = np.maximum(-x, 0)
#     return y
