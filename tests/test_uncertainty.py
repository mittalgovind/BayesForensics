import pytest

import numpy as np

import helpers.uncertainty as un


MAX = int(2e16)
MIN = -int(2e16)


def test_1():
    """
    Yarin Gal's example 1 for classification uncertainty.
    All class probabilities become [1 0] after softmax.
    High confidence, therefore, all uncertainty measures are 0.
    """
    test = np.array(
        [
            [[MAX, MIN]],
            [[MAX, MIN]],
            [[MAX, MIN]],
            [[MAX, MIN]],
            [[MAX, MIN]],
            [[MAX, MIN]],
        ]
    )

    np.testing.assert_allclose(un.variation_ratio(test), [0], atol=1e-7)
    np.testing.assert_allclose(un.predictive_entropy(test), [0], atol=1e-7)
    np.testing.assert_allclose(un.mutual_information(test), [0], atol=1e-7)


def test_2():
    """
    Yarin Gal's example 2 for classification uncertainty.
    All class probabilities are predicted equal after softmax.
    Variation ratio and predictive entropy capture uncertainty in prediction,
    therefore, they attain their highest values (0.5 and 1 in binary classification).
    Mutual information captures model confidence in its output,
    so its uncertainty is 0.
    """
    test = np.array(
        [
            [[0.5, 0.5]],
            [[0.5, 0.5]],
            [[0.5, 0.5]],
            [[0.5, 0.5]],
            [[0.5, 0.5]],
            [[0.5, 0.5]],
        ]
    )

    np.testing.assert_allclose(un.variation_ratio(test), [0.5], atol=1e-7)
    np.testing.assert_allclose(un.predictive_entropy(test), [1], atol=1e-7)
    np.testing.assert_allclose(un.mutual_information(test), [0], atol=1e-7)


def test_3():
    """
    Yarin Gal's example 2 for classification uncertainty.
    Equal number of [1 0] and [0 1] class probabilities after softmax.
    All uncertainty measures attain their highest value (0.5, 1, and 1 in binary classification).
    """
    test = np.array(
        [
            [[MAX, MIN]],
            [[MIN, MAX]],
            [[MAX, MIN]],
            [[MIN, MAX]],
            [[MAX, MIN]],
            [[MIN, MAX]],
        ]
    )

    np.testing.assert_allclose(un.variation_ratio(test), [0.5], atol=1e-7)
    np.testing.assert_allclose(un.predictive_entropy(test), [1], atol=1e-7)
    np.testing.assert_allclose(un.mutual_information(test), [1], atol=1e-7)
