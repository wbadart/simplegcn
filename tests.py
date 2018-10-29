#!/usr/bin/env python3

"""
tests/test_model.py

Unit tests of `simplegcn.model' module.

Will Badart
created: OCT 2018
"""

import numpy as np
from pytest import fixture

from simplegcn import GCN, GCNLayer, relu, softmax


RANDOM_STATE = 0xdecafbad
np.random.seed(RANDOM_STATE)

FEATURES_IN = 3
FEATURES_HIDDEN = 5
FEATURES_OUT = 2

LAYER = GCNLayer(FEATURES_IN, FEATURES_OUT)
MODEL = GCN(FEATURES_IN, FEATURES_HIDDEN, FEATURES_OUT)

ADJ = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]])


@fixture
def X():
    """Generate a random input matrix"""
    return np.random.random((len(ADJ), FEATURES_IN))


class TestNetwork(object):
    """Tests of overarching network onject, GCN."""

    def test_repr(self):
        """Make sure repr shows expected values."""
        assert repr(MODEL) == 'GCN(d_in=3, d_hidden=5, d_out=2)'

    def test_output_dim(self, X):
        """Make sure output has expected dimensionality."""
        assert MODEL(X, ADJ).shape == (len(ADJ), FEATURES_OUT)

    def test_output_vals(self, X):
        y = MODEL(X, ADJ)
        assert not (np.isnan(y) | np.isinf(y)).any()


class TestLayer(object):
    """Tests of the GCNLayer class."""

    def test_repr(self):
        """Make sure repr shows expected values."""
        assert repr(LAYER) == 'GCNLayer(d_in=3, d_out=2)'

    def test_output_dim(self, X):
        """Make sure dimensions line up."""
        assert LAYER(X, ADJ).shape == (len(ADJ), FEATURES_OUT)

    def test_output_vals(self, X):
        """Make sure output is all valid."""
        y = LAYER(X, ADJ)
        assert not (np.isnan(y) | np.isinf(y)).any()


class TestUtils(object):
    """Test the other utilities in the `simplegcn.model' module."""

    def test_relu(self):
        """Make sure relu turns negative values to 0 and nothing else."""
        x = np.array([0, 1, 2, 3, -1, -3, 0.1])
        for e in x:
            assert relu(e) == max(e, 0)

    def test_softmax(self):
        """Make sure softmax is accurate on matrices."""
        def sk_softmax(x):
            # Pulled from sklearn.utils.extmath.softmax
            max_prob = np.max(x, axis=1).reshape((-1, 1))
            x -= max_prob
            x = np.exp(x)
            sum_prob = np.sum(x, axis=1).reshape((-1, 1))
            x /= sum_prob
            return x
        x = np.array([
            [1, 2, 3, 6],
            [2, 4, 5, 6],
            [3, 8, 7, 6]])
        expected = sk_softmax(x)
        assert np.allclose(softmax(x), expected)
