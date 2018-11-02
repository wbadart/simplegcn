#!/usr/bin/env python3

"""
simplegcn/demo.py

While simplegcn.model gives the core implementation of the graph convolutional
layer, this module seeks to demonstrate how you could use that layer in a
broader network.

This module demonstrates the canonical two-layer GCN with ???

Will Badart
created: NOV 2018
"""

__all__ = [
    'GCN',
    'relu',
    'softmax',
]

import numpy as np

from simplegcn import GCNLayer


class GCN(object):
    """The `GCN' class represents a complete network, to be trained and used
    for inference at the graph (as opposed to node) level.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        """Initialize a GCN with random weights and bias 0 at all layers. Uses
        the architecture common in GCN literature: 2 convolutional layers with
        relu activation. TODO: implement dropout.

        Parameters
        ==========
        d_int : int
            Input dimensionality (number of features per node of the graph)
        d_hidden : int
            Size of the hidden layer
        d_out : int
            Output dimensionality (number of output classes)
        """
        self._d_in = d_in
        self._d_hidden = d_hidden
        self._d_out = d_out

        self._gc1 = GCNLayer(d_in, d_hidden)
        self._gc2 = GCNLayer(d_hidden, d_out)

    def __call__(self, X: np.ndarray, adj: np.ndarray):
        """Forward propagate through the network.

        Parameters
        ==========
        X : numpy.ndarray
            Features of the learned graph
        adj : numpy.ndarray
            Adjacency matrix of the learned graph.

        Returns
        =======
        output : np.ndarray
            Predicted classes of each node(?)
        """
        H = self._gc1(X, adj)
        H = relu(H)
        H = self._gc2(H, adj)
        return softmax(H)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(d_in={self._d_in}, '
            f'd_hidden={self._d_hidden}, d_out={self._d_out})')


def relu(X: np.ndarray):
    """Rectified linear unit. Sends negative values to zero and preserves
    positive values.

    Parameters
    ==========
    X : numpy.ndarray
        Array-like structure on which to compute relu. Should be compatible
        with numpy ufuncs. (So it could theoretically be a scalar.)

    Returns
    =======
    output : numpy.ndarray
        Rectified value(s)
    """
    return np.maximum(X, np.zeros(np.shape(X)))


def softmax(X: np.ndarray):
    """Compute the softmax of an array-like object. Defined as the exponential
    of the input divided by the sum of the exponential. See:

    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

    for discussion.

    Parameters
    ==========
    X : numpy.ndarray
        Array-like object on which to compute softmax

    Returns
    =======
    output : numpy.ndarray
        Softmax'd array
    """
    e = np.exp(X - X.max(axis=1).reshape((-1, 1)))  # Subtract max for numeric stability
    return e / e.sum(axis=1, keepdims=True)


def softmax_prime(X: np.ndarray):
    """Compute the gradient of the softmax.

    See:
    https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d

    Parameters
    ==========
    X : numpy.ndarray
        Softmax'd matrix

    Returns
    =======
    output : numpy.ndarray
        Derivative of the softmax values
    """
    s = X.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)
