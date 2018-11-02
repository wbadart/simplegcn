#!/usr/bin/env python3

"""
simplegcn/model.py

This module contains the core implementation of the GCN model.

Will Badart
created: OCT 2018
"""

__all__ = [
    'GCN',
    'GCNLayer',
    'relu',
    'softmax',
]

import warnings
import numpy as np


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


class GCNLayer(object):
    """Implements one layer of a GCN using the forward propagation rule
    described in Kipf & Welling (ICLR 2017). See:
    https://arxiv.org/pdf/1609.02907.pdf
    """

    def __init__(self, d_in: int, d_out: int):
        """Initialize layer with `d_in' input dimensionality and `d_out' output
        dimensionality. Initialize weights randomly and bias to 0.

        Parameters
        ==========
        d_in : int
            Dimensionality of the input
        d_out : int
            Dimensionality of the output
        """
        self._d_in = d_in
        self._d_out = d_out

        self._W = np.random.random((d_in, d_out))
        self._b = np.zeros(d_out)

    def __call__(self, X: np.ndarray, adj: np.ndarray):
        """Perform a forward pass through the network with the given input
        features `X' and adjacency matrix `adj'. Note, while the paper
        indicates that the layer propagation rule *includes* the application of
        the activation function, it is excluded here to decouple the activation
        from the convolution.

        Parameters
        ==========
        X : numpy.ndarray
            Input feature matrix (may be output of previous layer)
        adj : numpy.ndarray
            Adjacency matrix representing the learned graph

        Returns
        =======
        output : numpy.ndarray
            Output of the layer, per the forward propagation rule described in
            Kipf & Welling.
        """
        A = adj + np.identity(len(adj))
        with warnings.catch_warnings():  # We know/ don't care about DivideByZero
            warnings.simplefilter('ignore')
            D = np.power(self._degree_matrix(A), -1/2)
        D = np.where(np.isinf(D), 0, D)
        return D @ A @ D @ X @ self._W + self._b  # '@' = matmul, see PEP 465

    @staticmethod
    def _degree_matrix(adj: np.ndarray):
        """Compute the diagonal degree matrix of a graph (in adjacency matrix
        form).
        """
        return np.diag(adj.sum(axis=1))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(d_in={self._d_in}, d_out={self._d_out})')


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
