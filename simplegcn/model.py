#!/usr/bin/env python3

"""
simplegcn/model.py

This module contains the core implementation of the GCN model.

Will Badart
created: OCT 2018
"""

__all__ = [
    'GCNLayer',
    'adj_normalized',
    'degree_matrix',
]

import warnings
import numpy as np


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

        self._output = None

    @property
    def output(self):
        return self._output

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
            Adjacency matrix representing the learned graph should be
            normalized against the degree matrix. See 'adj_normalized'.

        Returns
        =======
        output : numpy.ndarray
            Output of the layer, per the forward propagation rule described in
            Kipf & Welling.
        """
        self._output = adj @ X @ self._W + self._b  # '@' = matmul, see PEP 465
        return self._output

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(d_in={self._d_in}, d_out={self._d_out})')


def degree_matrix(adj: np.ndarray):
    """Compute the degree matrix of the given adjacency matrix."""
    return np.diag(adj.sum(axis=1))


def adj_normalized(adj: np.ndarray):
    """Adjacency matrix normalized according to the pre-processing described in
    Kipf & Welling.
    """
    adj_norm = np.identity(len(adj))
    with warnings.catch_warnings():  # We know/ don't care about DivideByZero
        warnings.simplefilter('ignore')
        diag = np.power(degree_matrix(adj_norm), -1/2)
    diag = np.where(np.isinf(diag), 0, diag)
    return diag @ adj_norm @ diag
