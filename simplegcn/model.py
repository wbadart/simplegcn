#!/usr/bin/env python3

"""
simplegcn/model.py

This module contains the core implementation of the GCN model.

Will Badart
created: OCT 2018
"""

import numpy as np

__all__ = [
    'GCN',
    'GCNLayer',
]


class GCN(object):
    """The `GCN' class represents a complete network, to be trained and used
    for inference at the graph (as opposed to node) level.
    """


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
        D = np.power(self._degree_matrix(adj), -1/2)
        A = adj + np.identity(len(A))
        return D * A * D * X * self._W

    @staticmethod
    def _degree_matrix(adj: np.ndarray):
        """Compute the diagonal degree matrix of a graph (in adjacency matrix
        form).
        """
        return np.diag(adj.sum(axis=1).astype(float))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(d_in={self._d_in}, d_out={self._d_out})')
