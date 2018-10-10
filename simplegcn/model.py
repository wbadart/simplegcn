#!/usr/bin/env python3

"""
simplegcn/model.py

This module defines the structure of a single GCN layer, and one example of a
full network. Users may create arbitrary networks from GraphConvolution.

Will Badart <badart_william (at) bah (dot) com>
created: OCT 2018
"""

import torch
from torch.nn import functional as F


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network. It's a linear model which also accounts for
    the structure of a graph associated with the input.
    """

    def __init__(self, n_features, n_hidden, n_classes, p_dropout):
        """
        Construct a GCN with `n_features` features, `n_hidden` hidden layers,
        `n_classes` output classes, and `p_dropout` probability of dropping
        out a given node from the network.
        """
        super().__init__()
        self.gc1 = GraphConvolution(D_in=n_features, D_out=n_hidden)
        self.gc2 = GraphConvolution(D_in=n_hidden, D_out=n_classes)
        self._p_dropout = p_dropout

    def forward(self, x, adj):
        """
        Forward propagation of input through GCN:

            X -> 1st Graph Convolution
              -> ReLU
              -> Dropout
              -> 2nd Graph Convolution
              -> Log Softmax
         -> Z
        """
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self._p_dropout)  # training=self.training?
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def train_one_pass(self, X, adj, labels, optimizer):
        self.train()
        optimizer.zero_grad()

        output = self(X, adj)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

        return loss


    def __repr__(self):
        return (
            f'{self.__class__.__qualname__}('
            f'n_features={self.gc1.d_in}, n_hidden={self.gc1.d_out}, '
            f'n_classes={self.gc2.d_out}, p_dropout={self._p_dropout})')


class GraphConvolution(torch.nn.Module):
    """
    Each layer is a chosen function of the previous layer's output and the
    graph (adjacency matrix). This function defines the forward propagation of
    data through the network. See GraphConvolution.forward for further details.
    """

    def __init__(self, D_in, D_out):
        """
        Initialize the layer with input dimensionality `D_in` and output
        dimensionality `D_out`.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(D_in, D_out).float())
        self.bias = torch.nn.Parameter(torch.zeros(D_out))

        self.d_in, self.d_out = D_in, D_out

    def forward(self, x, adj):
        """
        Here, we choose to implement the same forward propagation rule as Kipf
        in his original implementation, simply: multiply the input values by
        the layer's weights, then multiply that by the adjacency matrix
        (represented as a sparse matrix), and finally, add the biases.
        """
        support = torch.mm(x, self.weight)
        return torch.mm(adj, support) + self.bias

    def __repr__(self):
        return (
            f'{self.__class__.__qualname__}'
            f'(D_in={self.d_in}, D_out={self.d_out})')


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
