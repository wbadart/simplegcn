#!/usr/bin/env python3

"""
simplegcn/__main__.py

Main entry point for running the simplegcn module as a command. Reads a
serialized graph from disk, trains a GCN based on it, and evaluates that model.

Will Badart <badart_william (at) bah (dot) com
created: OCT 2018
"""

from collections import namedtuple
from functools import wraps
from importlib import import_module
from pathlib import Path

from pudb import post_mortem
from sklearn.model_selection import train_test_split as sk_ttsplit

import networkx as nx
import numpy as np
import torch

from simplegcn.model import GCN, accuracy


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
            enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
            dtype=np.int32)
    return labels_onehot


def inspect(f):
    """Try calling a function and run a post mortem if it fails."""
    @wraps(f)
    def _impl(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            post_mortem()
    return _impl


def train_test_split(X, y, G):
    """
    Split data `X` and labels `y` into a training partition and a testing
    partition. Also generates appropriate adjacency matrices for the split.
    Assumes that the first column of `X` is the node ID.
    """
    Split = namedtuple(
        'Split', 'X_train, X_test, y_train, y_test, adj_train, adj_test')
    X_train, X_test, y_train, y_test = sk_ttsplit(X, y)
    idx_train, idx_test = X_train[:, 0], X_test[:, 0]
    X_train, X_test = X_train[:, 1:], X_test[:, 1:]
    def to_matrix(indices):
        mx = nx.to_numpy_matrix(G.subgraph(map(str, indices)))
        return mx + mx.T * (mx.T > mx) - mx * (mx.T > mx)
    adj_train, adj_test = map(to_matrix, (idx_train, idx_test))
    return Split(X_train, X_test, y_train, y_test, adj_train, adj_test)


@inspect
def main():
    """
    Main function to run from command line. Install the simplegcn package with
    pip and then run `simplegcn --help` to see command line usage.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Create, train, and evaluate a GCN')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='Number of training epochs to run (default:500)')
    parser.add_argument('path_feat', type=Path, metavar='FEAT_PATH',
                        help='Path to feature data file')
    parser.add_argument('path_edges', type=Path, metavar='EDGE_PATH',
                        help='Path to edgelist file')
    parser.add_argument('--cuda', action='store_true',
                        help='Perform all computations on the GPU')
    args = parser.parse_args()

    tensors = import_module('torch.cuda' if args.cuda else 'torch')

    dat = np.genfromtxt(args.path_feat, skip_header=1, dtype=np.int32)
    X, y = dat[:, :-1], dat[:, -1]

    # Make adjacency matrix symmetric and convert to tensor
    adj = nx.read_edgelist(args.path_edges)

    splits = train_test_split(X, y, adj)

    features_train = tensors.FloatTensor(splits.X_train)
    features_test = tensors.FloatTensor(splits.X_test)

    labels_train = tensors.LongTensor(
        encode_onehot(splits.y_train)).squeeze_()
    labels_test = tensors.LongTensor(
        encode_onehot(splits.y_test)).squeeze_()

    model = GCN(
        n_features=features_train.shape[1],
        n_hidden=10,
        n_classes=(labels_train.max()+1).data.numpy().astype(np.int32),
        p_dropout=0.5)

    if args.cuda:
        model = model.cuda(device=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adj_train = tensors.FloatTensor(splits.adj_train)
    adj_test = tensors.FloatTensor(splits.adj_test)

    for e in range(args.epochs):
        loss = model.train_one_pass(
            features_train, adj_train, labels_train, optimizer)
        if e % 10 == 0:
            model.eval()
            output = model(features_test, adj_test)
            acc = accuracy(output, labels_test)
            print(f'Epoch {e}: loss={loss}, accuracy={acc}')


if __name__ == '__main__':
    main()
