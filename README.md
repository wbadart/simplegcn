# simplegcn

`simplegcn` implements a [Graph Convolutional Network][gcn] using PyTorch and
is based strongly on Thomas Kipf's [research][tkifp site] and [old
implementation][pygcn].

[gcn]: https://arxiv.org/abs/1609.02907
[tkipf site]: https://tkipf.github.io/graph-convolutional-networks/
[pygcn]: https://github.com/tkipf/pygcn

The goal of this package is to be as simple to use as possible, and aims to
function for any arbitrary graph. Introspection into the network is enabled by
[tensorboardX][tensorboardX], a tool which summarizes PyTorch information in a
format that [tensorboard][tensorboard] can consume. Finally, the package aims
to support GPU-enabled models out of the box.

[tensorboardX]: https://github.com/lanpa/tensorboardX
[tensorboard]: https://www.tensorflow.org/guide/summaries_and_tensorboard
