# Tensor Networks for MNIST Classification

A short demonstration of Google's TensorNetwork library (https://arxiv.org/pdf/1905.01330) for the classification of MNIST.
The goal is to employ a few different architectures for this:

1. Matrix product states (http://papers.nips.cc/paper/6211-supervised-learning-with-tensor-networks and https://arxiv.org/pdf/1906.06329).
2. Tree tensor networks (https://arxiv.org/abs/1901.02217 and https://arxiv.org/abs/1801.00315).
3. Implementing architectures (1) and (2) in a quantum circuit (https://arxiv.org/pdf/1803.11537).

Motivating Question: In machine learning, tensors are used, where they are usually treated as multidimensional arrays. 
In physics, tensors are also used, where they are treated as multilinear maps invariant under coordinate transformations.
Can we use the extensive tensor network literature developed in physics to improve machine learning? (For more information
by experts, please see the references I've linked above.)

I'll demonstrate a few strategies for doing MNIST classification tasks using tensor networks. Namely, the matrix product state and
tree tensor network. The MPS architecture doesn't preserve the dimensionality of the image, while the TTN
does to some extent.