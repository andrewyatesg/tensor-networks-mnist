"""
[1] Stoudenmire, Schwab. Supervised Learning with Tensor Networks. http://papers.nips.cc/paper/6211-supervised-learning-with-tensor-networks.
"""
import math
import time

import numpy as np
import tensornetwork as tn
import os
import mnist
import matplotlib
import matplotlib.pyplot as plt
import pickle
import skimage.measure
import tensornetwork.visualization.graphviz
from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

f1 = lambda x: np.cos(np.pi * x / 2)
f2 = lambda x: np.sin(np.pi * x / 2)
f = lambda x: np.array(f1(x), f2(x))


def feature_map(Xs):
    (d, n) = Xs.shape
    Xs_f = np.zeros((2 * d, n))
    for i in range(d):
        for j in range(n):
            Xs_f[2 * i, j] = f1(Xs[i, j])
            Xs_f[2 * i + 1, j] = f2(Xs[i, j])
    return Xs_f


def load_MNIST_dataset():
    """
    Loads the MNIST dataset from data/ which has been gitignored.
    Credit: This function alone has been adapted from Cornell's CS 4787
    Large Scale Machine Learning course.
    :return:
    """
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training()

        # Begin training set compression
        Xs_tr = Xs_tr.reshape((60000, 28, 28))
        Xs_tr_compressed = np.zeros((60000, 14, 14))
        for i in range(60000):
            Xs_tr_compressed[i] = skimage.measure.block_reduce(Xs_tr[i], (2, 2), np.mean)
        Xs_tr = Xs_tr_compressed.reshape((60000, 14 ** 2))
        # End training set compression

        Xs_tr = Xs_tr.transpose() / 255.0
        # Xs_tr = Xs_tr.transpose()
        Xs_tr = feature_map(Xs_tr)
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing()

        # Begin test set compression
        Xs_te = Xs_te.reshape((10000, 28, 28))
        Xs_te_compressed = np.zeros((10000, 14, 14))
        for i in range(10000):
            Xs_te_compressed[i] = skimage.measure.block_reduce(Xs_te[i], (2, 2), np.mean)
        Xs_te = Xs_te_compressed.reshape((10000, 14 ** 2))
        # End test set compression

        Xs_te = Xs_te.transpose() / 255.0
        # Xs_te = Xs_te.transpose()
        Xs_te = feature_map(Xs_te)
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


def prediction(mps, img_feature_vector):
    prod = mps_product(mps, create_input_tensor(img_feature_vector))
    return np.argmax(np.abs(tn.contractors.auto(prod, output_edge_order=tn.get_all_dangling(prod)).tensor))


def model_error(mps, Xs, Ys):
    num_mis_classified = 0
    size = int(Xs.shape[1] / 1000)
    for i in range(size):
        pred = prediction(mps, Xs[:, np.random.randint(Xs.shape[1])])
        num_mis_classified = num_mis_classified + int(pred != np.argmax(Ys[:, i]))
    return float(num_mis_classified) / float(size)


def create_input_tensor(vector):
    """
    Creates the tensor corresponding to the features {vector} of a single image.
    :param vector: vector of the image being converted to tensor
    :param dim: dimension of each input node leg
    :return: a list of nodes of the tensor
    """
    d = vector.shape[0]
    input_tn = []
    # Loop through features
    for i in range(int(d / 2)):
        feature = np.array([vector[2 * i], vector[2 * i + 1]])
        input_tn.append(tn.Node(feature, axis_names=['in']))
    return input_tn


def project_input(input_tensor, right_part, previous_projection, output_idx, last_node):
    """
    Projects input tensor onto the MPS w/o the bond tensor.
    See FIG 6(c) for a drawing of this process.

    :param input_tensor: input image feature tensor
    :param right_part: tensors to the right of the bond tensor
    :return: input tensor projected onto left and right partitions
    """
    input_tensor = tn.replicate_nodes(input_tensor)
    right_part = tn.replicate_nodes(right_part)

    if previous_projection is not None:
        last_node = last_node.copy()
        leftmost_before = previous_projection[0] is None
        if leftmost_before:
            node2_prev = previous_projection[1].copy()
            node2_prev['in'] ^ last_node['in']
            node1 = tn.contractors.auto([node2_prev, last_node])
        else:
            node1_prev = previous_projection[0].copy()
            node2_prev = previous_projection[1].copy()
            node1_prev['in'] ^ last_node['l']
            node2_prev['in'] ^ last_node['in']
            node1 = tn.contractors.auto([node1_prev, node2_prev, last_node])
        node1.add_axis_names(['in'])
    else:
        node1 = None

    node2 = input_tensor[output_idx].copy()
    node3 = input_tensor[output_idx + 1].copy()

    # Connect right partition to input nodes
    for i in range(len(right_part) - 1):
        right_part[i]['r'] ^ right_part[i + 1]['l']
    offset = output_idx + 2
    for i, node in enumerate(right_part):
        node['in'] ^ input_tensor[offset + i]['in']
    node4 = tn.contractors.auto(right_part + input_tensor[offset:])
    node4.add_axis_names(['in'])

    return [node1, node2, node3, node4]


def mps_product(mps, input_tensor):
    input_tensor = tn.replicate_nodes(input_tensor)
    mps = tn.replicate_nodes(mps)

    # Fully connect mps
    for i in range(len(mps) - 1):
        mps[i]['r'] ^ mps[i + 1]['l']

    # Connect inputs to mps
    for i, node in enumerate(mps):
        node['in'] ^ input_tensor[i]['in']

    return mps + input_tensor


def sweeping_mps_optimization(Xs_tr, Ys_tr, alpha, bond_dim):
    """
    Run algorithm depicted in FIG. 6 of [1].
    :param bond_dim: bond dimension
    :param Xs_tr: training set images
    :param Ys_tr: training set one-hot encodings of image labels
    :param alpha: step size
    :return: trained MPS
    """
    (img_feature_dim, num_ex) = Xs_tr.shape
    img_dim = int(img_feature_dim / 2)
    num_labels = Ys_tr.shape[0]

    output_idx = 0
    mps = create_mps_state(img_dim, 2, bond_dim, num_labels)

    alpha = tn.Node(alpha)

    batch_size = 20
    sample_idxs = np.random.randint(num_ex, size=batch_size)
    previous_projections = [None] * batch_size

    # Stochastic Gradient descent
    for i in tqdm(range(img_dim - 2)):
        # Form the bond tensor and partitions
        bond_tensor, left, right = form_bond_tensor(mps, output_idx)
        # Initialize gradient to zero
        grad = tn.Node(np.zeros_like(bond_tensor, dtype=float))
        for j in range(batch_size):
            # Draw random sample feature and convert to tensor
            sample_idx = sample_idxs[j]
            # Form the tensor for the sample
            input_tensor_sample = create_input_tensor(Xs_tr[:, sample_idx])
            # Project input onto MPS (w/o the bond tensor)
            proj = project_input(input_tensor_sample, right, previous_projections[j], output_idx, mps[output_idx - 1])
            previous_projections[j] = proj
            # Update the gradient
            grad = grad + gradient(bond_tensor, proj, Ys_tr[:, sample_idx])

        # Update bond tensor with via gradient descent like scheme
        bond_tensor = bond_tensor - alpha * grad
        if i == 0:
            bond_tensor.add_axis_names(['r', 'in1', 'in2', 'out'])
        elif i == img_dim - 2:
            bond_tensor.add_axis_names(['l', 'in1', 'in2', 'out'])
        else:
            bond_tensor.add_axis_names(['l', 'r', 'in1', 'in2', 'out'])

        mps, output_idx = split_bond_and_update(mps, bond_tensor, bond_dim, output_idx)
        # print(np.max(bond_tensor.tensor))
    print('Training Error ' + str(model_error(mps, Xs_tr, Ys_tr)))


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    sweeping_mps_optimization(Xs_tr, Ys_tr, 0.01, 25)

