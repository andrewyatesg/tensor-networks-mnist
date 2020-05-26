"""
[1] Stoudenmire, Schwab. Supervised Learning with Tensor Networks. http://papers.nips.cc/paper/6211-supervised-learning-with-tensor-networks.
"""
import math

import numpy as np
import tensornetwork as tn
import os
import mnist
import matplotlib
import matplotlib.pyplot as plt
import pickle
import skimage.measure
from tensornetwork import FiniteMPS
import tensornetwork.visualization.graphviz
from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")


def feature_map(Xs, f1, f2):
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
        f1 = lambda x: np.cos(np.pi * x / 2)
        f2 = lambda x: np.sin(np.pi * x / 2)

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
        Xs_tr = feature_map(Xs_tr, f1, f2)
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
        Xs_te = feature_map(Xs_te, f1, f2)
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


def create_mps_state(length, input_dim, bond_dim, output_dim, output_idx, low=-100., high=100.):
    """
    Creates a MPS as a list of tensornetwork Node objects.

    For convenience, each node in the network will have shape (left, right, input, output)
    If a node doesn't have a left or right left (i.e. the endpoint nodes) then these
    dimensions will be set to zero
    :param length: number of nodes
    :param input_dim: input dimension of each node (usually 2)
    :param bond_dim: bond dimension of MPS, for now each will be the same
    :param output_dim: output dimension of a single node in the MPS (usually 10)
    :param output_idx: index of the node to place the output leg
    :param high: maximum value that node components can be randomly initialized to
    :param low: minimum value that node components can be randomly initialized to
    :return: list of tensornetwork nodes in the proper format of [1]
    """
    # Initialize the MPS with no left leg
    mps_lst = [tn.Node(np.random.uniform(low, high, (bond_dim, input_dim)), axis_names=['r', 'in'])]
    # Add each non-endpoint node to the MPS
    for i in range(length - 2):
        mps_lst.append(tn.Node(np.random.uniform(low, high, (bond_dim, bond_dim, input_dim)), axis_names=['l', 'r', 'in']))
    # Finally add the right-most node
    mps_lst.append(tn.Node(np.random.uniform(low, high, (bond_dim, input_dim)), axis_names=['l', 'in']))
    # Replace the node at {output_idx} with a node that has an output leg
    if output_idx == length - 1:
        mps_lst[output_idx] = tn.Node(np.random.uniform(low, high, (bond_dim, input_dim, output_dim)),
                                      axis_names=['l', 'in', 'out'])
    elif output_idx == 0:
        mps_lst[output_idx] = tn.Node(np.random.uniform(low, high, (bond_dim, input_dim, output_dim)),
                                      axis_names=['r', 'in', 'out'])
    else:
        mps_lst[output_idx] = tn.Node(np.random.uniform(low, high, (bond_dim, bond_dim, input_dim, output_dim)),
                                      axis_names=['l', 'r', 'in', 'out'])
    return mps_lst


def create_input_tensor(vector, dim):
    """
    Creates the tensor corresponding to the features {vector} of a single image.
    :param vector: vector of the image being converted to tensor
    :param dim: dimension of each input node leg
    :return: a list of nodes of the tensor
    """
    d = vector.shape[0]
    input_tn = []
    # Loop through features
    for i in range(int(d / dim)):
        feature = np.zeros((dim,))
        for j in range(dim):
            feature[j] = vector[i + j]
        input_tn.append(tn.Node(feature, axis_names=['in']))
    return input_tn


def form_bond_tensor(mps, output_idx):
    """
    Forms the bond tensor and partitions the MPS around the bond tensor, returning
    the bond tensor and both partitions.
    :param mps: the list of nodes corresponding to the MPS
    :param output_idx: the index of the node with the output leg
    :return: triple (bond, left, right) where {bond} is the bond tensor, {left} ({right})
    is the partition to the left (right) of the bond tensor
    """
    mps = tn.replicate_nodes(mps)
    # First, select two nodes where one has the output leg
    output_node = mps[output_idx]
    node2_idx = output_idx + 1
    # Partitions the MPS around the two selected nodes
    left = np.min([output_idx, node2_idx])
    right = np.max([output_idx, node2_idx]) + 1
    left_part = mps[:left]
    right_part = mps[right:]
    # Form the bond tensor
    node2 = mps[node2_idx]
    # if output_idx < node2_idx:
    # Connect right leg of {output_node} with left leg of {node2}
    output_node['r'] ^ node2['l']
    in1_edge = output_node.get_edge('in')
    in2_edge = node2.get_edge('in')
    out_edge = output_node.get_edge('out')
    if output_idx == 0:
        r_edge = node2.get_edge('r')
        edge_order = [r_edge, in1_edge, in2_edge, out_edge]
        axis_lbs = ['r', 'in1', 'in2', 'out']
    elif output_idx == len(mps) - 1:
        l_edge = output_node.get_edge('l')
        edge_order = [l_edge, in1_edge, in2_edge, out_edge]
        axis_lbs = ['l', 'in1', 'in2', 'out']
    else:
        l_edge = output_node.get_edge('l')
        r_edge = node2.get_edge('r')
        edge_order = [l_edge, r_edge, in1_edge, in2_edge, out_edge]
        axis_lbs = ['l', 'r', 'in1', 'in2', 'out']
    bond_tensor = tn.contractors.greedy([output_node, node2], output_edge_order=edge_order)
    bond_tensor.add_axis_names(axis_lbs)
    return bond_tensor, left_part, right_part, output_idx < node2_idx


def project_input(input_tensor, left_part, right_part):
    """
    TODO: reuse projections from last iteration of SGD
    Projects input tensor onto the MPS w/o the bond tensor.
    See FIG 6(c) for a drawing of this process.
    :param input_tensor: input image feature tensor
    :param left_part: tensors to the left of the bond tensor
    :param right_part: tensors to the right of the bond tensor
    :return: input tensor projected onto left and right partitions
    """
    input_tensor = tn.replicate_nodes(input_tensor)

    # Fully connect mps
    for i in range(len(left_part) - 1):
        e = left_part[i]['r'] ^ left_part[i + 1]['l']
    for i in range(len(right_part) - 1):
        e = right_part[i]['r'] ^ right_part[i + 1]['l']

    # Connect left partition to input nodes
    for i, node in enumerate(left_part):
        node['in'] ^ input_tensor[i]['in']

    # Connect right partition to input nodes
    offset = len(left_part) + 2
    for i, node in enumerate(right_part):
        node['in'] ^ input_tensor[offset + i]['in']

    in1_edge = input_tensor[len(left_part)]['in']
    in2_edge = input_tensor[len(left_part) + 1]['in']

    if len(left_part) > 0:
        l_edge = left_part[-1]['r']
    if len(right_part) > 0:
        r_edge = right_part[0]['l']
    if len(left_part) == 0:
        edge_order = [r_edge, in1_edge, in2_edge]
        ax_names = ['r', 'in1', 'in2']
    elif len(right_part) == 0:
        edge_order = [l_edge, in1_edge, in2_edge]
        ax_names = ['l', 'in1', 'in2']
    else:
        edge_order = [l_edge, r_edge, in1_edge, in2_edge]
        ax_names = ['l', 'r', 'in1', 'in2']

    proj = tn.contractors.auto(input_tensor + left_part + right_part, output_edge_order=edge_order)
    proj.add_axis_names(ax_names)
    return proj


def inner_product(mps, input_tensor):
    mps = tn.replicate_nodes(mps)
    # Fully connect mps
    for i in range(len(mps) - 1):
        mps[i]['r'] ^ mps[i + 1]['l']
    # Connect inputs to mps
    for i, node in enumerate(mps):
        node['in'] ^ input_tensor[i]['in']
    return mps + input_tensor


def gradient(mps, projection, input_tensor, output):
    mps_input_product = inner_product(mps, input_tensor)
    dangling = tn.get_all_dangling(mps_input_product)
    sub = tn.contractors.auto(mps_input_product, output_edge_order=dangling) - tn.Node(output)
    out_edge = sub.get_edge(0)
    in1_edge = projection['in1']
    in2_edge = projection['in2']
    edge_order = [in1_edge, in2_edge, out_edge]
    if projection.axis_names.count('r') > 0:
        r_edge = projection['r']
        edge_order.insert(0, r_edge)
    if projection.axis_names.count('l') > 0:
        l_edge = projection['l']
        edge_order.insert(0, l_edge)

    return tn.contractors.auto([sub, projection], output_edge_order=edge_order)


def split_bond_and_update(mps, bond, bond_dim, output_idx, prev_orientation_left):
    # if prev_orientation_left:
    left_edges = []
    right_edges = [bond['in2']]
    if bond.axis_names.count('l') > 0:
        left_edges.append(bond['l'])
    if bond.axis_names.count('r') > 0:
        right_edges.append(bond['r'])
    left_edges.append(bond['in1'])
    right_edges.append(bond['out'])
    left, right, sings = tn.split_node(bond, left_edges=left_edges,
                                       right_edges=right_edges,
                                       max_singular_values=bond_dim, edge_name='connect')
    left['connect'].disconnect()
    left.add_axis_names(['l', 'in', 'r'])
    right.add_axis_names(['l', 'in', 'r', 'out'])
    mps[output_idx] = left
    mps[output_idx + 1] = right
    output_idx = output_idx + 1
    return output_idx


def prediction(mps, img_feature_vector):
    return np.argmax(inner_product(mps, create_input_tensor(img_feature_vector, 2)))


def sweeping_mps_optimization(Xs_tr, Ys_tr, alpha, bond_dim, num_epochs):
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
    # Tensor train choo choo
    mps = create_mps_state(img_dim, 2, bond_dim, num_labels, output_idx, low=0, high=0.165)

    # Stochastic Gradient descent
    for i in tqdm(range(num_epochs)):
        # Draw random sample feature and convert to tensor
        sample_idx = np.random.randint(num_ex)
        input_tensor_sample = create_input_tensor(Xs_tr[:, sample_idx], 2)
        # Form the bond tensor and partitions
        bond_tensor, left, right, output_to_left = form_bond_tensor(mps, output_idx)
        proj = project_input(input_tensor_sample, left, right)
        grad = gradient(mps, proj, input_tensor_sample, Ys_tr[:, sample_idx])
        bond_tensor = bond_tensor - tn.Node(alpha) * grad
        if i == 0:
            bond_tensor.add_axis_names(['r', 'in1', 'in2', 'out'])
        elif i == img_dim - 1:
            bond_tensor.add_axis_names(['l', 'in1', 'in2', 'out'])
        else:
            bond_tensor.add_axis_names(['l', 'r', 'in1', 'in2', 'out'])
        output_idx = split_bond_and_update(mps, bond_tensor, bond_dim, output_idx, output_to_left)


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # print(Xs_tr.shape)
    # print(Ys_tr.shape)
    # print(Xs_te.shape)
    # print(Ys_te.shape)
    sweeping_mps_optimization(Xs_tr, Ys_tr, 0.1, 12, 100)

