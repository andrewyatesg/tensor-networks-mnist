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
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


def create_mps_state(length, input_dim, bond_dim, output_dim, output_idx, min=-100, max=100):
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
    :param max: maximum value that node components can be randomly initialized to
    :param min: minimum value that node components can be randomly initialized to
    :return: list of tensornetwork nodes in the proper format of [1]
    """
    # Initialize the MPS with no left leg
    mps_lst = [tn.Node(np.random.uniform(min, max, (0, bond_dim, input_dim, 0)))]
    # Add each non-endpoint node to the MPS
    for i in range(length - 2):
        mps_lst.append(tn.Node(np.random.uniform(min, max, (bond_dim, bond_dim, input_dim, 0))))
    # Finally add the right-most node
    mps_lst.append(tn.Node(np.random.uniform(min, max, (bond_dim, 0, input_dim, 0))))
    # Replace the node at {output_idx} with a node that has an output leg
    if output_idx == length - 1:
        mps_lst[output_idx] = tn.Node(np.random.uniform(min, max, (bond_dim, 0, input_dim, output_dim)))
    elif output_idx == 0:
        mps_lst[output_idx] = tn.Node(np.random.uniform(min, max, (0, bond_dim, input_dim, output_dim)))
    else:
        mps_lst[output_idx] = tn.Node(np.random.uniform(min, max, (bond_dim, bond_dim, input_dim, output_dim)))
    return mps_lst


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
    if output_node.shape[0] == 0:
        # {output_node} is the leftmost node
        # select node on right
        node2_idx = output_idx + 1
    elif output_node.shape[1] == 0:
        # {output_node} is the rightmost node
        # select node on left
        node2_idx = output_idx - 1
    else:
        # Randomly select a left or right node
        node2_idx = output_idx + np.random.choice([-1, 1])

    # Partitions the MPS around the two selected nodes
    left = np.min(output_idx, node2_idx)
    right = np.max(output_idx, node2_idx) + 1
    left_part = mps[:left]
    right_part = mps[right:]
    # Form the bond tensor
    node2 = mps[node2_idx]
    if output_idx < node2_idx:
        # Connect right leg of {output_node} with left leg of {node2}
        output_node[1] ^ node2[0]
        bond_tensor = tn.contractors.greedy([output_node, node2])
        return bond_tensor, left_part, right_part
    else:
        # Connect left leg of {output_node} with right leg of {node2}
        node2[1] ^ output_node[0]
        bond_tensor = tn.contractors.greedy([output_node, node2])
        return bond_tensor, left_part, right_part



def project_input(feature_tensor: tensornetwork.Node, mps_lst, j):
    # Replicate the tensor representing the features of the image
    mps_lst = tn.replicate_nodes(mps_lst)
    feature_tensor = tn.replicate_nodes(feature_tensor)
    for i, node in enumerate(feature_tensor):
        # Connect the ith feature node to the input axis on the ith MPS tensor
        if i != j and i != j + 1:
            node[0] ^ mps_lst[i][2]
    mps_lst.remove(mps_lst[j])
    mps_lst.remove(mps_lst[j + 1])
    # Now connect edges between mps_list nodes
    for i in range(len(mps_lst) - 1):
        mps_lst[i][1] ^ mps_lst[i + 1][0]
    return tn.contractors.greedy(mps_lst + feature_tensor)


def sweeping_mps_optimization(Xs_tr, Ys_tr, alpha, bond_dim, num_epochs):
    """
    Run algorithm depicted in FIG. 6 of [1].
    :param bond_dim: bond dimension
    :param Xs_tr: training set images
    :param Ys_tr: training set one-hot encodings of image labels
    :param alpha: step size
    :return: trained MPS
    """
    (img_dim, num_ex) = Xs_tr.shape
    num_labels = Ys_tr.shape[0]
    output_idx = np.random.randint(img_dim)
    # Tensor train choo choo
    mps = create_mps_state(img_dim, 2, bond_dim, num_labels, output_idx)

    project_input(Xs_tr[:, 0], mps_lst, 0)

    # Now that we have constructed the MPS, let's optimize its parameters
    # using gradient descent
    label_pos = 0
    for i in tqdm(range(num_epochs)):
        # First, form the bond tensor
        # Remember, nodes and edges are mutable
        node1 = mps_lst[label_pos]
        node2 = mps_lst[label_pos + 1]
        edge = node1[1] ^ node2[0]
        bond = tn.contract(edge)



if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # print(Xs_tr.shape)
    # print(Ys_tr.shape)
    # print(Xs_te.shape)
    # print(Ys_te.shape)
    sweeping_mps_optimization(Xs_tr, Ys_tr, 0.1, 12, 1)

