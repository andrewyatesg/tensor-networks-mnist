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
from mps import MPS
import sweeper
import feature_tensor

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
    nodes = feature_tensor.FeatureTensor(img_feature_vector).get_nodes()
    prod = mps_product(mps, nodes)
    return np.argmax(np.abs(tn.contractors.auto(prod, output_edge_order=tn.get_all_dangling(prod)).tensor))


def model_error(mps, Xs, Ys):
    num_mis_classified = 0
    size = int(Xs.shape[1] / 1000)
    for i in range(size):
        pred = prediction(mps, Xs[:, np.random.randint(Xs.shape[1])])
        num_mis_classified = num_mis_classified + int(pred != np.argmax(Ys[:, i]))
    return float(num_mis_classified) / float(size)


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

    model = MPS.random(bond_dim, 2, num_labels, img_dim)

    training_sample_size = 1000
    sample_idxs = np.random.randint(num_ex, size=training_sample_size)
    num_threads = 10
    Xs_tr_sample = Xs_tr[:, sample_idxs]
    Ys_tr_sample = Ys_tr[:, sample_idxs]
    sweep = sweeper.Sweeper(model, Xs_tr_sample, Ys_tr_sample)
    is_running = True
    while is_running:
        is_running = sweep.translate(alpha, bond_dim, num_threads)
        print('Training Error ' + str(model_error(model.tensors, Xs_tr, Ys_tr)))


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    sweeping_mps_optimization(Xs_tr, Ys_tr, 0.01, 25, 1)

