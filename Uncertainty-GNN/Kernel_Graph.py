from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from utils import parse_index_file, sample_mask
import math
from metrics import masked_accuracy_numpy
from Load_npz import load_npz_data_ood

def get_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj.A, idx_train, y_train, train_mask


def get_data_ood(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    test_mask = np.array(1 - train_mask, dtype=bool)
    category = np.argmax(labels, axis=1)
    test_mask_all = np.array(test_mask)
    idx_train = list(idx_train)
    if dataset_str == 'cora':
        for i in range(labels.shape[0]):
            if category[i] > 3:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:4]

    if dataset_str == 'citeseer':
        for i in range(labels.shape[0]):
            if category[i] > 2:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:3]
    if dataset_str == 'pubmed':
        for i in range(labels.shape[0]):
            if category[i] > 1:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:2]
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    return adj.A, idx_train, y_train, train_mask


def kernel_distance(x, sigma=1.0):
    coffit = 1.0/ (2*sigma*np.sqrt(2 * math.pi))
    k_dis = np.exp(-np.square(x)/(2 * np.square(sigma)))
    return k_dis


def all_kde(sigma):
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[:]:
        adj, idx_train, y_train, train_mask = get_data(dataset)
        node_num = len(y_train)
        class_num = y_train.shape[1]
        G = nx.from_numpy_array(adj)
        alpha = np.zeros_like(y_train)

        for i in range(len(y_train)):
            graph_dis_i = np.zeros(node_num)
            for j in idx_train:
                try:
                    graph_dis_i_j = nx.shortest_path_length(G, i, j)
                except nx.NetworkXNoPath:
                    # print('No path between ', i, ' and ', j)
                    graph_dis_i_j = 1e10
                graph_dis_i[j] = graph_dis_i_j
            kernel_dis = kernel_distance(graph_dis_i, sigma=sigma)
            kernel_alpha_i = np.reshape(kernel_dis, [-1, 1]) * y_train
            alpha_i = np.sum(kernel_alpha_i, axis=0)
            alpha[i] = alpha_i
        train_mask = True
        acc = masked_accuracy_numpy(alpha, y_train, train_mask)
        print(acc)
        np.save('data/prior/all_prior_alpha_{}_sigma_{}.npy'.format(dataset, sigma), alpha + 1)
    print('Xujiang Zhao')


def all_kde_ood(sigma):
    datasets = ['cora', 'citeseer', 'pubmed']
    for dataset in datasets[0:3]:
        adj, idx_train, y_train, train_mask = get_data_ood(dataset)
        node_num = len(y_train)
        class_num = y_train.shape[1]
        G = nx.from_numpy_array(adj)
        alpha = np.zeros_like(y_train)

        for i in range(len(y_train)):
            graph_dis_i = np.zeros(node_num)
            for j in idx_train:
                try:
                    graph_dis_i_j = nx.shortest_path_length(G, i, j)
                except nx.NetworkXNoPath:
                    # print('No path between ', i, ' and ', j)
                    graph_dis_i_j = 1e10
                graph_dis_i[j] = graph_dis_i_j
            kernel_dis = kernel_distance(graph_dis_i, sigma=sigma)
            kernel_alpha_i = np.reshape(kernel_dis, [-1, 1]) * y_train
            alpha_i = np.sum(kernel_alpha_i, axis=0)
            alpha[i] = alpha_i
        train_mask = True
        acc = masked_accuracy_numpy(alpha, y_train, train_mask)
        print(acc)
        np.save('data/prior/all_prior_alpha_{}_sigma_{}_ood.npy'.format(dataset, sigma), alpha + 1)
    print('Xujiang Zhao')

def all_kde_ood_npy(sigma):
    datasets = ['amazon_electronics_photo', 'amazon_electronics_computers', 'ms_academic_phy']
    for dataset in datasets[2:3]:
        adj, idx_train, y_train, train_mask = load_npz_data_ood(dataset, 223)
        node_num = len(y_train)
        class_num = y_train.shape[1]
        G = nx.from_numpy_array(adj)
        alpha = np.zeros_like(y_train)

        for i in range(len(y_train)):
            graph_dis_i = np.zeros(node_num)
            for j in idx_train:
                try:
                    graph_dis_i_j = nx.shortest_path_length(G, i, j)
                except nx.NetworkXNoPath:
                    # print('No path between ', i, ' and ', j)
                    graph_dis_i_j = 1e10
                graph_dis_i[j] = graph_dis_i_j
            kernel_dis = kernel_distance(graph_dis_i, sigma=sigma)
            kernel_alpha_i = np.reshape(kernel_dis, [-1, 1]) * y_train
            alpha_i = np.sum(kernel_alpha_i, axis=0)
            alpha[i] = alpha_i
        # train_mask = True
        acc = masked_accuracy_numpy(alpha, y_train, train_mask)
        print(acc)
        np.save('data/prior/all_prior_alpha_{}_sigma_{}_ood.npy'.format(dataset, sigma), alpha + 1)
    print('Xujiang Zhao')

if __name__ == '__main__':
    all_kde(sigma=1)
    all_kde_ood_npy(sigma=1)
    all_kde_ood(sigma=1)

