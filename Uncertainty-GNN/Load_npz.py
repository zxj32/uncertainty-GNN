# import os
# import numpy as np
# import scipy.sparse as sp
# from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize
import numpy as np
from npz_io import load_npz_to_sparse_graph
from npz_preprocess import to_binary_bag_of_words, remove_underrepresented_classes, \
    eliminate_self_loops, binarize_labels

def get_dataset(data_path, standardize):
    dataset_graph = load_npz_to_sparse_graph(data_path)

    if standardize:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    adj_matrix, attr_matrix, labels = dataset_graph.unpack()

    labels = binarize_labels(labels)
    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(attr_matrix):
        attr_matrix = to_binary_bag_of_words(attr_matrix)

    # some assertions that need to hold for all datasets
    # adj matrix needs to be symmetric
    # assert (adj_matrix != adj_matrix.T).nnz == 0
    # features need to be binary bag-of-word vectors
    # assert is_binary_bag_of_words(attr_matrix), f"Non-binary node_features entry!"

    return adj_matrix, attr_matrix, labels

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)
        train_mask = sample_mask(train_indices, labels.shape[0])

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)
        val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask

def get_train_val_test_split2(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)
        train_mask = sample_mask(train_indices, labels.shape[0])

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)
        val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, test_indices


def get_train_val_test_split_ood(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)
        train_mask = sample_mask(train_indices, labels.shape[0])

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)
        val_mask = sample_mask(val_indices, labels.shape[0])

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
        test_mask = sample_mask(test_indices, labels.shape[0])
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_mask = sample_mask(test_indices, labels.shape[0])

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, train_indices

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_npz_data(dataset_str, seed):  # load co-author, Amazon datasets.
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    random_state = np.random.RandomState(seed)
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_val_test_split(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_npz_data2(dataset_str, seed):  # load co-author, Amazon datasets.
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    random_state = np.random.RandomState(seed)
    y_train, y_val, y_test, train_mask, val_mask, test_mask, test_idx = get_train_val_test_split2(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, test_idx


def load_npz_data_ood(dataset_str, seed):  # load co-author, Amazon datasets.
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    random_state = np.random.RandomState(seed)
    y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train = get_train_val_test_split_ood(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])

    test_mask = np.array(1 - train_mask, dtype=bool)
    category = np.argmax(labels, axis=1)
    test_mask_all = np.array(test_mask)
    idx_train = list(idx_train)
    if dataset_str == 'amazon_electronics_photo':
        for i in range(labels.shape[0]):
            if category[i] > 3:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:4]

    if dataset_str == 'amazon_electronics_computers':
        for i in range(labels.shape[0]):
            if category[i] > 4:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:5]
    if dataset_str == 'ms_academic_phy':
        for i in range(labels.shape[0]):
            if category[i] > 2:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:3]
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    return adj.A, idx_train, y_train, train_mask

def load_npz_data_ood_train(dataset_str, seed):  # load co-author, Amazon datasets.
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    random_state = np.random.RandomState(seed)
    y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train = get_train_val_test_split_ood(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])

    test_mask = np.array(1 - train_mask, dtype=bool)
    category = np.argmax(labels, axis=1)
    test_mask_all = np.array(test_mask)
    idx_train = list(idx_train)
    if dataset_str == 'amazon_electronics_photo':
        for i in range(labels.shape[0]):
            if category[i] > 3:
                train_mask[i] = False
                test_mask_all[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:4]

    if dataset_str == 'amazon_electronics_computers':
        for i in range(labels.shape[0]):
            if category[i] > 4:
                train_mask[i] = False
                test_mask_all[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:5]
    if dataset_str == 'ms_academic_phy':
        for i in range(labels.shape[0]):
            if category[i] > 2:
                train_mask[i] = False
                test_mask_all[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:3]
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    return adj, features, labels, labels, labels, train_mask, test_mask_all, test_mask_all

def load_npz_data_ood_train2(dataset_str, seed):  # load co-author, Amazon datasets.
    adj, features, labels = get_dataset("data/{}.npz".format(dataset_str), True)
    random_state = np.random.RandomState(seed)
    y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train = get_train_val_test_split_ood(random_state, labels, train_size=20*labels.shape[1], val_size=30*labels.shape[1])

    test_mask = np.ones_like(train_mask, dtype=bool)
    category = np.argmax(labels, axis=1)
    idx_train = list(idx_train)
    if dataset_str == 'amazon_electronics_photo':
        for i in range(labels.shape[0]):
            if category[i] > 3:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:4]

    if dataset_str == 'amazon_electronics_computers':
        for i in range(labels.shape[0]):
            if category[i] > 4:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:5]
    if dataset_str == 'ms_academic_phy':
        for i in range(labels.shape[0]):
            if category[i] > 2:
                train_mask[i] = False
                if i in idx_train:
                    idx_train.remove(i)
            else:
                test_mask[i] = False
        labels = labels[:, 0:3]

    return adj, features, labels, labels, labels, train_mask, test_mask, idx_train

