import os.path
import pickle
import scipy

import numpy as np

from torch import nn

init = nn.init.xavier_uniform_


def make_dir(path):
    if os.path.isfile(path):
        pass
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(prefix='datasets/Mixed/processed/2fold'):
    def read_adjlist(filepath):
        with open(filepath, 'r') as file:
            return [line.strip() for line in file]

    def read_pickle(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)

    adjlist_paths = {
        'group0': ['0/0-1-0.adjlist', '0/0-2-0.adjlist'],
        'group1': ['1/1-0-1.adjlist', '1/1-2-1.adjlist', '1/1-1.adjlist']
    }

    idx_paths = {
        'group0': ['0/0-1-0_idx.pickle', '0/0-2-0_idx.pickle'],
        'group1': ['1/1-0-1_idx.pickle', '1/1-2-1_idx.pickle', '1/1-1_idx.pickle']
    }

    adj_lists = []
    for group in adjlist_paths.values():
        group_adjlists = []
        for filename in group:
            filepath = os.path.join(prefix, filename)
            group_adjlists.append(read_adjlist(filepath))
        adj_lists.append(group_adjlists)

    edge_indices = []
    for group in idx_paths.values():
        group_indices = []
        for filename in group:
            filepath = os.path.join(prefix, filename)
            group_indices.append(read_pickle(filepath))
        edge_indices.append(group_indices)

    adj_matrix = scipy.sparse.load_npz(os.path.join(prefix, 'adjM.npz'))
    node_types = np.load(os.path.join(prefix, 'node_types.npy'))
    pos_data = np.load(os.path.join(prefix, 'train_val_test_pos_circrna_disease.npz'))
    neg_data = np.load(os.path.join(prefix, 'train_val_test_neg_circrna_disease.npz'))

    data = {
        "adj": adj_lists,
        "edge": edge_indices,
        "adjM": adj_matrix,
        "type_mask": node_types,
        "pos_data": pos_data,
        "neg_data": neg_data
    }

    return data
