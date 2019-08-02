# %load utils.py
import numpy as np
import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import scipy.sparse as sp
import tensorflow as tf
import os

seed = 123
np.random.seed(seed)

def test_inputs(features, support, labels):
    if np.isnan(features).any(): np.nan_to_num(features, copy = False)
    if np.isnan(support).any(): np.nan_to_num(support, copy = False)
    if np.isnan(labels).any(): np.nan_to_num(labels, copy = False)
    if np.isinf(features).any(): raise ValueError("Features contains inf")
    if np.isinf(support).any(): raise ValueError("Support contains inf")
    if np.isinf(labels).any(): raise ValueError("Labels contains inf")

        
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => arr of the feature vectors of the training instances as numpy.ndarray object;
    ind.dataset_str.y => arr of the one-hot labels of the labeled training instances as numpy.ndarray object (|label| = number of classes); 
    ind.dataset_str.graph => arr of adjacency matrices as numpy objects
    ind.dataset_str.test.index => index file for test values. To ensure we properly do ONE split for all possible hyperparameters
    it simply is in Data/ind.all.test.index. This is NOT regenerated

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    protease = dataset_str.replace("protease_","")
    protease = protease.split("_selector")[0]
    cwd = os.getcwd()
    os.chdir("..")
    names = ['x', 'y', 'graph', 'sequences', 'labelorder']
    objects = []
    for i in range(len(names)):
        with open("Data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    features, y_arr, adj_ls, sequences, labelorder = tuple(objects)
    os.chdir(cwd)
    proteases = [protease for x in sequences]
    
    # Split all datasets into testing, training, and validation. The split of this data is fixed for each dataset
    # because the numpy seed is fixed, currently the breakdown is train: 60, validation: 10, test: 30
    idx = [y_ind for y_ind in range(y_arr.shape[0])]
    np.random.shuffle(idx)
    cutoff_1 = int(6*len(idx)/10)
    cutoff_2 = int(7*len(idx)/10)
    idx_train = idx[:cutoff_1]
    idx_val = idx[cutoff_1:cutoff_2]
    idx_test = idx[cutoff_2:]
    idx_train, idx_val, idx_test = np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)
    
    # make logical indices (they are the size BATCH)
    train_mask = sample_mask(idx_train, y_arr.shape[0])
    val_mask = sample_mask(idx_val, y_arr.shape[0])
    test_mask = sample_mask(idx_test, y_arr.shape[0])

    return adj_ls, features, y_arr, sequences, proteases, labelorder, train_mask, val_mask, test_mask


def parse_many_datasets(datasets):
    """This method deals with many datasets being provided. The datasets MUST have the cardinality of node set."""
    datasets = datasets.strip()
    if datasets[0] != "[" and datasets[-1] != "]":
        return load_data(datasets)
    datasets = datasets.strip("[]")
    datasets = datasets.replace(" ", "")
    datasets = datasets.split(",")
    
    # initialize with initial or first dataset, then simply concatenate each new dataset onto existing structure
    adj_ls, features, y_arr, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(datasets[0])
    for dataset in datasets[1:]:
        if dataset != "":
            adj_ls_curr, features_curr, y_arr_curr, sequences_curr, proteases_curr, _, train_curr, val_curr, test_curr = load_data(dataset)
            adj_ls = np.concatenate((adj_ls, adj_ls_curr), axis = 0)
            features = np.concatenate((features, features_curr), axis = 0)
            y_arr = np.concatenate((y_arr, y_arr_curr), axis = 0)
            train_mask = np.concatenate((train_mask, train_curr), axis = 0)
            val_mask = np.concatenate((val_mask, val_curr), axis = 0)
            test_mask = np.concatenate((test_mask, test_curr), axis = 0)
            sequences = sequences + sequences_curr
            proteases = proteases + proteases_curr
    return adj_ls, features, y_arr, sequences, proteases, labelorder, train_mask, val_mask, test_mask
        

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    for i in range(features.shape[0]):
        feature_arr = features[i,:,:]
        rowsum = np.array(feature_arr.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features[i,:,:] = r_mat_inv.dot(feature_arr)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # added a shift to be non negative
    adj += np.amax(adj)
    # normalize
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.identity(adj.shape[0]))
    return adj_normalized


def construct_feed_dict(features, support, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    adj_normalized = normalize_adj(adj)
    laplacian = np.identity(adj.shape[0]) - adj_normalized
    
    # try to get eigenvalues 10 times
    for i in range(10):
        try:
            largest_eigval, _ = eigsh(laplacian, 1, which='LM') # should still work
            passed = True
        except:
            passed = False
        if passed:
            break
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - np.identity(adj.shape[0])
    t_k = list()
    t_k.append(np.identity(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k
