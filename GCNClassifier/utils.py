# %load utils.py
import numpy as np
import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import scipy.sparse as sp
import tensorflow as tf
import os

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

    x_arr, y_arr, graph_arr, sequences, labelorder = tuple(objects)
    
    # this is the tensor of datapoints converted to a list of sparse matrices (BATCHxNxF)
    features = x_arr
    
    # make all the adjacency lists into nx graph objects (BATCHxGRAPHS)
    adj_ls = graph_arr
    
    # read in the test indices from the index file
    test_idx_reorder = parse_index_file("Data/ind.all.test.index")
    test_idx_range = np.sort(test_idx_reorder)
    idx_test = test_idx_range.tolist()
    
    os.chdir(cwd)
    
    # get training indexes and then split this group up into testing and validation
    idx_train = [y_ind for y_ind in range(y_arr.shape[0]) if y_ind not in idx_test]
    np.random.shuffle(idx_train)
    cutoff = int(6*len(idx_train)/7)
    idx_val = idx_train[cutoff:]
    idx_train = idx_train[:cutoff]
    idx_train, idx_val = np.sort(idx_train), np.sort(idx_val)
    idx_test = np.array(idx_test)
    
    # make logical indices (they are the size BATCH)
    train_mask = sample_mask(idx_train, y_arr.shape[0])
    val_mask = sample_mask(idx_val, y_arr.shape[0])
    test_mask = sample_mask(idx_test, y_arr.shape[0])
    
    
    
    print("|Training| {}, |Validation| {}, |Testing| {}".format(len(idx_train), len(idx_val), len(idx_test)))
    
    return adj_ls, features, y_arr, sequences, labelorder, train_mask, val_mask, test_mask


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
    try:
        largest_eigval, _ = eigsh(laplacian, 1, which='LM') # should still work
    except:
        largest_eigval, _ = eigsh(laplacian, 1, which='LM') # should still work, some wierd bug
        
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
