
import numpy as np
import pickle as pkl
import sys
import os
import scipy.sparse as sp
from pyrosetta.rosetta.core.scoring import *

def graph_list_pickle(graph_ls, label_ls, sequence_ls, dataset_name, destination_path):
    """Takes in a list of graphs and labels and pickles them in proper format. It also puts an index file in the directory"""
    
    # get population size, number of nodes, and number of node features
    population = len(graph_ls)
    N = graph_ls[0].V.shape[0]
    F = graph_ls[0].V.shape[1]
    M = graph_ls[0].A.shape[2]

    # find number of classifications possible
    s = set()
    for el in label_ls:
        s.add(el)
    s = list(s)
    num_classifiers = len(s)
    
    # generate feature matrices
    x = np.zeros(shape = (population, N, F))
    y = np.zeros(shape = (population, num_classifiers), dtype = np.int64)
    graph = np.zeros(shape = (population, N, N, M))
    
    # populate all elements
    for i in range(population):
        x[i,:,:] = graph_ls[i].V
        graph[i,:,:,:] = graph_ls[i].A
        one_hot_label_vec = np.zeros(num_classifiers)
        one_hot_label_vec[s.index(label_ls[i])] = 1
        y[i,:] = one_hot_label_vec
    
    # get indices of the test set
    idx = [x for x in range(population)]
    np.random.shuffle(idx)
    test_fraction = .3
    cutoff = int(len(idx) * test_fraction)
    test_index = idx[:cutoff]
    
    # pickle everything
    pkl.dump(x, open( os.path.join(destination_path,\
            "ind.{}.x".format(dataset_name)), "wb"))
    pkl.dump(y, open( os.path.join(destination_path,\
            "ind.{}.y".format(dataset_name)), "wb"))
    pkl.dump(graph, open( os.path.join(destination_path,\
            "ind.{}.graph".format(dataset_name)), "wb"))
    pkl.dump(sequence_ls, open( os.path.join(destination_path,\
            "ind.{}.sequences".format(dataset_name)), "wb"))
    pkl.dump(s, open( os.path.join(destination_path,\
            "ind.{}.labelorder".format(dataset_name)), "wb"))
    
    # if not already generated, generate test indices
    test_index_file = os.path.join(destination_path, "ind.all.test.index")
    if not os.path.exists(test_index_file):
        f = open(test_index_file, "w")
        for el in test_index:
            f.write(str(el) + "\n")
        f.close()

def pickle_many_param_dicts():
    energy_terms = [fa_intra_sol_xover4, fa_intra_rep, rama_prepro, omega, p_aa_pp, fa_dun, ref]
    energy_edge_terms = [pro_close, fa_atr, fa_rep, fa_sol, fa_elec, lk_ball_wtd]
    d = dict()
    d["0"] = {"amino_acids":True,
                    "distance":True}
    d["1"] = {"amino_acids":True,
                    "energy":True}
    d["2"] = {"amino_acids":True,
                    "energy":True,
                    "distance":True}
    d["3"] = {"amino_acids":True,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":energy_terms,
                    "distance":True,
                    "energy":True}
    d["4"] = {"amino_acids":True,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":energy_terms,
                    "energy_edge_terms":energy_edge_terms}
    d["5"] = {"amino_acids":True,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":energy_terms,
                    "energy_edge_terms":energy_edge_terms,
                    "interface_edge":True,
                    "hydrogen_bonding":True,
                    "covalent_edge":True}
    d["6"] = {"amino_acids":True,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":energy_terms,
                    "energy_edge_terms":energy_edge_terms,
                    "interface_edge":True,
                    "hydrogen_bonding":True,
                    "covalent_edge":True}
    d["7"] = {"amino_acids":True,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":energy_terms,
                    "energy_edge_terms":energy_edge_terms,
                    "distance":True,
                    "energy":True,
                    "interface_edge":True,
                    "hydrogen_bonding":True,
                    "covalent_edge":True}

    for key in d:
        pkl.dump(d[key], open("./Dicts/{}".format(key), "wb"))
