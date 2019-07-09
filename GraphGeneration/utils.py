
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
    ls_d = []
    ls_d.append({"amino_acids":True,
                    "blosum": False,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":[],
                    "energy_edge_terms":[],
                    "distance":False,
                    "energy":True})
    ls_d.append({"amino_acids":True,
                    "blosum": False,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":[],
                    "energy_edge_terms":[fa_rep, fa_sol, fa_elec, lk_ball_wtd, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, fa_dun],
                    "distance":True,
                    "energy":False})
    ls_d.append({"amino_acids":True,
                    "blosum": False,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":[],
                    "energy_edge_terms":[fa_rep, fa_sol, fa_elec, lk_ball_wtd, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, fa_dun],
                    "distance":False,
                    "energy":False})
    ls_d.append({"amino_acids":True,
                    "blosum": False,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":[],
                    "energy_edge_terms":[fa_rep, fa_sol, fa_elec, lk_ball_wtd, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, fa_dun],
                    "distance":True,
                    "energy":False})
    ls_d.append({"amino_acids":True,
                    "blosum": False,
                    "sinusoidal_encoding":3,
                    "cosine_similarity":True,
                    "center_measure":True,
                    "interface_boolean":True,
                    "energy_terms":[],
                    "energy_edge_terms":[],
                    "distance":True,
                    "energy":True})

    ls_datasets = ["all_onehot_energy", "all_onehot_distance",
        "all_onehot_energy_terms", "all_onehot_energy_terms_distance", "all_onehot_energy_distance"]

    for i in range(len(ls_datasets)):
        pkl.dump(ls_d[i], open("./Dicts/{}".format(ls_datasets[i]), "wb"))
