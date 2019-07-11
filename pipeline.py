import pandas as pd
import os
import GraphGeneration.protein_graph as protein_graph
import GraphGeneration.pathing as pathing
import GraphGeneration.utils
from pyrosetta import *
import argparse
import time
from pyrosetta.rosetta.core.scoring import *
import pickle as pkl


# instantiating a few things
init()
sfxn = get_fa_scorefxn() # this apparently takes 1.5 secs to load... good to preset
parser = argparse.ArgumentParser()
root = os.getcwd()
classifier_path = os.path.join(root, "ClassifierData")
data_path = os.path.join(root, "Data")
graph_path = os.path.join(root, "GraphGeneration")
dict_path = os.path.join(graph_path, "Dicts")

###################################################################################################################
# this area is reserved for parameters we will want to pass in as optional flags

parser.add_argument("-db", "--database_name", help="Database name")
parser.add_argument("-pr_path", "--protease_path", help="Path to silent pose directory for protease")
parser.add_argument("-class", "--classification_file", help="Name of txt for sequences to use, must be in folder")
parser.add_argument("-si", "--size_interface", help="|Interface|", type = int)
parser.add_argument("-is", "--interface_selector", help="Way Interface Is Selected, either k_nearest or residue_wise currently")
parser.add_argument("-unsafe", "--unsafe", help="Overwrite datasets", type = bool)
parser.add_argument("-params", "--params", help= "parameters dict for the graphs")

args = parser.parse_args()

db = args.database_name
pr_path = args.protease_path
class_file = args.classification_file
unsafe = args.unsafe
params = args.params
sel = args.interface_selector
size_interface = args.size_interface

if db == None:
    db = "testing_dataset"
if pr_path == None:
    pr_path = r"/scratch/ss3410/models"
if class_file == None:
    class_file = "experimental_binary_classifications.txt"
if unsafe == None:
    unsafe = False
if params == None:
    params = {"amino_acids":True,
                "blosum": True,
                "sinusoidal_encoding":0,
                "cosine_similarity":False,
                "center_measure":False,
                "interface_boolean":False,
                "energy_terms":[],
                "energy_edge_terms":[],
                "distance":False,
                "energy":True}
else:
    file = open(os.path.join(dict_path, params),'rb')
    params = pkl.load(file)
    
if sel == None:
    sel = protein_graph.index_interface_k_nearest
elif sel == "k_nearest":
    sel = protein_graph.index_interface_k_nearest
elif sel == "residue_wise":
    sel = protein_graph.index_interface_nearest_residuewise
elif sel == "8_ang":
    sel = protein_graph.index_interface_8ang_original
else:
    raise ValueError("Invalid selector for protease interface!")

if size_interface == None:
    k = 0
else:
    k = size_interface
    
####################################################################################################################
# Ensure database's name is unique!
if not unsafe:
    future_file = os.path.join(data_path, "ind.{}.y".format(db))
    if os.path.exists(future_file):
        raise ValueError("Non-unique database name used. Will cause data to be overwritten. \
Manually delete old files or rename with -db mydatabase tag")

# Read in labels and sequences
try:
    df = pd.read_csv(os.path.join(classifier_path, class_file), sep = "\t")
    labels = list(df["Result"])
    sequences = list(df["Sequence"])
except:
    raise ValueError("Path either invalid to classsifications or not properly formatted. \
Please check template experimental_binary_classifications.txt")

# Goes from a sequence to a graph representation.
def seq_pose(seq, pr_path):
    pose = pathing.get_pose(seq, pr_path)
    if type(pose) == type("string"):
        return pose
    substrate_ind = protein_graph.index_substrate(pose)
    index_p1 = seq.index(".") + 1
    active_ind = protein_graph.index_substrate_active_site(pose, index_p1 = index_p1,
                                                              upstream_buffer = 7, downstream_buffer = 1)
    interface_ind = sel(pose, active_ind, substrate_ind, k)
    g = protein_graph.protein_graph(pose = pose,
                                       substrate_indices = active_ind,
                                       interface_indices = interface_ind,
                                       sfxn = sfxn,
                                       params = params)
    return g


# get all graphs into a list
missed_sequences = []
error_sequences = []
seq_final = []
label_final = []
graphs = []
for i in range(len(sequences)):
    seq = sequences[i]
    graph = seq_pose(seq, pr_path)
    if graph == "Error: No Silent":
        missed_sequences.append(seq)
    elif graph == "Error: Invalid Silent":
        error_sequences.append(seq)
    else:
        seq_final.append(seq)
        graphs.append(graph)
        label_final.append(labels[i])

print("\n\n\n")
print("There were {} poses which loaded".format(len(graphs)))
print("There were {} poses missing due to silent files.".format(len(missed_sequences)))
print("There were {} poses which failed to be loaded.".format(len(error_sequences)))
print("\n\n\n")

# pickle the labels and graphs
GraphGeneration.utils.graph_list_pickle(graphs, label_final, seq_final, db, data_path)
