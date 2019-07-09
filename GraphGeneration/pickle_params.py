# this just pickles a param dict with a given name for the param dict DOESNT SUPPORT ENERGY TERMS (no clue is possible)

import argparse
import pickle as pkl
import os

parser = argparse.ArgumentParser()

parser.add_argument("-amino_acids", type=bool)
parser.add_argument("-sinusoidal_encoding", type=int)
parser.add_argument("-cosine_similarity", type=bool)
parser.add_argument("-center_measure", type=bool)
parser.add_argument("-energy_terms", type=list)
parser.add_argument("-edge_feature", type=str)
parser.add_argument("-interface_boolean", type=bool)
parser.add_argument("-name", type=str)
parser.add_argument("-blosum", type=bool)

args = parser.parse_args()

terms = [args.amino_acids, args.sinusoidal_encoding, args.cosine_similarity,
         args.center_measure, args.edge_feature, args.interface_boolean, args.blosum]
names = ["amino_acids", "sinusoidal_encoding", "cosine_similarity",
        "center_measure", "edge_feature", "interface_boolean", "blosum"]

for i in range(len(terms)):
    if terms[i] == None:
        raise ValueError("{} term not formatted properly".format(names[i]))

dict_loc = os.path.join(os.getcwd(), "Dicts")
file_loc = os.path.join(dict_loc, args.name)

if os.path.exists(file_loc):
    file = open(file_loc,'rb')
    params = pkl.load(file)
    raise ValueError("Params dict already exists! It looks like this:\n{}".format(params))
    file.close()


param_dict ={"amino_acids":args.amino_acids,
             "blosum":args.blosum,
                "sinusoidal_encoding":args.sinusoidal_encoding,
                "cosine_similarity":args.cosine_similarity,
                "center_measure":args.center_measure,
                "energy_terms":[],
                "edge_feature":args.edge_feature,
                "interface_boolean":args.interface_boolean}

pkl.dump(param_dict, open("./Dicts/{}".format(args.name), "wb"))
