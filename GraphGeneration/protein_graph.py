
from collections import defaultdict
from pyrosetta import *
from pyrosetta.toolbox import cleanATOM
import numpy as np
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.core.select.residue_selector import *
import sys
import math
import pandas as pd

####################################################################################################################

# BLOSUM62 matrix input
all_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
mat = r"blosum.txt"
if not os.path.isfile(mat):
     mat = r"./GraphGeneration/blosum.txt"
df = pd.read_csv(mat, sep = " ", index_col = 0)
df = df[[x for x in all_amino_acids]]
mat = df

#global variables, need to get mw of all these atoms
types = {'1HG2', ' NE2', ' NV ', ' CE1', '3HG2', ' SG ', ' C  ', ' OG1', '1HB ', ' CZ3', ' CD2',
             ' HH ', '2HA ', ' OD1', '2H  ', ' HZ ', '3HD1', ' HG ', '3HG1', ' ND2', '1HG ', ' NE1',
             ' NH2', ' OG ', '1HH1', '2HG ', ' HE ', ' OXT', '1HE2', ' CZ2', ' N  ', ' CG ', '1HD ',
             '3H  ', ' CZ ', '2HH1', ' CE3', '2HB ', '2HZ ', '1HE ', '2HG2', ' HZ2', ' CG2', '1HH2',
             ' O  ', '1HD1', '2HH2', ' CD ', ' HD1', ' HE3', ' ND1', '3HZ ', ' HA ', '1HD2', '1H  ',
             ' HE2', ' HE1', ' SD ', '2HD2', ' CE2', '1HG1', ' CE ', '2HD1', ' OH ', ' CD1', ' HD2',
             '3HE ', ' H  ', '2HE2', ' HH2', ' OE2', '2HD ', ' OE1', ' HZ3', '1HZ ', ' CA ', '3HB ',
             '2HG1', '1HA ', ' HG1', ' CB ', ' NH1', ' CH2', ' HB ', ' OD2', ' NE ', ' NZ ', '3HD2', ' CG1', '2HE '}
atom_mw = defaultdict(int)
for t in types:
    atom_mw[t] = 1

####################################################################################################################

# helper classes
def to_numpy(vec):
    """Takes a vector from pyrosetta and converts it into a numpy array"""
    return np.array([vec.x, vec.y, vec.z])

def com(residue, indices = None):
    """helper class to calculate COM (weighted) for specified atoms and a residue. Indices is optional,
    if not provided COM of entire residue used"""
    if indices == None:
        indices = np.arange(1, 1 + residue.natoms())
    if len(indices) == 0:
        return to_numpy(residue.xyz(1))
    
    weighted_vec = sum([(to_numpy(residue.xyz(x)) * atom_mw[residue.atom_name(x)]) for x in indices])
    tot_mass = sum([atom_mw[residue.atom_name(x)] for x in indices])
    return weighted_vec / tot_mass

####################################################################################################################

class protein_graph:
    """This class is going to hold a graphical representation of a protein. It can be generated from two sources:
    a pose object, or the file path of a pdb. Since we are attempting to model a substrate/protein complex,
    the substrate and interface indices are ROSETTA(starting at 1) based indexes. When specified, these indices are
    the indices that are used as nodes. When not supplied, all indices are used. It is assumed that:
    
    The substrate's indices are the last in the pdb/pose
    When supplied interface and substrate are non-zero length
    The intersection of substrate and interface indices is empty
    
    The indices of V are as follows:
    one hot vectors for amino acid type
    sinusoidal positional encoding (each increment represents a higher encoding)
    cosine similarity between sidechain and residue COM (actual weighted center of mass used)
    normalized (div by max) distance from Ca carbon to COM protein (rosetta's 'com')
    
    The indices of A are as follows:
    distance between residue i,j pair's Ca
    interaction energies for i,j pair"""
    
    def __init__ (self, substrate_indices = None,
                  interface_indices = None,
                  pdb_file_path = None,
                  pose = None,
                  params = {"amino_acids":True,
                            "blosum": True,
                            "sinusoidal_encoding":0,
                            "cosine_similarity":False,
                            "center_measure":False,
                            "interface_boolean":False,
                            "energy_terms":[],
                            "energy_edge_terms":[],
                            "distance":False,
                            "energy":True},
                 sfxn = None):
        
        # assure user provided a source
        if pdb_file_path == None and pose == None:
            raise PathNotDeclaredError("No pose or pdb path provided")
        
        # make pose from pdb
        if pdb_file_path != None:
            try:
                # clean the pdb file so that it conforms to rosetta standards
                cleanATOM(pdb_file_path)
                # read in pdb file 
                pose = pose_from_pdb(pdb_file_path)
            except:
                raise PathNotDeclaredError("Failed to generate pose, file path invalid or other issue")
        
        # if substrate or interface indices are given we will make vertice_arr specially tailored
        if substrate_indices != None and interface_indices != None:
            ls = substrate_indices + interface_indices
            vertice_arr = np.array(ls)
            interface_indices = np.array(interface_indices)
            substrate_indices = np.array(substrate_indices)
        elif substrate_indices != None:
            substrate_indices = np.array(substrate_indices)
            vertice_arr = substrate_indices
            interface_indices = np.array([])
        else:
            vertice_arr = np.arange(1, len(pose.sequence()) + 1)
            interface_indices = np.array([])
            substrate_indices = np.array([])
        
        # parse through the provided params dict to determine F       
        if params["amino_acids"]:
            num_amino = 20
        else:
            num_amino = 0
        if params["blosum"]:
            num_blosum = 20
        else:
            num_blosum = 0
        num_dim_sine = params["sinusoidal_encoding"]
        if params["cosine_similarity"]:
            cosine_sim = 1
        else:
            cosine_sim = 0
        if params["center_measure"]:
            center_measure = 1
        else:
            center_measure = 0
        energy_terms = len(params["energy_terms"])
        
        # make score function and energies objects, apply the score function
        if sfxn == None:
            sfxn = get_fa_scorefxn()
        sfxn(pose)
        energies = pose.energies()
        
        # set N (number of residues)
        N = len(vertice_arr)
        
        # set F (number of node features) the tallies are here just for my sanity
        F = sum([num_amino, num_dim_sine, cosine_sim, center_measure, energy_terms, num_blosum])
        if params["interface_boolean"]:
            F += 1
        
        # initialize V (Feature Tensor NxF)
        self.V = np.zeros(shape = (N, F))
        
        # get M
        M = 0
        if params["energy"]: M += 1
        if params["distance"]: M += 1
        M += len(params["energy_edge_terms"])
        
        
        # initialize A (Multiple Adj. Mat. NxNxM)
        self.A = np.zeros(shape = (N, N, M))
        
        counter_F = 0 # this is just for sanity, it increments to index F that is being populated
        counter_M = 0
#####################################################################################################################

        # populates the 20 one-hot vectors for amino acid type
        if params["amino_acids"]:
            all_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            seq = pose.sequence()        
            # use the native ordering to generate features
            for i in range(len(vertice_arr)):
                i_ind = vertice_arr[i]
                if i_ind != None:
                    res = seq[i_ind - 1]
                    j = all_amino_acids.find(res)
                    self.V[i][j] = 1
            counter_F += 20

#####################################################################################################################

        if params["blosum"]:
            all_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            seq = pose.sequence()        
            # use the native ordering to generate features
            for i in range(len(vertice_arr)):
                i_ind = vertice_arr[i]
                if i_ind != None:
                    aa = seq[i_ind - 1]
                    blosum_vec = df.loc[aa, :]
                    self.V[i, counter_F:(counter_F + 20)] = blosum_vec
            counter_F += 20
            
#####################################################################################################################
        
        # add residue's sinusoidal positional encoding information
        if num_dim_sine != 0:
            #print("encoding from {} to {}".format(len(all_amino_acids), len(all_amino_acids) + num_dim_sine - 1))
            if not substrate_indices.any() and not interface_indices.any():
                n_position = N
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in range(n_position)])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                self.V[0:n_position,counter_F:(counter_F + num_dim_sine)] = position_enc
            elif substrate_indices.any() and interface_indices.any():
                # add substrates
                n_position = len(substrate_indices)
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in range(n_position)])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                
                for i in range(len(substrate_indices)):
                    if substrate_indices[i] != None:
                        self.V[i, counter_F:(counter_F + num_dim_sine)] = position_enc[i, :]

                # add interface
                n_position = len(pose.sequence()) #- len(substrate_indices)
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in range(n_position)])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                for i in range(N - len(substrate_indices)):
                    if interface_indices[i] != None:
                        self.V[(len(substrate_indices) + i), counter_F:(counter_F + num_dim_sine)] = position_enc[i, :]
            else:
                # add substrates
                n_position = len(substrate_indices)
                position_enc = np.array([
                    [pos / np.power(10000, 2*i/num_dim_sine) for i in range(num_dim_sine)]
                    if pos != 0 else np.zeros(num_dim_sine) for pos in (substrate_indices - substrate_indices[0])])
                position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
                position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
                self.V[0:n_position,counter_F:(counter_F + num_dim_sine)] = position_enc
            counter_F += num_dim_sine

#####################################################################################################################

        # get all vector calculations to get cosine similarity for sidechain
        if params["cosine_similarity"]:
            ind_cosine = counter_F
            #print("cosine similarity at {}".format(ind_cosine))
            # protein's COM
            center_protein_xyz = to_numpy(pyrosetta.rosetta.core.pose.center_of_mass(pose, i, i))

            # iterate through all residues
            for i in range(len(vertice_arr)):
                if vertice_arr[i] != None:
                    i_ind = vertice_arr[i]
                    res = pose.residue(i_ind)

                    # alpha-Carbon to protein's COM vector
                    ca_com_xyz = to_numpy(res.xyz("CA"))

                    # alpha-Carbon to protein's COM vector
                    ca_protein = center_protein_xyz - ca_com_xyz

                    # all indices of side-chains
                    indices = []
                    for atom_index in range(1, res.natoms() + 1):
                        if not res.atom_is_backbone(atom_index):
                            indices.append(atom_index)

                    # alpha-Carbon to sidechain COM vector
                    center_sidechain_xyz = com(res, indices)
                    ca_sidechain = center_sidechain_xyz - ca_com_xyz

                    # add cosine similarity
                    divisor = np.linalg.norm(ca_sidechain) * np.linalg.norm(ca_protein)
                    if divisor != 0:
                        self.V[i, ind_cosine] = ca_sidechain.dot(ca_protein) / divisor
            counter_F += 1
                
        # get all vector calculations to get distance from protein COM to Ca position
        if params["center_measure"]:
            ind_com = counter_F
            #print("COM at {}".format(ind_com))
            # protein's COM
            center_protein_xyz = to_numpy(pyrosetta.rosetta.core.pose.center_of_mass(pose, i, i))
            # iterate through all residues
            for i in range(len(vertice_arr)):
                if vertice_arr[i] != None:
                    i_ind = vertice_arr[i]
                    res = pose.residue(i_ind)
                    # alpha-Carbon to protein's COM vector
                    ca_com_xyz = to_numpy(res.xyz("CA"))
                    # add distance from ca to com of protein
                    self.V[i, ind_com] = np.linalg.norm(ca_com_xyz - center_protein_xyz)
            # normalize across the com to be between 0 and 1 (max normalization)
            self.V[:, ind_com] = self.V[:, ind_com] / max(self.V[:, ind_com])
            counter_F += 1
            
#####################################################################################################################
            
        # add all energy terms
        
        #print("energy from {} to {}".format(num_amino + num_dim_sine + cosine_sim + center_measure,
        #                                    num_amino + num_dim_sine + cosine_sim + center_measure + energy_terms - 1))
        for counter, term in enumerate(params["energy_terms"], counter_F):
            for i in range(N):
                if vertice_arr[i] != None:
                    self.V[i, counter] = energies.residue_total_energies(vertice_arr[i])[term]
                if max(abs(self.V[:, counter])) != 0:
                    self.V[:, counter] = self.V[:, counter] / max(abs(self.V[:, counter]))
        counter_F += energy_terms
        
#####################################################################################################################

        # add to the end the interface/substrate boolean if interface is not None
        if params["interface_boolean"]:
            #print("interface one-hot vector {}".format(num_amino + num_dim_sine + \
            #                                       cosine_sim + center_measure + energy_terms))
            self.V[0:len(substrate_indices),counter_F] = np.array([1 for x in range(len(substrate_indices))])

#####################################################################################################################

        # fill in A with distances
        if params["distance"]:
            for i in range(len(vertice_arr)):
                for j in range(i, len(vertice_arr)):
                    if vertice_arr[i] != None and vertice_arr[j] != None:
                        ca_com1_xyz = pose.residue(vertice_arr[i]).xyz("CA")
                        ca_com2_xyz = pose.residue(vertice_arr[j]).xyz("CA")
                        dist = (ca_com1_xyz - ca_com2_xyz).norm()
                        self.A[i, j, counter_M] = dist
                        self.A[j, i, counter_M] = dist
            counter_M += 1
        if params["energy"] or len(params["energy_edge_terms"]) != 0:
            # add residue-residue interaction energies or terms or both!
            add = 1 if params["energy"] else 0
            for i in range(len(vertice_arr)):
                for j in range(i, len(vertice_arr)):
                    if vertice_arr[i] != None and vertice_arr[j] != None:
                        rsd1 = pose.residue(vertice_arr[i])
                        rsd2 = pose.residue(vertice_arr[j])
                        weights = sfxn.weights()
                        emap = EMapVector()
                        sfxn.eval_ci_2b(rsd1, rsd2, pose, emap)
                        paired_energy = emap.dot(weights)
                        if params["energy"]:
                            self.A[i, j, counter_M] = paired_energy
                            self.A[j, i, counter_M] = paired_energy
                        for counter, term in enumerate(params["energy_edge_terms"]):
                            self.A[i, j, counter_M + add + counter] = emap[term]
                            self.A[j, i, counter_M + add + counter] = emap[term]
            counter_M += add
            counter_M += len(params["energy_edge_terms"])

#####################################################################################################################

def index_substrate(pose):
    """Takes a pose and returns the indices of the substrate."""
    # get substrate with built in selector
    num_chains = pose.num_chains()
    chain_name = pyrosetta.rosetta.core.pose.get_chain_from_chain_id(num_chains, pose)
    sub_sel = ChainSelector(chain_name)
    v1 = sub_sel.apply(pose)

    substrate_indices = []
    for count,ele in enumerate(v1):
        if ele:
            substrate_indices.append(count + 1)
    return substrate_indices

def index_substrate_active_site(pose, index_p1 = 7, upstream_buffer = 7, downstream_buffer = 1):
    """This function takes the ROSETTA INDEX of the P1 residue for a substrate within its chain, a pose, and
    the number of upstream and downstream residues to model, and returns the indices of the substrate. If the
    buffer actually goes OOB of the substrate, a None type for that ind is instead returned for 0 pad modelling"""
    ind_sub = index_substrate(pose)
    ind_active = []
    for i in range(-upstream_buffer, downstream_buffer):
        index_interest = i + index_p1
        if index_interest < 0 or index_interest >= len(ind_sub):
            ind_active.append(None)
        else:
            ind_active.append(ind_sub[index_interest])
    return ind_active

def index_interface_k_nearest(pose, active_site, substrate_indices, k):
    """This function takes a pose and a number of interface/substrate to consider and returns interface indices.
    Interface is defined as the k closest residues. They are returned in the order of how close they are."""
    # get protease indices, done by selecting all indices that are not the substrate
    prot_indices = []
    for i in range(1, 1 + len(pose.sequence())):
        if i not in substrate_indices:
            prot_indices.append(i)
    
    # get min distances from all protease residues to substrate (using CA)
    arr = np.zeros(shape = (len(prot_indices), 2))
    for i in range(len(prot_indices)):
        ca_protease_xyz = pose.residue(prot_indices[i]).xyz("CA")
        # list of distances for this prot indice
        ls = np.zeros(len(active_site))
        for j in range(len(active_site)):
            if active_site[j] != None:
                ca_substrate_xyz = pose.residue(substrate_indices[j]).xyz("CA")
                vec = to_numpy(ca_protease_xyz - ca_substrate_xyz)
                ls[j] = sum([x*x for x in vec])
            else:
                ls[j] = sys.float_info.max
        # put distance^2 in second column, put indice in prot in first column
        arr[i, 0] = i
        arr[i, 1] = min(ls)
    # sort the array by minimum distance
    arr = arr[arr[:,1].argsort(kind='mergesort')]
    # get top N interface residues
    size_of_interface = k
    interface_indices = list(arr[:size_of_interface,0])
    interface_indices = [int(x) for x in interface_indices]
    return interface_indices

def index_interface_nearest_residuewise(pose, active_site, substrate_indices, k):
    """This function takes a pose and a number of interface_residues to consider and returns interface indices.
    Interface is defined as the k closest residues in the protease BY RESIDUE IN SUBSTRATE. They are in the order
    of substrates supplied i.e. substrate_indices = [1, 2] -> [1th ... 1kth, 2th ... 2kth]. IF the number is not
    divisible by len(substrate_indices) then a hard cutoff is used (definitely bad)"""
    
    total_substrates = k
    
    if len(substrate_indices) % k == 0:
        k = int(len(substrate_indices) / k)
    else:
        k = int(len(substrate_indices) / k) + 1
    
    # get protease indices, done by selecting all indices that are not the substrate
    prot_indices = []
    for i in range(1, 1 + len(pose.sequence())):
        if i not in substrate_indices:
            prot_indices.append(i)
    
    # get min distances by substrate residue and append to int_ind
    interface_indices = np.zeros(len(active_site) * k)
    arr = np.zeros(shape = (len(prot_indices), 2))
    interface_ls = []
    for j in range(len(active_site)):
        if active_site[j] == None:
            interface_indices[j*k:(j+1)*k] = [None for x in range(k)]
        else:
            ca_substrate_xyz = pose.residue(active_site[j]).xyz("CA")
            for i in range(len(prot_indices)):
                ca_protease_xyz = pose.residue(prot_indices[i]).xyz("CA")
                vec = to_numpy(ca_protease_xyz - ca_substrate_xyz)
                arr[i, 1] = sum([x*x for x in vec])
                arr[i, 0] = i
            arr = arr[arr[:,1].argsort(kind='mergesort')]
            interface_indices[j*k:(j+1)*k] = arr[:k, 0]
    for i in range(len(interface_indices)):
        if not math.isnan(interface_indices[i]):
            interface_ls.append(int(interface_indices[i]))
        else:
            interface_ls.append(None)
    
    return interface_ls[0:total_substrates]

def index_interface_8ang_original(pose,
                                    active_site,
                                    substrate_indices,
                                    k,
                                    crystal_struct = "./GraphGeneration/CrystalStructures/HCV.pdb"):
    """This function takes a pose and a number of interface/substrate to consider and returns interface indices. The
    value k and pose are not used..."""
    # load default pose as original
    print(os.getcwd())
    pose = pose_from_pdb(crystal_struct)
    
    # get protease indices, done by selecting all indices that are not the substrate
    prot_indices = []
    for i in range(1, 1 + len(pose.sequence())):
        if i not in substrate_indices:
            prot_indices.append(i)
    
    # get min distances from all protease residues to substrate (using CA)
    interface_indices = set()
    for i in range(len(prot_indices)):
        ca_protease_xyz = pose.residue(prot_indices[i]).xyz("CA")
        # list of distances for this prot indice
        ls = np.zeros(len(active_site))
        for j in range(len(active_site)):
            if active_site[j] != None:
                ca_substrate_xyz = pose.residue(substrate_indices[j]).xyz("CA")
                vec = to_numpy(ca_protease_xyz - ca_substrate_xyz)
                dist = (sum([x*x for x in vec]))**.5
                if dist <= 8:
                    interface_indices.add(prot_indices[i])
    interface_indices = list(interface_indices)
    interface_indices.sort()
    return interface_indices
