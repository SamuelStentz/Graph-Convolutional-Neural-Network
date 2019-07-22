import pandas as pd
import numpy as np
from pyrosetta import *
import os

def attention_command(path_csv, path_protease, substrate, bias):
	init()
	root = os.getcwd()
	os.chdir("../GraphGeneration")
	import pathing
	import protein_graph
	pose = pathing.get_pose(substrate, path_protease)
	print(pose)
	sub_ind = protein_graph.index_substrate(pose)
	index_p1 = substrate.index(".") + 1
	act_ind = protein_graph.index_substrate_active_site(pose,
		index_p1 = index_p1, upstream_buffer = 7, downstream_buffer = 1)
	os.chdir("..")
	int_ind = protein_graph.index_interface_8ang_original(pose,
		act_ind, sub_ind, k, protease = path_protease.split("/")[-1])
	os.chdir(root)
	ls = act_ind + int_ind
	
	df = pd.read_csv(path_csv)
	N = int(list(df["N"])[0])
	if N != len(ls):
		raise ValueError("Non matching dimensions")
	attentions = list(df.iloc[0:N, list(df["Sequence"]).index(substrate)])
	d = dict()
	j = 0
	for i in range(1,1+len(pose.sequence())):
		pdb_ind = pose.pdb_info().pose2pdb(i)
		if i in ls:
			d[pdb_ind] = attentions[j]
			j += 1
		else:
			d[pdb_ind] = 0
	pose.dump_pdb("attention.pdb")
	output = """
load attention.pdb
alter attention, b = 0.0
show cartoon, attention
"""
	for i in range(1,1+len(pose.sequence())):
		if i not in ls:
			index = pose.pdb_info().pose2pdb(i)
			index = list(index)
			ind = int("".join(index[0:-2]))
			c = "".join(index[-2])
			output += "hide cartoon, attention and n. CA and resi {} and chain {}\n".format(ind, c)
	for key in d:
		index = list(key)
		ind = int("".join(index[0:-2]))
		c = "".join(index[-2])
		output += "alter attention and n. CA and resi {} and chain {}, b={}\n".format(ind, c, d[key])
	output += """
spectrum b, white_red, attention and n. CA"""
	with open("attention_command", "w") as fh:
		fh.write(output)

