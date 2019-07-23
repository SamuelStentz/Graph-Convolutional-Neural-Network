import pandas as pd
import numpy as np
from pyrosetta import *
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.patches as mpatches
from inspect import signature

def plot_auc(path_csv):
	df = pd.read_csv(path_csv)
	logits = []
	for (n,p) in zip(df["Negative Class Logit"], df["Positive Class Logit"]):
		logits.append(n - p)
	to_bool = lambda x: int(x == "CLEAVED")
	labels = [to_bool(x) for x in df["Label"]]

	fpr, tpr, _ = roc_curve(labels, logits)
	roc_auc = auc(fpr, tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (AUC)')
	plt.legend(loc="lower right")
	plt.show()

def plot_aupr(path_csv):
	df = pd.read_csv(path_csv)
	logits = []
	for (n,p) in zip(df["Negative Class Logit"], df["Positive Class Logit"]):
		logits.append(n - p)
	to_bool = lambda x: int(x == "CLEAVED")
	labels = [to_bool(x) for x in df["Label"]]
	precision, recall, _ = precision_recall_curve(labels, logits)
	average_precision = average_precision_score(labels, logits)
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
	               if 'step' in signature(plt.fill_between).parameters
	               else {})
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
	plt.legend(loc="lower right")
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall Curve'.format(
          average_precision))
	red_patch = mpatches.Patch(color='b', label='Average Precision = %0.3f' % average_precision)
	plt.legend(handles=[red_patch], loc="lower left")
	plt.show()

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
	attentions = list(df.iloc[list(df["Sequence"]).index(substrate), 0:N])
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

	gsel = "select graph, "
	not_gsel = "select not_graph, "
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
			not_gsel += "(resi {} and chain {})+".format(ind,c)
		else:
			index = pose.pdb_info().pose2pdb(i)
			index = list(index)
			ind = int("".join(index[0:-2]))
			c = "".join(index[-2])
			gsel += "(resi {} and chain {})+".format(ind,c)
	gsel = gsel[0:-1]
	not_gsel = not_gsel[0:-1]
	for key in d:
		index = list(key)
		ind = int("".join(index[0:-2]))
		c = "".join(index[-2])
		output += "alter attention and n. CA and resi {} and chain {}, b={}\n".format(ind, c, d[key])
	output += """
spectrum b, white_red, attention and n. CA\n"""
	output += gsel + "\n"
	output += not_gsel + "\n"
	with open("attention_command.txt", "w") as fh:
		fh.write(output)

def plot_progression(path_csv):
	df = pd.read_csv(path_csv)
	ls = ["train_loss","train_acc","test_loss","test_acc"]
	vals = dict()
	for c in ls: vals[c] = list(df[c])
	vals["epoch"] = list(range(len(vals["train_loss"])))
	fig, ax1 = plt.subplots()
	lw = .5
	c1 = "r"
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Accuracy', color=c1)
	ax1.plot(vals["epoch"], vals["train_acc"],
		c1 + "--", label = "Training Accuracy", lw = lw)
	ax1.plot(vals["epoch"], vals["test_acc"],
		color=c1, label = "Testing Accuracy", lw = lw)
	ax1.tick_params(axis='y', labelcolor=c1)
	plt.legend(loc='lower left')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	c2 = "b"
	ax2.set_ylabel('Loss', color=c2)  # we already handled the x-label with ax1
	ax2.plot(vals["epoch"], vals["train_loss"],
		c2 + "--", label = "Training Loss", lw = lw)
	ax2.plot(vals["epoch"], vals["test_loss"],
		color=c2, label = "Testing Loss", lw = lw)
	ax2.tick_params(axis='y', labelcolor=c2)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.legend(loc='upper left')
	plt.savefig('accuracy.png', format='png', dpi=1000)

def attention_average(path_csv, path_classifications, path_protease):
	# go from label to list of sequences
	df = pd.read_csv(path_classifications)
	label_sequences = dict()
	for seq, label in zip(df["Sequence"],df["Label"]):
		if label in label_sequences: label_sequences[label].append(seq)
		else: label_sequences[label] = [seq]
	substrate = list(df["Sequence"])[0]
	# get pose indices in the graph
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
	bias_dimension = max(list(df["Bias"])) + 1
	N = int(list(df["N"])[0])
	if N != len(ls):
		raise ValueError("Non matching dimensions")
	# get all attention lists for all labels
	attentions = np.zeros(shape = (N))
	output = ""
	for label in label_sequences:
		for substrate in label_sequences[label]:
			si = list(df["Sequence"]).index(substrate)
			arr = df.iloc[si:(si + bias_dimension), 0:N]
			arr = np.sum(arr, 0)
			attentions += arr
		attentions = attentions / len(label_sequences[label])
		d = dict()
		j = 0
		for i in range(1,1+len(pose.sequence())):
			pdb_ind = pose.pdb_info().pose2pdb(i)
			if i in ls:
				d[pdb_ind] = attentions[j]
				j += 1
			else:
				d[pdb_ind] = 0
		pose.dump_pdb("attention_{}.pdb".format(label))

		gsel = "select graph, "
		not_gsel = "select not_graph, "
		output += """
load attention_{0}.pdb
alter attention_{0}, b = 0.0
show cartoon, attention_{0}
""".format(label)
		for i in range(1,1+len(pose.sequence())):
			if i not in ls:
				index = pose.pdb_info().pose2pdb(i)
				index = list(index)
				ind = int("".join(index[0:-2]))
				c = "".join(index[-2])
				output += "hide cartoon, attention_{} and n. CA and resi {} and chain {}\n".format(label, ind, c)
				not_gsel += "(resi {} and chain {})+".format(ind,c)
			else:
				index = pose.pdb_info().pose2pdb(i)
				index = list(index)
				ind = int("".join(index[0:-2]))
				c = "".join(index[-2])
				gsel += "(resi {} and chain {})+".format(ind,c)
		gsel = gsel[0:-1]
		not_gsel = not_gsel[0:-1]
		for key in d:
			index = list(key)
			ind = int("".join(index[0:-2]))
			c = "".join(index[-2])
			output += "alter attention_{} and n. CA and resi {} and chain {}, b={}\n".format(label, ind, c, d[key])
		output += """
spectrum b, white_red, attention_{} and n. CA\n""".format(label)
		output += gsel + "\n"
		output += not_gsel + "\n"
	with open("attention_average_command.txt", "w") as fh:
		fh.write(output)