import os
import pandas as pd
import numpy as np

root = os.getcwd()
opt = os.path.join(root, "Optimization")
master_file = os.path.join(root, "master_file.csv")

# ensure optimization exists
if not os.path.isdir(opt):
    raise ValueError("Invalid Location for calling file, or Optimization doesn't exist for some reason. {} attempted".format(opt))

# makes a master_file if it doesn't already exist 
if not os.path.isfile(master_file):
    df = pd.DataFrame(columns =  ["Graph Description","Model Description","Validation Accuracy"])
    df.to_csv(master_file, index = False)

# get all tuples (graph, model) from master_file into a set so redundancy is avoided
tuple_set = set()
df_orig = pd.read_csv(master_file)
for i in range(len(df_orig.iloc[:,0])):
    graph = df_orig.iloc[i,0]
    model = df_orig.iloc[i,1]
    tuple_set.add((graph, model))

# adds all the new elements IF THEIR VALUES ARE UNIQUE
row_ls = []
for file in os.listdir(opt):
    path_file = os.path.join(opt, file)
    if os.path.isfile(path_file):
        # reading in file values
        h = open(path_file, "r")
        ls = h.readlines()
        ls = [l.strip() for l in ls]
        h.close()
        # get instance values
        validation = float(ls[0])
        model = ls[1]
        graph = ls[2]
        # if the parameters are unique it is added to master list
        if (graph, model) not in tuple_set:
            new_row = [graph, model, validation]
            row_ls.append(new_row)

# makes a DataFrame of only the new entities and populates it
df_new = pd.DataFrame(np.zeros(shape = (len(row_ls), 3)))

df_new.columns =  ["Graph Description","Model Description","Validation Accuracy"]
for i in range(len(row_ls)):
    row = row_ls[i]
    df_new.iloc[i,0] = row[0]
    df_new.iloc[i,1] = row[1]
    df_new.iloc[i,2] = row[2]

# concat for a final df
final = pd.concat([df_orig, df_new])

# sort by validation accuracy
final = final.sort_values(by=['Validation Accuracy'], ascending = False)

final.to_csv(master_file, index = False)