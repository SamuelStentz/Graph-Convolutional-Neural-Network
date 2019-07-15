
import itertools

#path to protein
pr_path = "/projects/f_sdk94_1/EnzymeModelling/SilentFiles/HCV_A171T" 

# possible graphs
params = ["all_onehot_distance",  "all_onehot_energy", "all_onehot_energy_distance",  "all_onehot_energy_terms", "all_onehot_energy_terms_distance"]
ratios = [0]
selectors = ["8_ang"]

graph_options = [params, ratios, selectors]

graph_options_combos = list(itertools.product(*graph_options))
pipeline_command_template = "python3 pipeline.py -params {0} -si {1} -is {2} -db protease_{3}_selector_{2}_ratio_{1}_params_{0} -unsafe False -pr_path {3}\n"

with open("graph_generation.txt", "w") as fh:
    for combo in graph_options_combos:
        fh.write(pipeline_command_template.format(combo[0], combo[1], combo[2], pr_path))
        #print(pipeline_command_template.format(combo[0], combo[1], combo[2]))

# possible parameters for model
epochs = 2000
early_stopping = 100
hidden1 = [20, 15, 10, 5]
hidden2 = [20, 15, 10, 5]
dropout = [0, .1,.2,.3]
attention_dim = [4, 5, 6, 7, 8, 9, 10]
attention_bias = [1, 2]
model = ["gcn", "gcn_cheby"]
max_degree = [3]
accuracy_test = False
accuracy_validation = True

model_options = [hidden1, hidden2, dropout, attention_dim, attention_bias, model, max_degree]

model_options_combos = list(itertools.product(*model_options))
train_command_template = "python3 train.py -epochs {0} -early_stopping {1} -hidden1 {2} -hidden2 {3} -dropout {4} "\
                            "-attention_dim {5} -attention_bias {6} -model {7} -max_degree {8} -dataset {9} "\
                            "-save_test {10} -save_validation {11}"

with open("model_generation.txt", "w") as fh:
    for graph_option in graph_options_combos:
        for model_option in model_options_combos:
            # name of the graph from first slurm batch command
            graph_name = "protease_{3}_selector_{2}_ratio_{1}_params_{0}".format(graph_option[0], graph_option[1], graph_option[2], pr_path)
            train_command = train_command_template.format(epochs, early_stopping, model_option[0],
                                                         model_option[1], model_option[2], model_option[3],
                                                         model_option[4], model_option[5], model_option[6],
                                                         graph_name, accuracy_test, accuracy_validation)
            #print(train_command)
            fh.write(train_command + "\n")
