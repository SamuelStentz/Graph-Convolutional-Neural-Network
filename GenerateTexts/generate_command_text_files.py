# %load generate_command_text_files.py

import itertools

#protein
proteins = ["HCV", "HCV_A171T", "HCV_D183A", "HCV_R170K_A171T_D183A"]

#path to protein
pr_path = "/projects/f_sdk94_1/EnzymeModelling/SilentFiles/{}" 

#classifier txt
classifier = "{}.txt"

# possible graphs
params = [str(i) for i in range(8)]
ratios = [0]
selectors = ["k_nearest", "8_ang", "10_ang"]

graph_options = [params, ratios, selectors, proteins]

graph_options_combos = list(itertools.product(*graph_options))
pipeline_command_template = "python3 pipeline.py -params {0} -si {1} -is {2} -db protease_{5}_selector_{2}_ratio_{1}_params_{0} -pr_path {3} -class {4}\n"

with open("graph_generation.txt", "w") as fh:
    for combo in graph_options_combos:
        fh.write(pipeline_command_template.format(combo[0], combo[1], combo[2], pr_path.format(combo[3]), classifier.format(combo[3]), combo[3]))
        
# possible parameters for model
learning_rate = [.005, .01]
epochs = [2000]
early_stopping = [300]
graph_conv = [20, 10]
num_conv = [1, 2, 3]
connected = [20]
num_connected = [0, 1, 2]
dropout = [0,.2]
attention_dim = [10]
attention_bias = [1, 2, 3]
model = ["gcn_cheby"]
max_degree = [3]
accuracy_test = False
accuracy_validation = True

model_options = [learning_rate, epochs, graph_conv, num_conv, connected, num_connected,
                 dropout, attention_dim, attention_bias, model, max_degree, early_stopping]

model_options_combos = list(itertools.product(*model_options))
train_command_template = "python3 train.py -learning_rate {0} -epochs {1} -graph_conv_dimensions {2} -dropout {3} "\
                            "-attention_dim {4} -attention_bias {5} -model {6} -max_degree {7} -connected_dimensions {8} -dataset {9} "\
                            "-save_test {10} -save_validation {11} -early_stopping {12}"

graph_options = [params, ratios, selectors]
graph_options_combos = list(itertools.product(*graph_options))



# preliminary models
with open("model_generation.txt", "w") as fh:
    for graph_option in graph_options_combos:
        # all datasets to search through
        dataset_ls = []
        p = "["
        for pr in proteins:
            p+="protease_{3}_selector_{2}_ratio_{1}_params_{0},".format(graph_option[0], graph_option[1], graph_option[2], pr)
        p += "]"
        dataset_ls.append(p)
        dataset_ls = dataset_ls + ["protease_{3}_selector_{2}_ratio_{1}_params_{0}".format(graph_option[0], graph_option[1], graph_option[2], pr) for pr in proteins]
        for model_option in model_options_combos:
            for dataset in dataset_ls:
                gcd = "["
                for x in range(model_option[3]): gcd += str(model_option[2])+","
                gcd += "]"
                cd = "["
                for x in range(model_option[5]): cd += str(model_option[4])+","
                cd += "]"
                train_command = train_command_template.format(model_option[0], model_option[1], gcd, model_option[6],
                                                             model_option[7], model_option[8], model_option[9],
                                                             model_option[10], cd, dataset,
                                                             accuracy_test, accuracy_validation, model_option[11])
                fh.write(train_command + "\n")
