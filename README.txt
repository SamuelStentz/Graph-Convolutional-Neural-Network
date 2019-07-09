This repository is for running the GCNClassifier

Author: Samuel Stentz
Email: samuelstentz@gmail.com / samuelstentz@gatech.edu

Two Step Process for Running:

1) (root) python3 pipeline.py -db [name] -unsafe [True/False] -pr_path [silentfile_directory] -class [classification_file] ...
2) (root/GCNClassifier) python3 train.py -model [gcn/gcn_cheby] -dataset [name] ...


Loading Information:

For (1) a few files are required. Inside ClassifierData a text file of the following format is needed.
This will be the training data, and an example called experimental_binary_classifications.txt is provided.

----------------------
Sequence	Result
AAAAAC.ASHL	CLEAVED
AAAAAD.ASHL	UNCLEAVED
...
----------------------

Additionally, pr_path specifies the path to the directory containing silentfiles. The pr_path architechture has been adapted to allow for different sized substrates from different sequencing methods or samples/datasets. All substrates should be in the format UPSTREAM.DOWNSTREAM. Silent files follow the same format, except their variable regions are denoted with underscores. An example is below:

pr_path -->AAAA__C.ASHL -->AAAA__C.ASHL

Where AAAA__C.ASHL is a silentfile. This silentfile would contain any substrate that has that variability. for example:

AAAACVC.ASHL, AAAAWCC.ASHL, AAAAHHC.ASHL, AAAAAAC.ASHL are contained within AAAA__C.ASHL

Poses in Silent Files should be of the following format
1) The substrate should be the last chain by chain number in the pose.
2) Multiple unit poses should be supported, but have not been thouroughly tested. The following components of __init__ in protein graph
may not function as intended: sinusoidal_encoding, interface_boolean
3) Any relaxing etc. should already be done. (A method for generating the silent files was also created but not in this directory)
4) The name of the poses should be "substrate.UPSTREAM.DOWNSTREAM" (ex. substrate.AAAACVC.ASHL). The name of a pose becomes a silentfile's
"tag", which is how silentfiles are traversed to find the proper pose.

Pipeline Tags:
    db: Name of the database, or graph and classification info for the specifications provided. the GCNN/Data folder holds all outputs of pipeline
    pr_path: The location of the silent_files for the protease we are currently considering.
    class: Name of txt file holding classification data mentioned earlier. MUST BE IN GCNN/ClassifierData
    si: Number of nodes to model from the protease. This argument is ignored if is is the 8_ang
    is: What selector to use for the interface. Currently, options are knearest, residue_wise, and 8_ang
    unsafe: If True then information from an old database with the same name is overwritten if it exists.
    params: a dictionary that indicates which edge and node features to include in the graph. If not provided a default in protein_graph.py is used. The dictionary name is the name of a pickled python dictionary of the proper format. They should be in GCNN/GraphGeneration/Dicts


(2) Many parameters can be used for the model to tune hyperparameters/change graph convolution methodology.

Train Tags:
    dataset: name of dataset
    model: either gcn or gcn_cheby
    learning_rate: initial learning rate
    hidden1: number of features in graph convolution layer 1
    hidden2: number of features in graph convolution layer 2
    attention_bias: bias dimension of attention
    attention_dim: number of nodes in attention layer
    dropout: rate of weights to drop during training
    weight_decay: 5e-4 weight for L2 loss
    early_stopping: number of epochs before early stopping could happen (early stopping occurs when mean loss is > epoch loss)
    max_degree: maximum degree for chebyshev polynomial support matrices. Only matters when gcn_cheby is used
    save_validation: when True a text file with information on the validation results is saved at GCNN/GCNClassifier/Optimization
    save_test: when True testing and validation used for training model and results are saved. Should be used after tuning.

*** After training if you wish to consolidate all those model run results into a csv from validation, you can use GCNN/GCNClassifier/consolidate_results.py to make a master_file.csv holding all result information. I would write to a file but this could crash the parallel runs ***

CommandTextFiles:
CommandTextFiles is where a few scripts I wrote to generate all the commands for parallelization on a SLURM machine (tested on AMAREL) are located. The order for creating sh scripts are shown below. After running the two python scripts, there should be a massive number of .sh batch files in the GCNN/Commands folder. You can run these commands in groups by running the following command; "for i in {start..stop..1}; do sbatch jobname$i; done". This is how I have gotten parallel runs to work well with amarel. Nothing actually depends on these scripts.
1) python3 generate_command_text_files.py
2) python3 text_to_slurm.py -txt textfile_from_above.py -job_name name -path_operation path_to_execution -mem how_many_mb_to_request -delay time_before_execution -batch how_many_NONPARALLEL_commands_per_sbatch


Output Files:
The output files from running the model, if specified, are output in the output directory. One output option is to save predictions in a file of the same format as a classifier. Additionally, confidences from a prediction using softmax normalization for binary classifications are output to generate AUCs.


Analysis:
I added this directory to do analysis on the outputs from the model and for data visualization. analysis.py is a massive file of wrappers for general machine learning and data vis stuff. I can't promise that all of it is 100% correct because it's homegrown code by your's truly.
