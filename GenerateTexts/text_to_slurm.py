# This lets you read a list of commands from a text file given in a flag and does all the slurming for you.
# By default they are run at /scratch/ss3410/GCNN. Additionally, you can specify where to put the .sh output file.
# By default they go down on file directory ex) /scratch/ss3410/GCNN/


"""
python text_to_slurm.py -txt model_generation.txt -tf -job_name M -path_operation /scratch/ss3410/Graph-Convolutional-Neural-Network/Model -path_sh /scratch/ss3410/Graph-Convolutional-Neural-Network/Commands -mem 12000 -batch 100 -time 3-00:00:00
python text_to_slurm.py -txt graph_generation.txt -job_name G -path_operation /scratch/ss3410/Graph-Convolutional-Neural-Network -path_sh /scratch/ss3410/Graph-Convolutional-Neural-Network/Commands -mem 8000 -batch 1 -time 10:00:00"""


import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-txt", type=str)
parser.add_argument("-job_name", type=str)
parser.add_argument("-path_operation", type=str)
parser.add_argument("-path_sh", type=str)
parser.add_argument("-mem", type=str)
parser.add_argument("-delay", type=int)
parser.add_argument("-batch", type=int)
parser.add_argument("-time", type=str)
parser.add_argument("-tf", action="store_true")

args = parser.parse_args()

filename = args.txt
job_name = args.job_name
path = args.path_operation
sh = args.path_sh
delay = args.delay
mem = args.mem
batch = args.batch
time = args.time
tf = args.tf

if time == None:
    time = "3-00:00:00"

if batch == None:
    batch == 1

if delay == None:
    delay = ""

if mem == None:
    mem = 12400
    
if path == None:
    path = "/scratch/ss3410/GCNN"

if job_name == None:
    raise ValueError("no name given")

if not os.path.exists(filename) and not os.path.exists(os.path.join(os.getcwd(), filename)):
    raise ValueError("file specified not found")

with open(filename) as f:
    lineList = f.readlines()
    
header ="""#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name {0}.{1}
#SBATCH --partition main
#SBATCH --ntasks {2}
#SBATCH --cpus-per-task 1
#SBATCH --mem {3}
#SBATCH --output {0}.log
#SBATCH --error {0}.err
#SBATCH --time {5}
#SBATCH --begin now

cd {4}

"""


lineList = [x.strip() for x in lineList]

if sh == None:
    sh = "../Commands"

i = 0
counter = 0

while i < len(lineList) + batch:
    command = r"{}/{}_{}.sh".format(sh, job_name, counter)
    header_specific = header.format(job_name, counter, 1, mem, path, time)
    if os.path.isfile(command):
        os.remove(command)
    f = open(command, "w")
    f.write(header_specific)
    if tf:
        f.write("\nmodule purge\nmodule load singularity/.2.5.1\n")
    for j in range(batch):
        if i + j < len(lineList):
            line = lineList[i+j]
            file_as_string = "\nsrun {}\n".format(line)
            f.write(file_as_string)
    f.close()
    i += batch
    counter += 1

with open(r"{}/{}.sh".format(sh, job_name), "w") as fh:
    fh.write(header.format(job_name, "all", 1, 4000, ".", "3:00:00"))
    fh.write("\nfor i in {0.." + str(counter) + "..1}; do sbatch " + job_name + "_$i.sh; done\n")
