import os
from pyrosetta import *
import random

#path_to_silent_files = r"/klab2/home/cl1205/silent_files"
#path_to_silent_files = r"/mnt/c/Users/Owner/Documents/Code/Research_Scripts/PyRosetta/GCNN"# local file for testing

def get_silent_file(sequence, path_to_silent_files):
    """This just returns an absolute path to the silent file (windows specific possibly) false if not found"""
    silent_file = None
    for silent in os.listdir(path_to_silent_files):
        correct = True
        for counter, char in enumerate(silent):
            if char != sequence[counter] and char != "_":
                correct = False
                break
        if correct:
            silent_file = silent
            break
    if silent_file == None:
        print("Silent dir for {} not found in {}!".format(sequence, path_to_silent_files))
        return False
    silent_dir = os.path.join(path_to_silent_files, silent_file)
    silent_file_path = os.path.join(silent_dir, silent_file)
    
    if os.path.exists(silent_file_path):
        return silent_file_path
    else:
        print("Silent file for {} not found {}!".format(sequence, silent_file_path))
        return False

def generate_dummy_silent(sequence, path_to_silent_files):
    silent_file = get_silent_file(sequence, path_to_silent_files)
    with open(silent_file) as f:
        lineList = f.readlines()
    tag_ending = "substrate.{}".format(sequence)
    found, done = (False, False)
    ind, start, end, last_score = (0,0,0,0)
    while ind < len(lineList) and not done:
        x = lineList[ind]
        if "SCORE" in x:
            last_score = ind
        if not found and "ANNOTATED_SEQUENCE: " in x and tag_ending in x:
            start = last_score
            found = True
        elif found and "ANNOTATED_SEQUENCE: " in x:
            end = last_score
            done = True
        ind += 1
    if not found:
        print("The requested sequence {} was not found in the silent file {} (Parsing Error)".format(sequence, silent_file))
        raise ValueError("The requested sequence {} was not found in the silent file {} (Parsing Error)".format(sequence, silent_file))
    if not done:
        end = len(lineList)
    
    filename = sequence + str(random.randint(10000, 100000))
    path_bin = os.path.join(os.getcwd(), "bin")
    filename = os.path.join(path_bin, filename)
    with open(filename, "w") as f:
        # add header
        for i in range(3):
            f.write(lineList[i])
        # add binary information
        for i in range(start, end):
            f.write(lineList[i])
    return filename
    
def get_pose(sequence, path_to_silent_files):
    if type(sequence) != type("string"):
        raise ValueError("Invalid entry for sequence according to current conventions")
    silent = get_silent_file(sequence, path_to_silent_files)
    if not silent:
        return "Error: No Silent"
    try:
        filename = generate_dummy_silent(sequence, path_to_silent_files)
        for pose in poses_from_silent(filename):
            ret = pose
        os.remove(filename)
        return ret
    except:
        os.remove(filename)
        return "Error: Invalid Silent"
