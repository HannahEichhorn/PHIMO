"""
Script to create symbolic links to data sources (to enable easier handling
when training on different machines)

To run this script: insert your file paths in lines 13-16.
"""
import os.path
import glob
import numpy as np
from utils import *

# input and output folders:
folder_in_WS = ("path_to_converted_raw_files_server1/")
folder_out_WS = "path_to/PHIMO/ismrm-abstract/iml-dl/data/links_to_data/"
folder_in_cluster = ("path_to_converted_raw_files_server2")
folder_out_cluster = ("path_to/PHIMO/ismrm-abstract/iml-dl/data/links_to_data/")
nr_val_datasets = 2
nr_train_datasets = 6

# what to run:
run_WS = False
new_train_test_split = False
run_cluster = True


SL = SymbolicLinks()

if run_WS:
    if new_train_test_split:
        files = glob.glob(folder_in_WS+"**/**wip_t2s**_sg_fV4.mat")

        # exclude subject 36 due to problem with SENSE Ref scan:
        files = [f for f in files if "SQ-struct-36" not in f]

        np.random.shuffle(files)

        (train_files, val_files,
         test_files) = train_val_test_split(files, nr_val_datasets,
                                            nr_train_datasets)

        val_subjects, train_subjects, test_subjects = [], [], []
        for files, set, subjects in zip([val_files, train_files, test_files],
                                        ['val', 'train', 'test'],
                                        [val_subjects, train_subjects,
                                         test_subjects]):
            if os.path.exists("{}files_recon_{}.txt".format(folder_out_WS,
                                                            set)):
                print(
                    "ERROR: {}files_recon_{}.txt already exists.".format(
                        folder_out_WS, set)
                )
                continue

            for file in files:
                subjects.append(os.path.basename(os.path.dirname(file)))

            np.savetxt("{}files_recon_{}.txt".format(folder_out_WS, set),
                       subjects, fmt="%s")

    SL.loop_through_subjects(folder_out_WS, "WS", folder_in_WS)


if run_cluster:
    SL.loop_through_subjects(folder_out_cluster, "cluster",
                             folder_in_cluster)

print('Done creating symbolic links')
