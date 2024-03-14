import numpy as np
import os
import subprocess
import glob


def train_val_test_split(files, nr_val_datasets, nr_train_datasets):
    """
    Custom dataset split based on exclusion criteria.

    Parameters
    ----------
    files : list of str
        List of file names.
    nr_val_datasets : int
        Number of datasets for validation.
    nr_train_datasets : int
        Number of datasets for training.

    Returns
    -------
    train_files, val_files, test_files : list of str
        Lists containing file names for training, validation, and testing,
        respectively.

    Notes
    -----
    - Sorts the subjects "SQ-struct-30", "SQ-struct-31", "SQ-struct-32"
     and "SQ-struct-42" into the train set, since they are acquired without
     motion data.
    """

    train_files = [f for f in files if
                   any(s in f for s in ["SQ-struct-30", "SQ-struct-31",
                                        "SQ-struct-32", "SQ-struct-42"])]
    files = [f for f in files if f not in train_files]
    val_files = files[:nr_val_datasets]
    ind_test = nr_val_datasets + nr_train_datasets - len(train_files)
    test_files = files[ind_test:]
    train_files.extend(f for f in files[nr_val_datasets:ind_test])

    return train_files, val_files, test_files


def create_dir(folder):
    """Create a directory if it does not exist."""

    if not os.path.exists(folder):
        os.makedirs(folder)
    return 0


class SymbolicLinks:
    def __init__(self):
        super(SymbolicLinks, self).__init__()


    def create_link(self, file_name, link_name):
        """Create a symbolic link from a source file to a destination link."""

        subprocess.run('ln -s ' + file_name + ' ' + link_name, shell=True)
        return 0


    def loop_through_subjects(self, folder_out, machine, folder_in):
        """Loops through subjects and creates symbolic links"""
        for set in ['val', 'train', 'test']:
            if os.path.exists("{}recon_{}_{}".format(folder_out, set,
                                                     machine)):
                print("ERROR: Directory {}recon_{}_{} already "
                      "exists.".format(folder_out, set, machine))
                continue

            subjects = np.loadtxt("{}files_recon_{}.txt".format(folder_out,
                                                                set),
                                  dtype=str)
            if subjects.size == 1:
                subjects = [subjects]

            for subject in subjects:
                create_dir("{}recon_{}_{}/raw_data/".format(folder_out,
                                                            set, machine))
                create_dir("{}recon_{}_{}/brain_masks/".format(folder_out,
                                                               set, machine))

                # raw data:
                file = glob.glob(
                    "{}{}/**wip_t2s**_sg_fV4.mat".format(folder_in, subject)
                )[0]
                link_name = "{}recon_{}_{}/raw_data/{}_{}".format(
                    folder_out, set, machine,
                    os.path.basename(os.path.dirname(file)),
                    os.path.basename(file)
                )
                self.create_link(file, link_name)

                # brain mask:
                file_bm = ("{}/output/sub-{}/task-still/qBOLD/T1w_coreg/"
                           "rcBrMsk_CSF.nii".format(
                    folder_in.replace("converted_raw", "bids_from_raw"),
                    subject)
                )
                link_name_bm = (link_name.replace(
                    "/raw_data/", "/brain_masks/").replace(
                    ".mat", "_bm.nii"))
                self.create_link(file_bm, link_name_bm)

        for set_motion, set in zip(
                ['val_motion', 'train_motion', 'test_motion'],
                ['val', 'train', 'test']
        ):
            if os.path.exists("{}recon_{}_{}".format(folder_out,
                                                     set_motion, machine)):
                print("ERROR: Directory {}recon_{}_{} already exists.".format(
                    folder_out, set_motion, machine)
                )
                continue

            subjects = np.loadtxt("{}files_recon_{}.txt".format(
                folder_out, set), dtype=str
            )
            if subjects.size == 1:
                subjects = [subjects]

            for subject in subjects:
                if subject not in ["SQ-struct-30", "SQ-struct-31",
                                   "SQ-struct-32", "SQ-struct-42"]:
                    create_dir("{}recon_{}_{}/raw_data/".format(
                        folder_out, set_motion, machine)
                    )
                    create_dir("{}recon_{}_{}/brain_masks/".format(
                        folder_out, set_motion, machine)
                    )

                    # raw data:
                    file = glob.glob(
                        "{}{}/**wip_t2s**_sg_f_moveV4.mat".format(
                            folder_in, subject)
                    )[0]
                    link_name = "{}recon_{}_{}/raw_data/{}_{}".format(
                        folder_out, set_motion, machine,
                        os.path.basename(os.path.dirname(file)),
                        os.path.basename(file)
                    )
                    self.create_link(file, link_name)

                    # brain mask:
                    file_bm = ("{}/output/sub-{}/task-move/qBOLD/T1w_coreg/"
                               "rcBrMsk_CSF.nii".format(
                        folder_in.replace("converted_raw", "bids_from_raw"),
                        subject)
                    )
                    link_name_bm = (link_name.replace(
                        "/raw_data/", "/brain_masks/").replace(
                        ".mat", "_bm.nii")
                    )
                    self.create_link(file_bm, link_name_bm)


                    # also the corresponding GT:
                    file = glob.glob(
                        "{}{}/**wip_t2s**_sg_fV4.mat".format(
                            folder_in, subject)
                    )[0]
                    link_name = "{}recon_{}_{}/raw_data/{}_{}".format(
                        folder_out, set_motion, machine,
                        os.path.basename(os.path.dirname(file)),
                        os.path.basename(file)
                    )
                    self.create_link(file, link_name)

                    # brain mask:
                    file_bm = ("{}/output/sub-{}/task-still/qBOLD/T1w_coreg/"
                               "rcBrMsk_CSF.nii".format(
                        folder_in.replace("converted_raw", "bids_from_raw"),
                        subject)
                    )
                    link_name_bm = link_name.replace(
                        "/raw_data/", "/brain_masks/").replace(
                        ".mat", "_bm.nii"
                    )
                    self.create_link(file_bm, link_name_bm)
