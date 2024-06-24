import os.path
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import h5py
import logging
from typing import NamedTuple
import nibabel as nib
import glob
from medutils.mri import ifft2c, rss, fft2c
from scipy.ndimage import binary_erosion

def generate_masks(configuration, k_shape):
    """
    Generate masks based on configuration list.

    Parameters
    ----------
    configuration : list
        First entry defines which mask to generate. Further entries
        define parameters for this type of mask.
    k_shape : tuple
        Shape of k-space to generate mask for.

    Returns
    -------
    mask : np.ndarray
    """

    nc, ne, npe, nfe = k_shape

    if configuration[0] == "Random":
        # configuration: type, fraction of 0 lines, size of fully sampled center
        mask = np.ones(npe)
        total_zeros = int(npe * configuration[1])
        outside_indices = np.concatenate((
            np.arange(0, npe // 2 - configuration[2] // 2),
            np.arange(npe // 2 + configuration[2] // 2, npe)
        ))
        zero_indices = np.random.choice(
            outside_indices, total_zeros, replace=False
        )
        mask[zero_indices] = 0
    elif configuration[0] == "Gaussian":
        # configuration: type, fraction of 0 lines, size of fully sampled center
        mask = np.zeros(npe)
        indices = np.random.normal(
            npe // 2, npe / 3, npe // (1 - configuration[1])
        ).astype(int)
        indices = indices[indices < 92]
        indices = indices[indices >= 0]
        mask[indices] = 1
        mask[npe // 2 - configuration[2] // 2:
             npe // 2 + configuration[2] // 2] = 1
    elif configuration[0] == "VarDensBlocks":
        # configuration: type, fraction of 0 lines, max width of blocks
        std_scale = 5
        mask = np.ones(npe)
        nr_var0lines = int(npe * configuration[1])
        # variable density via sampling from gaussian:
        count = 0
        while count < nr_var0lines:
            indx = int(np.round(np.random.normal(
                loc=npe // 2, scale=(npe - 1) / std_scale
            )))
            if indx < 46:
                indx += 46
            elif indx >= 46:
                indx -= 46
            width = np.random.randint(0, configuration[2])
            if 0 <= indx < npe - width and mask[indx] == 1:
                for w in range(0, width + 1):
                    if mask[indx+w] == 1:
                        mask[indx + w] = 0
                        count += 1
                        if count == nr_var0lines:
                            break
    elif configuration[0] == "Continuous":
        # configuration: type, exclusion rate
        mask = np.random.beta(a=1-configuration[1], b=configuration[1],
                              size=npe)
    else:
        print("T2star_loader::ERROR: This random mask is not implemented.: "
              "{}".format(configuration))

    return (
        mask.reshape(1, 1, npe, 1)
        .repeat(nc, axis=0)
        .repeat(ne, axis=1)
        .repeat(nfe, axis=3)
    )


def process_raw_data(hf_file, slice):
    """Load raw data from h5 file and process to proper complex data."""

    raw_data = hf_file['out']['Data'][slice, :, 0, 0, :, 0]
    sens_maps = hf_file['out']['SENSE']['maps'][slice, :, 0, 0, :, 0]

    if (isinstance(raw_data, np.ndarray) and
            raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]):
        return (raw_data.view(np.complex64).astype(np.complex64),
                sens_maps.view(np.complex64).astype(np.complex64))
    else:
        print('Error in load_raw_mat: Unexpected data format: ',
              raw_data.dtype)


def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images."""

    tmp = hf_file['out']['Parameter']['YRange'][:]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    return -int((tmp[0, 0] + tmp[1, 0]) / 2)


def pad_sensitivity_maps(sens_maps, kspace_shape):
    """Pad coil sensitivity maps to have same shape as images."""

    pad_width = ((0, 0), (0, 0), (0, 0),
                 (int((kspace_shape[-1] - sens_maps.shape[-1]) / 2),
                  int((kspace_shape[-1] - sens_maps.shape[-1]) / 2))
                 )
    sens_maps = np.pad(sens_maps, pad_width, mode='constant')
    return np.nan_to_num(sens_maps / rss(sens_maps, 1))


def remove_readout_oversampling(data, nr_lines):
    """Remove readout oversampling."""

    return data[..., nr_lines:-nr_lines]


def compute_coil_combined_reconstructions(kspace, sens_maps,
                                          y_shift):
    """Compute coil combined reconstructions."""

    coil_imgs = ifft2c(kspace)
    coil_imgs = np.roll(coil_imgs, shift=y_shift, axis=-2)
    img_cc = np.sum(coil_imgs * np.conj(sens_maps), axis=1)
    img_cc = remove_readout_oversampling(img_cc,
                                         int(img_cc.shape[-1] / 4))

    # retrieve kspace with applied y-shift
    kspace_shift = fft2c(coil_imgs)

    return img_cc, kspace_shift


def load_mask_from_nii(filename):
    """Load mask from nii file."""

    return np.where(
        nib.load(filename).get_fdata()[10:-10][::-1, ::-1, :]
        < 0.5, 0, 1
    )


def load_brainmask_nii(filename, no_CSF=False):
    """Load brainmask in correct orientation and threshold it."""

    filename_bm = (filename.replace("/raw_data/", "/brain_masks/")
                   .replace(".mat", "_bm.nii"))
    brain_mask = load_mask_from_nii(filename_bm)
    if no_CSF:
        filename_bm_noCSF = os.readlink(filename_bm).replace("_CSF", "")
        brain_mask_noCSF = load_mask_from_nii(filename_bm_noCSF)
        return brain_mask, brain_mask_noCSF
    else:
        return brain_mask


def load_segmentations_nii(filename):
    """
        Load segmentations in correct orientation and threshold them.

        Parameters
        ----------
        filename : str
            Path to folder containing the raw h5 files.

        Returns
        ----------
        brain_mask : np.ndarray
            Corresponding brainmask (excluding CSF).
        gray_matter : np.ndarray
            Corresponding gray matter segmentation.
        white_matter : np.ndarray
            Corresponding white matter segmentation.
    """

    filename_bm = (os.readlink(filename.replace("/raw_data/", "/brain_masks/")
                               .replace(".mat", "_bm.nii")).replace("_CSF", ""))
    filename_gm = glob.glob(os.path.dirname(filename_bm)
                            +"/rc1sub**.nii")[0]
    filename_wm = glob.glob(os.path.dirname(filename_bm)
                            + "/rc2sub**.nii")[0]

    return (load_mask_from_nii(filename_bm),
            load_mask_from_nii(filename_gm),
            load_mask_from_nii(filename_wm))


def apply_undersampling_mask(kspace, sens_maps, mask_configuration=None,
                             mask=None):
    """Apply undersampling mask to kspace data."""

    if mask is None:
        mask = generate_masks(mask_configuration, kspace.shape)

    kspace_zf = kspace * mask
    coil_imgs_zf = ifft2c(kspace_zf)
    img_cc_zf = np.sum(coil_imgs_zf * np.conj(sens_maps), axis=1)

    return img_cc_zf, mask


def normalize_images(ref_img, images_to_be_normalized):
    """Normalize images by maximum absolute value of reference image."""

    norm = np.nanmax(abs(ref_img)) + 1e-9
    return [img / norm for img in images_to_be_normalized]


def equalize_coil_dimensions(data):
    """Equalize coil dimension (axis=1) of input data to 32."""

    data_32 = np.zeros((data.shape[0], 32, data.shape[2], data.shape[3]),
                       dtype=data.dtype)
    data_32[:, :data.shape[1]] = data
    return data_32


def generate_and_apply_masks(img_cc_fs, sens_maps, mask_configuration=None,
                             var_acc_rate="None"):
    """Generate and apply masks to input data according to mask_configuration
    and var_acc_rate."""

    if var_acc_rate != "None":
        # Generate a new random acceleration rate for each batch
        acc_rate = np.random.uniform(var_acc_rate[0],
                                     var_acc_rate[1])
        mask_configuration[1] = acc_rate

    # transform to k-space:
    coil_imgs = img_cc_fs[:, :, None] * sens_maps
    kspace_batch = fft2c(coil_imgs)

    masks_batch, img_cc_zfs_batch = [], []
    for kspace, smap in zip(kspace_batch, sens_maps):
        img_cc_zf, mask = apply_undersampling_mask(
            kspace, smap, mask_configuration=mask_configuration
        )
        masks_batch.append(mask)
        img_cc_zfs_batch.append(img_cc_zf)

    return (torch.as_tensor(np.array(masks_batch), dtype=torch.complex64),
            torch.as_tensor(np.array(img_cc_zfs_batch), dtype=torch.complex64))


def load_hr_qr_moco_data(filename, slice):
    """Load HR/QR-MoCo images (exported from Matlab mqBOLD pipeline)."""

    filename_bm = (filename.replace("/raw_data/", "/brain_masks/")
                   .replace(".mat", "_bm.nii"))
    filename_hrqrmoco = os.readlink(filename_bm).replace(
        "T1w_coreg/rcBrMsk_CSF.nii", "T2S/RDmoco/T2sw_RDMoCo.mat"
    )
    with h5py.File(filename_hrqrmoco, "r") as r:
        img_hrqrmoco = r["out"]["Data"][:, slice]
        if isinstance(img_hrqrmoco, np.ndarray) and img_hrqrmoco.dtype == [
            ('real', '<f8'), ('imag', '<f8')]:
            img_hrqrmoco = img_hrqrmoco.view('complex').astype(np.complex64)

    return np.swapaxes(img_hrqrmoco, 1, 2)[:, 10:-10][:, ::-1, ::-1]


class RawT2starDataset(Dataset):
    """
    Dataset for loading undersampled, raw T2* data in the reconstruction task.

    Parameters
    ----------
    path : str
        Path to folder containing the relevant h5 files.
    only_bm_slices : bool
        Whether slices with percentage of brainmask voxels < bm_thr*100% should
        be excluded or not.
    bm_thr : float
        Threshold for including / excluding slices based on percentage of
        brainmask voxels.
    normalize : str
        Whether to normalize the data with maximum absolute value of image
        ('abs_image').
    select_echo : bool or int
        Whether to select a specific echo.
    random_mask : bool or list
        How to generate the random mask. First entry defines which mask to
        generate. Further entries define parameters for this type of mask. The
        default is None, corresponding to generating a VarDensBlocks mask with
        parameters 2 and 3.
    overfit_one_sample : bool
        Whether only one sample should be loaded to test overfitting.
    load_whole_set : bool
        Whether to load the entire dataset during initialization, which is
        quite memory-demanding.
    """

    def __init__(self, path, only_bm_slices=False, bm_thr=0.1,
                 normalize="abs_image", select_echo=False,
                 random_mask=None, var_acc_rate="None",
                 overfit_one_sample=False, load_whole_set=True):
        super().__init__()

        self.path = path
        self.raw_samples = []
        self.only_bm_slices = only_bm_slices
        self.bm_thr = bm_thr
        self.normalize = normalize
        self.overfit_one_sample = overfit_one_sample
        self.select_echo = select_echo
        if not random_mask:
            self.random_mask = ["VarDensBlocks", 2, 3]
        else:
            self.random_mask = random_mask
        self.var_acc_rate = var_acc_rate
        self.load_whole_set = load_whole_set

        files = glob.glob(self.path + "raw_data/**")
        for filename in sorted(files):
            slices_ind = self._get_slice_indices(filename)
            new_samples = []

            for dataslice in slices_ind:
                if not load_whole_set:
                    new_samples.append(T2StarMetaSample(filename,
                                                        dataslice))
                else:
                    (kspace, sens_maps,
                     img_cc_fs, brain_mask) = self._load_h5_data(
                        filename, dataslice, load_brainmask=True
                    )
                    new_samples.append(T2StarRawDataSample(
                        filename, dataslice, kspace, sens_maps,
                        img_cc_fs, brain_mask[:, :, dataslice])
                    )
            self.raw_samples += new_samples

    def _get_slice_indices(self, filename):
        """Get indices of slices based on specified conditions."""

        if self.overfit_one_sample:
            slices_ind = [self.overfit_one_sample]
        elif self.only_bm_slices:
            brain_mask = load_brainmask_nii(filename)
            bm_summed = np.sum(brain_mask, axis=(0, 1))
            slices_ind = np.where(bm_summed
                                  / (brain_mask.shape[0] * brain_mask.shape[1])
                                  > self.bm_thr)[0]
        else:
            with h5py.File(filename, "r") as hf:
                slices_ind = np.arange(hf['out']['Data'].shape[0])
        return slices_ind

    def _load_h5_data(self, filename, dataslice, load_brainmask=False):
        """Load data from h5 file."""

        with h5py.File(filename, "r") as f:
            kspace, sens_maps = process_raw_data(f, dataslice)
            y_shift = get_yshift(f)
            sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)

        img_cc_fs, kspace_shift = compute_coil_combined_reconstructions(
            kspace, sens_maps, y_shift
        )
        sens_maps = remove_readout_oversampling(sens_maps,
                                                int(kspace.shape[-1] / 4))

        if load_brainmask:
            _, bm_ = load_brainmask_nii(filename, no_CSF=True)
            brain_mask = np.zeros_like(bm_)
            for i in range(0, bm_.shape[2]):
                brain_mask[:, :, i] = binary_erosion(binary_erosion(bm_[:, :, i]))

            return kspace_shift, sens_maps, img_cc_fs, brain_mask
        else:
            return kspace_shift, sens_maps, img_cc_fs

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - img_cc_zf : torch.Tensor
                Zero-filled image.
            - img_cc_fs : torch.Tensor
                Fully sampled image.
            - mask : torch.Tensor
                Mask for undersampling in the Fourier domain, with equalized
                coil dimension.
            - sens_maps : torch.Tensor
                Coil sensitivity maps, with equalized coil dimension.
            - brain_mask : torch.Tensor
                Corresponding brain mask.
            - filename : str
                Path to the file containing the data.
            - dataslice : int
                Index of the slice within the dataset.

        Notes
        -----
        If 'load_whole_set' is False, the method loads data from the file
        specified by the 'idx' index, processes it, and returns relevant
        tensors. If 'load_whole_set' is True, the method directly retrieves
        pre-loaded data from the 'raw_samples' list (which requires more
        memory).
        """

        if not self.load_whole_set:
            filename, dataslice = self.raw_samples[idx]
            kspace, sens_maps, img_cc_fs, brain_mask = self._load_h5_data(
                filename, dataslice, load_brainmask=True
            )
            brain_mask = brain_mask[:, :, dataslice]
        else:
            (filename, dataslice, kspace,
             sens_maps, img_cc_fs, brain_mask) = self.raw_samples[idx]

        if self.select_echo:
            img_cc_fs = np.array([img_cc_fs[self.select_echo]])

        if self.normalize == "abs_image":
            img_cc_fs = normalize_images(img_cc_fs, [img_cc_fs])[0]

        sens_maps = equalize_coil_dimensions(sens_maps)

        return (torch.as_tensor(img_cc_fs, dtype=torch.complex64),
                torch.as_tensor(sens_maps, dtype=torch.complex64),
                torch.as_tensor(brain_mask),
                str(filename),
                dataslice,
                self.random_mask,
                self.var_acc_rate
                )


class RawT2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.batch_size = args.get('batch_size', 8)
        self.only_brainmask_slices = args.get('only_brainmask_slices', False)
        self.bm_thr = args.get('bm_thr', 0.1)
        self.normalize = args.get('normalize', 'abs_image')
        self.overfit_one_sample = args.get('overfit_one_sample', False)
        self.select_echo = args.get('select_echo', False)
        self.random_mask = args.get('random_mask', ["VarDensBlocks", 2, 3])
        self.var_acc_rate = args.get('var_acc_rate', None)
        self.load_whole_set = args.get('load_whole_set', True)
        self.drop_last = False if self.overfit_one_sample else True
        self.num_workers = args.get('num_workers', 1)
        self.data_dir = args.get('data_dir', None)

        assert type(self.data_dir) is dict, ("DefaultDataset::init(): "
                                             "data_dir variable should be "
                                             "a dictionary")

    def train_dataloader(self):
        """Loads a batch of training data."""

        trainset = RawT2starDataset(self.data_dir['train'],
                                    only_bm_slices=self.only_brainmask_slices,
                                    bm_thr=self.bm_thr,
                                    normalize=self.normalize,
                                    select_echo=self.select_echo,
                                    random_mask=self.random_mask,
                                    var_acc_rate=self.var_acc_rate,
                                    overfit_one_sample=self.overfit_one_sample,
                                    load_whole_set=self.load_whole_set)
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=True
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data."""

        valset = RawT2starDataset(self.data_dir['val'],
                                  only_bm_slices=self.only_brainmask_slices,
                                  bm_thr=self.bm_thr,
                                  normalize=self.normalize,
                                  select_echo=self.select_echo,
                                  random_mask=self.random_mask,
                                  var_acc_rate=self.var_acc_rate,
                                  overfit_one_sample=self.overfit_one_sample,
                                  load_whole_set=self.load_whole_set)
        logging.info(f"Size of the validation dataset: {valset.__len__()}.")

        dataloader = DataLoader(
            valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True
        )
        return dataloader

    def test_dataloader(self):
        """Loads a batch of testing data.

        Note: the test dataset is not shuffled.
        """
        testset = RawT2starDataset(self.data_dir['test'],
                                   only_bm_slices=self.only_brainmask_slices,
                                   bm_thr=self.bm_thr,
                                   normalize=self.normalize,
                                   select_echo=self.select_echo,
                                   random_mask=self.random_mask,
                                   var_acc_rate=self.var_acc_rate,
                                   overfit_one_sample=self.overfit_one_sample,
                                   load_whole_set=self.load_whole_set)
        logging.info(f"Size of the test dataset: {testset.__len__()}.")

        dataloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class RawMotionT2starDataset(Dataset):
    """
    Dataset for loading undersampled, raw T2* data (motion-corrupted and
    motion-free) in the motion correction task

    Parameters
    ----------
    path : str
        Path to folder containing the relevant h5 files.
    only_bm_slices : bool
        Whether slices with percentage of brainmask voxels < bm_thr*100% should
        be excluded or not.
    bm_thr : float
        Threshold for including / excluding slices based on percentage of
        brainmask voxels.
    normalize : str
        Whether to normalize the data with maximum absolute value of image
        ('abs_image').
    select_echo : bool or int
        Whether to select a specific echo.
    random_mask : bool or list
        How to generate the random mask. First entry defines which mask to
        generate. Further entries define parameters for this type of mask. The
        default is None, corresponding to generating a VarDensBlocks mask with
        parameters 2 and 3.
    overfit_one_sample : bool
        Whether only one sample should be loaded to test overfitting.
    load_whole_set : bool
        Whether to load the entire dataset during initialization, which is
        quite memory-demanding.
    """

    def __init__(self, path, only_bm_slices=False, bm_thr=0.1,
                 normalize="abs_image", select_echo=False,
                 random_mask=None, var_acc_rate="None",
                 overfit_one_sample=False, load_whole_set=True,
                 select_one_scan=False, min_slice_nr=None):
        super().__init__()

        self.path = path
        self.raw_samples = []
        self.only_bm_slices = only_bm_slices
        self.bm_thr = bm_thr
        self.normalize = normalize
        self.overfit_one_sample = overfit_one_sample
        self.select_echo = select_echo
        self.random_mask = random_mask
        if not random_mask:
            self.random_mask = ["VarDensBlocks", 2, 3]
        self.var_acc_rate = var_acc_rate
        self.load_whole_set = load_whole_set
        self.select_one_scan = select_one_scan
        self.min_slice_nr = min_slice_nr

        files_move = sorted(glob.glob(self.path+"raw_data/**_move**"))
        files_gt = sorted(glob.glob(self.path+"raw_data/**sg_fV4**"))
        if len(files_move) != len(files_gt):
            logging.info("[RawMotionT2starDataset::ERROR: number of "
                         "motion-free  and motion-corrupted files in {} "
                         "does not match.".format(self.path))

        if self.select_one_scan is not False:
            files_move = [f for f in files_move if self.select_one_scan in f]
            files_gt = [f for f in files_gt if self.select_one_scan in f]

            if len(files_move) == 0:
                logging.info("[RawMotionT2starDataset::ERROR: no files match"
                             " the selected scan: {}".format(
                    self.select_one_scan)
                )
            if len(files_move) > 1:
                logging.info("[RawMotionT2starDataset::ERROR: more than one "
                             "file matches the selected scan: {}".format(
                    self.select_one_scan)
                )

        for filename_move, filename_gt in zip(files_move, files_gt):
            slices_ind = self.get_slice_indices(filename_move)

            new_samples = []

            for dataslice in slices_ind:
                if not load_whole_set:
                    new_samples.append(T2StarMotionMetaSample(
                        filename_move, filename_gt, dataslice))
                else:
                    (sens_maps, img_cc_fs,
                     img_cc_fs_gt, img_rdmoco,
                     brain_mask, brain_mask_noCSF) = self.load_h5_data(
                        filename_move, filename_gt, dataslice,
                        load_brainmask=True
                    )
                    new_samples.append(T2StarRawMotionDataSample(
                        filename_move, dataslice, sens_maps,
                        img_cc_fs, brain_mask[:, :, dataslice],
                        brain_mask_noCSF[:, :, dataslice], img_cc_fs_gt,
                        img_rdmoco)
                    )
            self.raw_samples += new_samples

    def get_slice_indices(self, filename):
        """Get indices of slices based on specified conditions."""

        if self.overfit_one_sample:
            slices_ind = np.array([self.overfit_one_sample])
        elif self.only_bm_slices:
            brain_mask = load_brainmask_nii(filename)
            bm_summed = np.sum(brain_mask, axis=(0, 1))
            slices_ind = np.where(bm_summed
                                  / (brain_mask.shape[0] * brain_mask.shape[1])
                                  > self.bm_thr)[0]
        else:
            with h5py.File(filename, "r") as hf:
                slices_ind = np.arange(hf['out']['Data'].shape[0])

        if self.select_one_scan:
            # ensure proper sorting of slices (interleaved acquisition)
            slices_ind = np.concatenate((slices_ind[slices_ind % 2 == 0],
                                         slices_ind[slices_ind % 2 != 0]))
        if self.min_slice_nr is not None:
            if isinstance(self.min_slice_nr, dict):
                if self.select_one_scan in self.min_slice_nr.keys():
                    ind = self.min_slice_nr[self.select_one_scan]
                else:
                    ind = self.min_slice_nr["default"]
                slices_ind = slices_ind[slices_ind >= ind]
            else:
                print("ERROR: expecting a dictionary for min_slice_nr.")
        return slices_ind

    def load_h5_data(self, filename_move, filename_gt, dataslice,
                     load_brainmask=False):
        """Load data from h5 file."""

        # Motion-corrupted data:
        with h5py.File(filename_move, "r") as f:
            kspace, sens_maps = process_raw_data(f, dataslice)
            y_shift = get_yshift(f)
            sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)

        img_cc_fs, kspace_shift = compute_coil_combined_reconstructions(
            kspace, sens_maps, y_shift
        )
        sens_maps = remove_readout_oversampling(sens_maps,
                                                int(kspace.shape[-1] / 4))

        # Motion-free ground truth data:
        with h5py.File(filename_gt, "r") as g:
            kspace_gt, sens_maps_gt = process_raw_data(g, dataslice)
            y_shift = get_yshift(g)
            sens_maps_gt = pad_sensitivity_maps(sens_maps_gt, kspace_gt.shape)

        img_cc_fs_gt, kspace_gt_shift = compute_coil_combined_reconstructions(
            kspace_gt, sens_maps_gt, y_shift
        )

        # HR/QR-MoCo data:
        img_hrqrmoco = load_hr_qr_moco_data(filename_move, dataslice)

        if self.normalize == "abs_image":
            img_cc_fs_gt = normalize_images(img_cc_fs_gt,
                                            [img_cc_fs_gt])[0]
            img_hrqrmoco = normalize_images(img_hrqrmoco,
                                            [img_hrqrmoco])[0]
            img_cc_fs = normalize_images(img_cc_fs, [img_cc_fs])[0]

        if load_brainmask:
            brain_mask, brain_mask_noCSF = load_brainmask_nii(filename_move,
                                                              no_CSF=True)
            return (sens_maps, img_cc_fs, img_cc_fs_gt, img_hrqrmoco,
                    brain_mask, brain_mask_noCSF)
        else:
            return sens_maps, img_cc_fs, img_cc_fs_gt, img_hrqrmoco

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - img_cc_fs : torch.Tensor
                Fully sampled image (with motion).
            - sens_maps : torch.Tensor
                Coil sensitivity maps, with equalized coil dimension.
            - img_cc_fs_gt : torch.Tensor
                Fully sampled ground truth image (without motion).
            - img_hrqrmoco : torch.Tensor
                HR/QR-MoCo image (from mqBOLD Matlab pipeline).
            - brain_mask : torch.Tensor
                Corresponding brain mask.
            - brain_mask_noCSF : torch.Tensor
                Corresponding brain mask without CSF.
            - filename : str
                Path to the file containing the data.
            - dataslice : int
                Index of the slice within the dataset.
            - random_mask : list
                List defining the configuration of the random mask to be
                generated.

        Notes
        -----
        If 'load_whole_set' is False, the method loads data from the file
        specified by the 'idx' index, processes it, and returns relevant
        tensors. If 'load_whole_set' is True, the method directly retrieves
        pre-loaded data from the 'raw_samples' list (which requires more
        memory).
        """

        if not self.load_whole_set:
            filename_move, filename_gt, dataslice = self.raw_samples[idx]
            (sens_maps, img_cc_fs,
             img_cc_fs_gt, img_hrqrmoco,
             brain_mask, brain_mask_noCSF) = self.load_h5_data(
                filename_move, filename_gt, dataslice,
                load_brainmask=True
            )
            brain_mask = brain_mask[:, :, dataslice]
            brain_mask_noCSF = brain_mask_noCSF[:, :, dataslice]
        else:
            (filename_move, dataslice,
             sens_maps, img_cc_fs,
             brain_mask, brain_mask_noCSF,
             img_cc_fs_gt, img_hrqrmoco) = self.raw_samples[idx]

        if self.select_echo is not False:
            img_cc_fs = np.array([img_cc_fs[self.select_echo]])
            img_cc_fs_gt = np.array([img_cc_fs_gt[self.select_echo]])
            img_hrqrmoco = np.array([img_hrqrmoco[self.select_echo]])

        sens_maps = equalize_coil_dimensions(sens_maps)

        return (
            torch.as_tensor(img_cc_fs, dtype=torch.complex64),
            torch.as_tensor(sens_maps, dtype=torch.complex64),
            torch.as_tensor(img_cc_fs_gt, dtype=torch.complex64),
            torch.as_tensor(img_hrqrmoco.copy(), dtype=torch.complex64),
            torch.as_tensor(brain_mask),
            torch.as_tensor(brain_mask_noCSF),
            str(filename_move),
            dataslice,
            self.random_mask,
            self.var_acc_rate
        )


class RawMotionT2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.batch_size = args.get('batch_size', 8)
        self.only_brainmask_slices = args.get('only_brainmask_slices', False)
        self.bm_thr = args.get('bm_thr', 0.1)
        self.normalize = args.get('normalize', 'abs_image')
        self.overfit_one_sample = args.get('overfit_one_sample', False)
        self.select_echo = args.get('select_echo', False)
        self.random_mask = args.get('random_mask', ["VarDensBlocks", 2, 3])
        self.var_acc_rate = args.get('var_acc_rate', None)
        self.load_whole_set = args.get('load_whole_set', True)
        self.drop_last = args.get('drop_last', False)
        self.num_workers = args.get('num_workers', 1)
        self.data_dir = args.get('data_dir', None)
        self.select_one_scan = args.get('select_one_scan', False)
        self.min_slice_nr = args.get('min_slice_nr', None)

        assert type(self.data_dir) is dict, ("DefaultDataset::init(): "
                                             "data_dir variable should be "
                                             "a dictionary")

    def train_dataloader(self):
        """Loads a batch of training data."""

        trainset = RawMotionT2starDataset(
            self.data_dir['train'],
            only_bm_slices=self.only_brainmask_slices,
            bm_thr=self.bm_thr,
            normalize=self.normalize,
            select_echo=self.select_echo,
            random_mask=self.random_mask,
            var_acc_rate=self.var_acc_rate,
            overfit_one_sample=self.overfit_one_sample,
            load_whole_set=self.load_whole_set,
            select_one_scan=self.select_one_scan,
            min_slice_nr=self.min_slice_nr
        )
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data."""

        if 'val' in self.data_dir:
           valset = RawMotionT2starDataset(
                self.data_dir['val'],
                only_bm_slices=self.only_brainmask_slices,
                bm_thr=self.bm_thr,
                normalize=self.normalize,
                select_echo=self.select_echo,
                random_mask=self.random_mask,
                var_acc_rate=self.var_acc_rate,
                overfit_one_sample=self.overfit_one_sample,
                load_whole_set=self.load_whole_set,
                select_one_scan=self.select_one_scan,
                min_slice_nr=self.min_slice_nr
            )
           logging.info(f"Size of the validation dataset: {valset.__len__()}.")

           dataloader = DataLoader(
                valset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=self.drop_last,
                pin_memory=True,
            )
           return dataloader
        else:
            logging.info("No validation dataset provided.")
            return None

    def test_dataloader(self):
        """Loads a batch of testing data.

        Note: the test dataset is not shuffled.
        """

        if 'val' or 'test' in self.data_dir:
            testset = RawMotionT2starDataset(
                self.data_dir['test'],
                only_bm_slices=self.only_brainmask_slices,
                bm_thr=self.bm_thr,
                normalize=self.normalize,
                select_echo=self.select_echo,
                random_mask=self.random_mask,
                var_acc_rate=self.var_acc_rate,
                overfit_one_sample=self.overfit_one_sample,
                load_whole_set=self.load_whole_set,
                select_one_scan=self.select_one_scan,
                min_slice_nr=self.min_slice_nr
            )
            logging.info(f"Size of the test dataset: {testset.__len__()}.")

            dataloader = DataLoader(
                testset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
            return dataloader
        else:
            logging.info("No test dataset provided.")
            return None


class RawMotionBootstrapSamples:
    def __init__(self, nr_bootstrap_samples, random_mask, var_acc_rate="None"):
        super().__init__()

        self.nr_bootstrap_samples = nr_bootstrap_samples
        self.random_mask = [random_mask[0][0]] + [r[0].numpy()
                                                  for r in random_mask[1:]]
        self.var_acc_rate = var_acc_rate

    def apply_random_masks(self, img_cc_fs, sens_maps):
        """Apply randomly generated masks to fully sampled images."""

        # transform to k-space:
        coil_imgs = img_cc_fs[:, :, None] * sens_maps
        kspace_batch = fft2c(coil_imgs)

        # create bootstrap samples:
        masks_batch, img_cc_zfs_batch = [], []
        for kspace, smap in zip(kspace_batch, sens_maps):
            masks, img_cc_zfs = [], []
            for i in range(self.nr_bootstrap_samples):
                if self.var_acc_rate != "None":
                    # Generate a new random acceleration rate for each batch
                    acc_rate = np.random.uniform(self.var_acc_rate[0],
                                                 self.var_acc_rate[1])
                    self.random_mask[1] = acc_rate

                img_cc_zf, mask = apply_undersampling_mask(
                    kspace, smap, mask_configuration=self.random_mask
                )
                masks.append(mask)
                img_cc_zfs.append(img_cc_zf)

            masks_batch.append(masks)
            img_cc_zfs_batch.append(img_cc_zfs)

        return (torch.as_tensor(np.array(masks_batch), dtype=torch.complex64),
                torch.as_tensor(np.array(img_cc_zfs_batch),
                                dtype=torch.complex64))

    def apply_aggr_masks(self, img_cc_fs, masks, sens_maps):
        """Apply aggregated masks to fully sampled images."""

        # transform to k-space:
        coil_imgs = img_cc_fs[:, :, None] * sens_maps
        kspace_batch = fft2c(coil_imgs)

        img_cc_zfs_batch = []
        for kspace, mask, smap in zip(kspace_batch, masks, sens_maps):
            img_cc_zf, _ = apply_undersampling_mask(kspace, smap, mask=mask)
            img_cc_zfs_batch.append(img_cc_zf)

        return torch.as_tensor(np.array(img_cc_zfs_batch),
                               dtype=torch.complex64)


class T2StarMetaSample(NamedTuple):
    """Generate named tuples consisting of filename and slice index."""

    fname: str
    slice_ind: int


class T2StarRawDataSample(NamedTuple):
    """
    Generate named tuples consisting of filename, slice index and
    loaded raw data.
    """

    fname: str
    slice_ind: int
    kspace: np.ndarray
    sens_maps: np.ndarray
    img_cc_fs: np.ndarray
    brain_mask: np.ndarray


class T2StarRawMotionDataSample(NamedTuple):
    """
    Generate named tuples consisting of filename, slice index and
    loaded raw data.
    """

    fname: str
    slice_ind: int
    sens_maps: np.ndarray
    img_cc_fs: np.ndarray
    brainmask: np.ndarray
    brainmask_noCSF: np.ndarray
    img_cc_fs_gt: np.ndarray
    img_rdmoco: np.ndarray


class T2StarMotionMetaSample(NamedTuple):
    """Generate named tuples consisting of filenames and slice index."""

    fname_move: str
    fname_gt: str
    slice_ind: int


def fft2c(x, shape=None, dim=(-2, -1)):
    """Centered Fourier transform."""

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=dim),
                                       axes=dim, norm='ortho', s=shape),
                           axes=dim)
