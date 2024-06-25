import merlinth
import numpy as np
import torch
import wandb
import os
import glob
import ants
import matplotlib.pyplot as plt
import h5py
import nibabel as nib
import warnings
from medutils.mri import ifft2c, rss
from scipy.stats import linregress
from scipy.ndimage import binary_erosion
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from optim.losses.ln_losses import L2
from optim.losses.image_losses import SSIM_Magn, PSNR_Magn


def create_dir(folder):
    """Create a directory if it does not exist."""

    if not os.path.exists(folder):
        os.makedirs(folder)
    return 0


def detach_torch(data):
    """Detach torch data and convert to numpy."""

    return (data.detach().cpu().numpy()
            if isinstance(data, torch.Tensor) else data)


def process_input_data(device, data):
    """Processes input data for training."""

    img_cc_zf = data[0].to(device)
    img_cc_fs = data[1].to(device)
    mask = data[2].to(device)
    sens_maps = data[3].to(device)
    A = merlinth.layers.mri.MulticoilForwardOp(
        center=True,
        channel_dim_defined=False
    )
    kspace_zf = A(img_cc_zf, mask, sens_maps)
    brain_mask = data[4].to(device)

    return img_cc_zf, kspace_zf, mask, sens_maps, img_cc_fs, brain_mask, A


def calculate_img_metrics(target, data, bm, metrics_to_be_calc,
                          include_brainmask=True):
    """ Calculate metrics for a given target array and data array."""

    metrics = {}
    methods_dict = {
        'MSE': L2,
        'SSIM': SSIM_Magn,
        'PSNR': PSNR_Magn
    }

    for descr in metrics_to_be_calc:
        for m in methods_dict:
            if descr.startswith(m):
                metric = methods_dict[m](include_brainmask)
                break
        if "magn" in descr:
            metrics[descr] = metric(
                torch.abs(target[None, :]), torch.abs(data[None, :]),
                bm[None, :]
            ).item()
        elif "phase" in descr:
            metrics[descr] = metric(
                torch.angle(target[None, :]), torch.angle(data[None, :]),
                bm[None, :]
            ).item()

    return metrics


def mean_abs_error(target, data, bm, include_brainmask=True):
    """Calculate Mean Absolute Error between two images."""

    if include_brainmask:
        return np.mean(abs(data - target)[bm[0] == 1])
    else:
        return np.mean(abs(data - target))


def calculate_t2star_metrics(target, data, bm, metrics_to_be_calc,
                             include_brainmask=True):
    """ Calculate metrics for a given target array and data array."""

    metrics = {}
    methods_dict = {
        'T2s_MAE': mean_abs_error
    }

    for descr in metrics_to_be_calc:
        for m in methods_dict:
            if descr.startswith(m):
                metric = methods_dict[m]
                break
        metrics[descr] = metric(
            np.abs(detach_torch(target[0])),
            np.abs(detach_torch(data[0])),
            detach_torch(bm), include_brainmask
        ).item()

    return metrics


def update_metrics_dict(new_dict, metrics_dict):
    """Update a dictionary of metrics with new values."""

    for key, value in new_dict.items():
        metrics_dict[key].append(value)
    return metrics_dict


def prepare_for_logging(data):
    """Detach the data and crop the third dimension if necessary"""

    data_prepared = detach_torch(data)

    return (data_prepared[:, :, 56:-56] if data_prepared.shape[2] > 112
            else data_prepared)


def log_images_to_wandb(prediction_example, ground_truth_example,
                        zero_filled_example, mask_example=None,
                        logging=None, logging_2=None, logging_0=None):
    """Log data to WandB for visualization"""

    data_operations = {
        "magn": np.abs,
        "phase": np.angle,
        "real": np.real,
        "imag": np.imag
    }

    for data_type in ["magn", "phase", "real", "imag"]:
        pred, gt, zf = map(data_operations[data_type],
                           [prediction_example, ground_truth_example,
                            zero_filled_example])

        # Max / Min values for normalization:
        max_value = np.nanmax(np.abs(np.array([pred, gt, zf])),
                              axis=(0, 2, 3))
        min_value = np.nanmin(np.abs(np.array([pred, gt, zf])),
                              axis=(0, 2, 3))

        # Track multi/single-echo data as video/image data:
        if prediction_example.shape[0] > 1:
            pred = convert2wandb(pred, max_value, min_value,
                                 media_type="video",
                                 caption="Reconstruction")
            gt = convert2wandb(gt, max_value, min_value,
                               media_type="video",
                               caption="Ground truth")
            zf = convert2wandb(zf, max_value, min_value,
                               media_type="video",
                               caption="Zero-filled")
            if mask_example is not None:
                mask = np.repeat(mask_example[None, None, :, None], 112, 3)
                mask = wandb.Video(np.swapaxes((mask*255).astype(np.uint8),
                                               -2, -1),
                                   fps=0.5, caption="Aggreg. mask")
        else:
            pred = convert2wandb(pred, max_value, min_value,
                                 media_type="image",
                                 caption="Reconstruction")
            gt = convert2wandb(gt, max_value, min_value,
                               media_type="image", caption="Ground truth")
            zf = convert2wandb(zf, max_value, min_value,
                               media_type="image", caption="Zero-filled")
            if mask_example is not None:
                mask = np.repeat(mask_example[:, None], 112, 1)
                mask = wandb.Image(np.swapaxes((mask*255).astype(np.uint8),
                                               -2, -1),
                                   caption="Aggreg. mask")
        if logging_0 is not None:
            log_key = "{}/MoCo_Examples_{}/{}/{}".format(logging_0,
                                                         logging,
                                                         data_type,
                                                         logging_2)
        elif logging_2 is not None:
            log_key = "Reconstruction_Examples_{}/{}/{}".format(logging,
                                                                data_type,
                                                                logging_2)
        else:
            log_key = "{}/Example_{}".format(logging, data_type)
        if mask_example is not None:
            wandb.log({log_key: [zf, pred, gt, mask]})
        else:
            wandb.log({log_key: [zf, pred, gt]})



def convert2wandb(data, abs_max_value, min_value, media_type="video",
                  caption=""):
    """
    Convert normalized data to a format suitable for logging in WandB.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    abs_max_value : np.ndarray
        Maximum absolute value for normalization.
    min_value : float
        Minimum value for normalization.
    media_type : str, optional
        Type of media ("video" or "image"). Default is "video".
    caption : str, optional
        Caption for the logged data. Default is "".

    Returns
    -------
    wandb.Video or wandb.Image
        Formatted data for WandB logging.
    """

    if media_type == "video":
        if np.amin(min_value) < 0:
            return wandb.Video(
                ((np.swapaxes(data[:, None], -2, -1)
                  / abs_max_value[:, None, None, None]+1) * 127.5
                 ).astype(np.uint8),
                fps=0.5, caption=caption
            )
        else:
            return wandb.Video(
                (np.swapaxes(data[:, None], -2, -1)
                 / abs_max_value[:, None, None, None] * 255
                 ).astype(np.uint8),
                fps=0.5, caption=caption
            )
    if media_type == "image":
        if np.amin(min_value) < 0:
            return wandb.Image(
                (np.swapaxes(data[0], -2, -1)
                 / abs_max_value + 1) * 127.5,
                caption=caption
            )
        else:
            return wandb.Image(
                np.swapaxes(data[0]-2, -1) / abs_max_value * 255,
                caption=caption
            )


def plot2wandb(map1, map2, map3, map4, titles, wandb_log_name):
    """
    Plot multiple maps and their differences and log the figure to WandB.

    Parameters
    ----------
    map1, map2, map3, map4, map4_reg1, map4_reg2, map4_reg3 : np.ndarray
        Arrays representing different maps.
    titles : list of str
        Titles for each map.
    wandb_log_name : str
        Name for WandB logging.

    Notes
    -----
    - Assumes that all input maps are 2D arrays.
    """

    map1, map2, map3, map4 = map(detach_torch, [map1, map2, map3, map4])

    fig = plt.figure(figsize=(6, 3), dpi=300)
    min_value = np.amin([map1, map2, map3, map4])
    max_value = np.amax([map1, map2, map3, map4])
    for nr, map_i, title in zip([0, 1, 2, 3],
                              [map1, map2, map3, map4],
                              titles):
        plt.subplot(2, 4, nr + 1)
        plt.imshow(map_i.T, vmin=min_value, vmax=max_value)
        plt.axis("off")
        plt.title(title, fontsize=8)
        if nr == 3:
            cax = plt.axes([0.91, 0.53, 0.025, 0.35])
            cbar = plt.colorbar(cax=cax)
            cbar.ax.tick_params(labelsize=8)
    min_value = np.amin([abs(map1 - map4),
                    abs(map2 - map4),
                    abs(map3 - map4)])
    max_value = 70
    for nr, map_i, title in zip([4, 5, 6],
                              [abs(map1 - map4),
                               abs(map2 - map4),
                               abs(map3 - map4)],
                              ["Diff", "Diff", "Diff"]):
        plt.subplot(2, 4, nr + 1)
        plt.imshow(map_i.T, vmin=min_value, vmax=max_value)
        plt.axis("off")
        plt.title(title, fontsize=8)
        if nr == 6:
            cax = plt.axes([0.71, 0.11, 0.025, 0.35])
            cbar = plt.colorbar(cax=cax)
            cbar.ax.tick_params(labelsize=8)
    wandb.log({wandb_log_name: fig})


def rigid_registration(fixed, moving, *images_to_move, numpy=False):
    """Perform rigid registration of moving to fixed image."""
    if not numpy:
        device = fixed.device
        # convert to numpy:
        fixed = detach_torch(fixed)
        moving = detach_torch(moving)
        images_to_move = [detach_torch(im) if not isinstance(im, np.ndarray)
                          else im for im in images_to_move]

    # calculate transform for fixed and moving
    fixed_image = ants.from_numpy(abs(fixed))
    moving_image = ants.from_numpy(abs(moving))

    # Perform registration
    registration_result = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform="Rigid"
    )

    # apply it to the other images
    images_reg = []
    for im in images_to_move:
        im_reg = np.zeros_like(im)
        for i in range(len(im)):
            ants_image = ants.from_numpy(im[i])

            # Apply the transformation to the image
            im_reg[i] = ants.apply_transforms(
                fixed_image,
                moving=ants_image,
                transformlist=registration_result['fwdtransforms']
            ).numpy()
        images_reg.append(im_reg)

    if not numpy:
        return [torch.tensor(im).to(device) for im in images_reg]
    else:
        return images_reg


def normalize_values(values):
    """Normalize values to sum up to 1."""

    min_value = np.amin(values, axis=1)[:, None]
    numerator = values - min_value
    denominator = np.sum(values - min_value, axis=1)[:, None]
    return numerator / denominator


class T2starFit:
    def __init__(self, data, bm, te1=5, d_te=5):
        super(T2starFit, self).__init__()

        self.data = abs(data)
        bm_ = bm[:, None]
        self.TE1 = te1
        self.dTE = d_te

        # apply binary erosion to brain mask to reduce size of brain mask
        # and reduce misalignment effects due to
        # large voxel size
        self.bm = np.zeros_like(bm_)
        for i in range(0, bm_.shape[0]):
            self.bm[i, 0] = binary_erosion(binary_erosion(bm_[i, 0]))


    def exponential_decay(self, t, A, T2star):
        """Exponential decay function for fitting."""

        return A * np.exp(-t / T2star)


    def t2star_linregr(self):
        """ Fit T2star and amplitude maps using linear regression.

        Parameters
        ----------
        data : array-like
            Input data to be fitted.
        bm : array-like
            Brainmask (0 / 1 for background / brain).
        TE1 : float, optional
            First echo time, default is 5ms.
        dTE : float, optional
            Echo distance, default is 5ms.

        Returns
        -------
        tuple
            Fitted T2star and amplitude maps.
        """

        TE = np.arange(self.TE1, self.data.shape[2] * self.TE1 + 1, self.dTE)
        slope, interc = (np.zeros(shape=self.data[:, :, 0].shape),
                         np.zeros(shape=self.data[:, :, 0].shape))

        fit_data = np.log(self.data+1e-9)

        for b in range(self.data.shape[0]):
            for n in range(self.data.shape[1]):
                for i in range(self.data.shape[3]):
                    for j in range(self.data.shape[4]):
                        if self.bm[b, 0, i, j]:
                            try:
                                result = linregress(TE, fit_data[b, n, :, i, j])
                                (slope[b, n, i, j],
                                 interc[b, n, i, j]) = (result.slope,
                                                        result.intercept)
                            except:
                                pass

        T2star, A = -1 / slope, np.exp(interc)
        T2star = np.clip(T2star, 0, 200)
        return T2star, A


    def weight_fit_error(self):
        """Calculate weighted fit error for each voxel."""

        T2star, A = self.t2star_linregr()

        TE = np.arange(self.TE1, self.data.shape[2] * self.TE1 + 1, self.dTE)
        TE = np.repeat(TE[np.newaxis, np.newaxis, :,
                       np.newaxis, np.newaxis],
                       self.data.shape[0], axis=0)
        TE = np.repeat(TE, self.data.shape[1], axis=1)
        TE = np.repeat(TE, self.data.shape[3], axis=3)
        TE = np.repeat(TE, self.data.shape[4], axis=4)

        T2star, A = T2star[:, :, None], A[:, :, None]
        data_fit = self.exponential_decay(TE, A, T2star)

        mean_data, mean_data_fit = (np.mean(self.data, axis=2)[:, :, None],
                                    np.mean(data_fit, axis=2)[:, :, None])
        emp_corr = (
                np.sum((self.data-mean_data)*(data_fit-mean_data_fit), axis=2)
                / np.sqrt(np.sum((self.data-mean_data)**2, axis=2)
                          * np.sum((data_fit-mean_data_fit)**2, axis=2))
        )

        emp_corr[np.repeat(self.bm, self.data.shape[1], axis=1) == 0] = np.nan
        metric_values = np.nanmean(emp_corr, axis=(-2, -1))

        return normalize_values(metric_values), metric_values


def load_predictions(checkpoint_path, run_id, task, varying_nr_samples=False,
                     max_nr_samples=20):
    """
    Load saved predictions (from DownstreamEvaluator) for offline evaluation.

    Parameters
    ----------
    checkpoint_path : str
        Path to DownstreamEvaluator output.
    run_id : str
        Run id of the evaluation or training that generated the desired
        predictions.
    task : str
        Task type, either "val" or "test".

    Returns
    -------
    tuple
        Corrected images, slice numbers, and file names of the original files.
    """
    downstream_tasks = os.listdir(checkpoint_path+run_id)
    img_corr = {d: {} for d in downstream_tasks}
    aggs = ["moco_mean", "moco_weighted", "moco_best-", "moco_bestmasks-"]

    for d in downstream_tasks:
        agg = [a for a in aggs if a in os.listdir(checkpoint_path + run_id +
                                                  "/" + d)]
        img_corr[d] = {a: [] for a in agg}
        for a in agg:
            if a == "moco_mean":
                files = sorted(
                    glob.glob("{}{}/{}/{}/{}_predictions/**/"
                              "**".format(checkpoint_path,
                                          run_id,
                                          d,
                                          a,
                                          task))
                )
                for f in files:
                    img_corr[d][a].append(np.load(f))
            if a == "moco_weighted":
                files = sorted(
                    glob.glob("{}{}/{}/{}/{}_predictions/**/"
                              "**".format(checkpoint_path,
                                          run_id,
                                          d,
                                          a,
                                          task))
                )
                files_fiterrors = sorted(
                    glob.glob("{}{}/{}/{}/{}_fit_errors/**/"
                              "**".format(checkpoint_path,
                                          run_id,
                                          d,
                                          a,
                                          task))
                )
                for f, g in zip(files, files_fiterrors):
                    img = np.load(f)
                    fiterrors = np.loadtxt(g, unpack=True)
                    weight = ((fiterrors - np.amin(fiterrors))
                              / np.sum(fiterrors - np.amin(fiterrors))
                              )
                    img_corr[d][a].append(np.sum(weight[:, None, None, None]
                                                 * img, axis=0)
                                          )
            if a == "moco_best-":
                files = sorted(
                    glob.glob("{}{}/{}/{}/{}_best_images/**/"
                              "**".format(checkpoint_path,
                                          run_id,
                                          d,
                                          a,
                                          task))
                )
                files_fiterrors = sorted(
                    glob.glob("{}{}/{}/{}/{}_best_fit_errors/**/"
                              "**".format(checkpoint_path,
                                          run_id,
                                          d,
                                          a,
                                          task))
                )
                if varying_nr_samples:
                    img_corr[d][a] = {i: [] for i in range(1,
                                                           max_nr_samples + 1)
                                      }
                for f, g in zip(files, files_fiterrors):
                    img = np.load(f)
                    fiterrors = np.loadtxt(g, unpack=True)
                    if not varying_nr_samples:
                        weight = ((fiterrors - np.amin(fiterrors))
                                  / np.sum(fiterrors - np.amin(fiterrors))
                                  )
                        img_corr[d][a].append(np.sum(weight[:, None, None, None]
                                                     * img, axis=0)
                                              )
                    else:
                        ind_sort = np.argsort(fiterrors)[::-1]
                        img_sorted = img[ind_sort]
                        fiterrors_sorted = fiterrors[ind_sort]
                        img_corr[d][a][1].append(img_sorted[0])
                        for i in range(2, max_nr_samples+1):
                            weight = ((fiterrors_sorted[0:i] -
                                       np.amin(fiterrors_sorted[0:i]))
                                      / np.sum(fiterrors_sorted[0:i] -
                                               np.amin(fiterrors_sorted[0:i]))
                                      )
                            img_corr[d][a][i].append(np.sum(
                                weight[:, None, None, None] * img_sorted[0:i],
                                axis=0)
                            )

        if not varying_nr_samples:
            l1 = len(next(iter(img_corr[d].values())))
            if not all(len(lst) == l1 for lst in img_corr[d].values()):
                print("ERROR: Not all agg lists have the same length.")

        for a in img_corr[d].keys():
            if a == "moco_best-":
                if not varying_nr_samples:
                    img_corr[d][a] = np.array(img_corr[d][a])
                else:
                    for i in img_corr[d][a].keys():
                        img_corr[d][a][i] = np.array(img_corr[d][a][i])
            else:
                img_corr[d][a] = np.array(img_corr[d][a])

    slice_nrs = [int(f[f.find("slice-")+6:f.find(".npy")]) for f in files]
    files_mat = [os.path.basename(f[:f.find("_slice")])+".mat" for f in files]

    return img_corr, slice_nrs, files_mat


def process_raw_data(hf_file):
    """Load raw data from h5 file and process to proper complex data."""

    raw_data = hf_file['out']['Data'][:, :, 0, 0, :, 0]
    sens_maps = hf_file['out']['SENSE']['maps'][:, :, 0, 0, :, 0]

    if (isinstance(raw_data, np.ndarray) and
            raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]):
        return (raw_data.view(np.complex64).astype(np.complex64),
                sens_maps.view(np.complex64).astype(np.complex64))
    else:
        print('Error in load_raw_mat: Unexpected data format: ',
              raw_data.dtype)


def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images once."""

    tmp = hf_file['out']['Parameter']['YRange'][:]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    return -int((tmp[0, 0] + tmp[1, 0]) / 2)


def load_hr_qr_moco_data(filename):
    """Load HR/QR-MoCo images (exported from Matlab mqBOLD pipeline)."""

    filename_bm = (filename.replace("/raw_data/", "/brain_masks/")
                   .replace(".mat", "_bm.nii"))
    filename_hrqrmoco = os.readlink(filename_bm).replace(
        "T1w_coreg/rcBrMsk_CSF.nii", "T2S/RDmoco/T2sw_RDMoCo.mat"
    )
    with h5py.File(filename_hrqrmoco, "r") as r:
        img_hrqrmoco = r["out"]["Data"][:]
        if isinstance(img_hrqrmoco, np.ndarray) and img_hrqrmoco.dtype == [
            ('real', '<f8'), ('imag', '<f8')]:
            img_hrqrmoco = img_hrqrmoco.view('complex').astype(np.complex64)

    return np.swapaxes(
        np.swapaxes(img_hrqrmoco,
                    2, 3
                    )[:, :, 10:-10][:, :, ::-1, ::-1],
        0, 1
    )


def pad_sensitivity_maps(sens_maps, kspace_shape):
    """Pad coil sensitivity maps to have same shape as images."""

    pad_width = ((0, 0), (0, 0), (0, 0),
                 (int((kspace_shape[-1] - sens_maps.shape[-1]) / 2),
                  int((kspace_shape[-1] - sens_maps.shape[-1]) / 2))
                 )
    sens_maps = np.pad(sens_maps, pad_width, mode='constant')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.nan_to_num(sens_maps / rss(sens_maps, 1))


def load_original_data(in_folder, dataset, files_mat, slice_nrs,
                       load_brainmask=True, crop_images=True,
                       normalize="abs_image"):
    """
    Load original motion-corrupted and ground truth data.
    This function is a copy and adaption of the
    function _load_h5_data in RawMotionT2starDataset

    """
    (img_gt, img_uncorr, img_rdcorr,
     bm_noCSF, bm, gm_seg, wm_seg) = [], [], [], [], [], [], []
    raw_data_move, sens_maps_move = {}, {}
    raw_data_gt, sens_maps_gt = {}, {}
    y_shifts_move, y_shifts_gt = {}, {}
    imgs_rdmoco = {}

    for filename in np.unique(files_mat):
        filename_move = in_folder + dataset + "raw_data/" + filename
        filename_gt = glob.glob(filename_move[:filename_move.find("_nr")]
                                + "**" + "_fV4.mat")[0]

        # Motion-corrupted data:
        with h5py.File(filename_move, "r") as f:
            (raw_data_move[filename],
             sens_maps_move[filename]) = process_raw_data(f)
            y_shifts_move[filename] = get_yshift(f)

        # Motion-free ground truth data:
        with h5py.File(filename_gt, "r") as g:
            raw_data_gt[filename], sens_maps_gt[filename] = process_raw_data(g)
            y_shifts_gt[filename] = get_yshift(g)

        # HR/QR-MoCo data:
        imgs_rdmoco[filename] = load_hr_qr_moco_data(filename_move)

    for filename, dataslice in zip(files_mat, slice_nrs):
        filename_move = in_folder + dataset + "raw_data/" + filename
        filename_gt = glob.glob(filename_move[:filename_move.find("_nr")]
                                + "**" + "_fV4.mat")[0]
        kspace = raw_data_move[filename][dataslice]
        sens_maps = sens_maps_move[filename][dataslice]
        kspace_gt = raw_data_gt[filename][dataslice]
        smaps_gt = sens_maps_gt[filename][dataslice]
        y_shift = y_shifts_move[filename]
        y_shift_gt = y_shifts_gt[filename]
        img_rdmoco = imgs_rdmoco[filename][dataslice]

        # fully sampled coil combined reconstructions:
        sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)
        coil_imgs_fs = ifft2c(kspace)
        coil_imgs_fs = np.roll(coil_imgs_fs, shift=y_shift, axis=-2)
        img_cc_fs = np.sum(coil_imgs_fs * np.conj(sens_maps), axis=1)
        if crop_images:
            # remove readout oversampling:
            tmp = int(kspace.shape[-1] / 4)
            img_cc_fs = img_cc_fs[:, :, tmp:-tmp]

        # fully sampled coil combined reconstructions:
        smaps_gt = pad_sensitivity_maps(smaps_gt, kspace_gt.shape)
        coil_imgs_fs_gt = ifft2c(kspace_gt)
        coil_imgs_fs_gt = np.roll(coil_imgs_fs_gt, shift=y_shift_gt, axis=-2)
        img_cc_fs_gt = np.sum(coil_imgs_fs_gt * np.conj(smaps_gt), axis=1)
        if crop_images:
            # remove readout oversampling:
            img_cc_fs_gt = img_cc_fs_gt[:, :, int(img_cc_fs_gt.shape[-1] / 4):
                                              -int(img_cc_fs_gt.shape[-1] / 4)]

        # normalize fully sampled gt image:
        if normalize == "abs_image":
            img_cc_fs_gt /= (np.nanmax(abs(img_cc_fs_gt)) + 1e-9)
            img_rdmoco /= (np.nanmax(abs(img_rdmoco)) + 1e-9)
            img_cc_fs /= (np.nanmax(abs(img_cc_fs)) + 1e-9)

        if load_brainmask:
            # load the GT brainmask in correct orientation
            filename_bm = (filename_gt.replace("/raw_data/", "/brain_masks/")
                           .replace(".mat", "_bm.nii"))
            brain_mask = np.where(
                nib.load(filename_bm).get_fdata()[10:-10][::-1, ::-1, :] < 0.5,
                0, 1
            )

            # load a second brainmask (excluding CSF voxels):
            filename_bm_2 = os.readlink(filename_bm).replace("_CSF", "")
            brain_mask_noCSF = np.where(
                nib.load(filename_bm_2)
                .get_fdata()[10:-10][::-1, ::-1, :] < 0.5,
                0, 1
            )

            # load gray and white matter segmentations:
            filename_gm = glob.glob(os.path.dirname(os.readlink(filename_bm))
                                    +"/rc1sub**.nii")[0]
            segmentation_gm = np.where(
                nib.load(filename_gm)
                .get_fdata()[10:-10][::-1, ::-1, :] < 0.5,
                0, 1
            )
            filename_wm = glob.glob(os.path.dirname(os.readlink(filename_bm))
                                    +"/rc2sub**.nii")[0]
            segmentation_wm = np.where(
                nib.load(filename_wm)
                .get_fdata()[10:-10][::-1, ::-1, :] < 0.5,
                0, 1
            )

            bm_noCSF.append(brain_mask_noCSF[:, :, dataslice])
            bm.append(brain_mask[:, :, dataslice])
            gm_seg.append(segmentation_gm[:, :, dataslice])
            wm_seg.append(segmentation_wm[:, :, dataslice])

        img_gt.append(img_cc_fs_gt)
        img_rdcorr.append(img_rdmoco)
        img_uncorr.append(img_cc_fs)

    return (np.array(img_gt), np.array(img_uncorr), np.array(img_rdcorr),
            np.array(bm_noCSF), np.array(bm), np.array(gm_seg),
            np.array(wm_seg))


def calc_masked_MAE(img1, img2, mask):
    """Calculate Mean Absolute Error between two images in a specified mask"""

    masked_diff = np.ma.masked_array(abs(img1 - img2),
                                     mask=(mask[:, None] != 1))
    return np.mean(masked_diff, axis=(1, 2)).filled(0)


def calc_masked_SSIM(img, img_ref,  mask):
    """Calculate SSIM between two 3D images in a specified mask"""

    ssims = []
    for i in range(len(img)):
        mssim, ssim_values = structural_similarity(
            img_ref[i], img[i], data_range=np.amax(img_ref[i]),
            gaussian_weights=True, full=True
        )
        masked = np.ma.masked_array(ssim_values, mask=(mask[i] != 1))
        ssims.append(np.mean(masked))
    return np.array(ssims)


def calc_masked_SSIM_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False):
    """Calculate SSIM between two 4D images in a specified mask"""

    ssims = []
    for i in range(img.shape[0]):
        ssim_echoes = []
        for j in range(img.shape[1]):
            mssim, ssim_values = structural_similarity(
                img_ref[i, j], img[i, j], data_range=np.amax(img_ref[i, j]),
                gaussian_weights=True, full=True
            )
            masked = np.ma.masked_array(ssim_values, mask=(mask[i] != 1))
            ssim_echoes.append(np.mean(masked))
        if av_echoes:
            ssims.append(np.mean(ssim_echoes))
        elif later_echoes:
            ssims.append(np.mean(ssim_echoes[-later_echoes:]))
        else:
            ssims.append(ssim_echoes)
    return np.array(ssims)


def calc_masked_PSNR_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False):
    """Calculate PSNR between two 4D images in a specified mask"""

    psnrs = []
    for i in range(img.shape[0]):
        psnr_echoes = []
        for j in range(img.shape[1]):
            d_ref = img_ref[i, j].flatten()[mask[i].flatten() > 0]
            d_img = img[i, j].flatten()[mask[i].flatten() > 0]
            psnr_echoes.append(
                peak_signal_noise_ratio(d_ref, d_img,
                                        data_range=np.amax(img_ref[i, j]))
            )
        if av_echoes:
            psnrs.append(np.mean(psnr_echoes))
        elif later_echoes:
            psnrs.append(np.mean(psnr_echoes[-later_echoes:]))
        else:
            psnrs.append(psnr_echoes)
    return np.array(psnrs)


def calc_regional_average(img1, mask):
    """Calculate the average value of an image in a specified mask"""

    masked = np.ma.masked_array(img1, mask=(mask[:, None] != 1))
    return np.mean(masked, axis=(1, 2)).filled(0)


def calc_regional_coeff_var(img1, mask):
    """Calculate coefficient of variation of an image in a specified mask"""

    masked = np.ma.masked_array(img1, mask=(mask[:, None] != 1))
    return (np.std(masked, axis=(1, 2)).filled(0)
            / np.mean(masked, axis=(1, 2)).filled(0))
