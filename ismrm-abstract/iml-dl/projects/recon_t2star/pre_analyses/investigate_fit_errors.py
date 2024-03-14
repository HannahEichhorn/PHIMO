"""To run this script: insert your file paths in lines 84-92."""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import h5py as h5
import nibabel as nib
from medutils.mri import ifft2c, rss
from scipy.ndimage import binary_erosion


def load_raw_mat_file(file, crop_ro=True):
    """Loading mat files containing raw data converted with MRecon"""

    f = h5.File(file, 'r')
    raw_data = f['out']['Data'][:, :, 0, 0, :, 0]
    tmp = f['out']['Parameter']['YRange'][:]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    y_shift = -int((tmp[0,0]+tmp[1,0])/2)

    sens_maps = f['out']['SENSE']['maps'][:, :, 0, 0, :, 0]

    # convert to proper complex data
    if isinstance(raw_data, np.ndarray) and raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]:
        return raw_data.view(np.complex64).astype(np.complex64), y_shift, sens_maps.view(np.complex64).astype(np.complex64)

    else:
        print('Error in load_raw_mat: Unexpected data format: ', raw_data.dtype)


def exponential_decay(t, A, T2star):
    return A * np.exp(-t / T2star)


def t2star_linregr(data, bm, TE1=5, dTE=5):
    """

    :param data: input data to be fitted
    :param bm: brainmask (0 / 1 for background / brain)
    :param TE1: first echo time, default: 5ms
    :param dTE: echo distance, default: 5ms
    :return: fitted T2star and amplitude maps
    """

    TE = np.arange(TE1, data.shape[0] * TE1 + 1, dTE)
    slope, interc = np.zeros(shape=bm.shape), np.zeros(shape=bm.shape)

    fit_data = np.log(data+1e-9)

    for i in range(bm.shape[0]):
        for j in range(bm.shape[1]):
            if bm[i, j]:
                try:
                    result = linregress(TE, fit_data[:, i, j])
                    slope[i, j], interc[i, j] = result.slope, result.intercept
                except:
                    pass

    T2star, A = -1 / slope, np.exp(interc)
    T2star = np.clip(T2star, 0, 200)
    return T2star, A


def t2_star_fit_error(data, bm, T2star, A, TE1=5, dTE=5):
    TE = np.arange(TE1, data.shape[0] * TE1 + 1, dTE)
    TE = np.repeat(TE[:, np.newaxis, np.newaxis], data.shape[1], axis=1)
    TE = np.repeat(TE, data.shape[2], axis=2)

    data_fit = exponential_decay(TE, A, T2star)

    mean_data, mean_data_fit = np.mean(data, axis=0), np.mean(data_fit, axis=0)
    emp_corr = (np.sum((data-mean_data)*(data_fit-mean_data_fit), axis=0) /
                np.sqrt(np.sum((data-mean_data)**2, axis=0) * np.sum((data_fit-mean_data_fit)**2, axis=0)))

    return (np.mean(np.mean(abs(data-data_fit), axis=0)[bm == 1]),
            np.mean(np.mean(abs(data-data_fit)/data, axis=0)[bm == 1]),
            np.nanmean(emp_corr[bm == 1]))


''' Look into T2* fit:'''
# train and validation images:
filenames_still = ["path/to/still_file.mat",
                   "path/to/still_file.mat",
                   "path/to/still_file.mat", 
                   "path/to/still_file.mat"]

filenames_move = ["path/to/motion_file.mat",
                   "path/to/motion_file.mat",
                   "path/to/motion_file.mat",
                   "path/to/motion_file.mat"]

MAE_still_withCSF, MAE_still_withoutCSF = [], []
MAE_move_withCSF, MAE_move_withoutCSF = [], []
MAPE_still_withCSF, MAPE_still_withoutCSF = [], []
MAPE_move_withCSF, MAPE_move_withoutCSF = [], []
ECC_still_withCSF, ECC_still_withoutCSF = [], []
ECC_move_withCSF, ECC_move_withoutCSF = [], []

for filename_raw_still, filename_raw_move in zip(filenames_still, filenames_move):
    data_still_, y_shift_still, sens_maps_still_ = load_raw_mat_file(filename_raw_still)
    data_move_, y_shift_move, sens_maps_move_ = load_raw_mat_file(filename_raw_move)

    for dataslice in range(0, data_still_.shape[0]):
        data_still, sens_maps = data_still_[dataslice], sens_maps_still_[dataslice]
        coil_images = ifft2c(data_still)
        coil_images = np.roll(coil_images, shift=y_shift_still, axis=-2)
        coil_images = coil_images[:, :, :, 56:-56]
        sens_maps = np.nan_to_num(sens_maps / rss(sens_maps, 1)[:, None])
        img_still = np.sum(coil_images * np.conj(sens_maps), axis=1)
        filename_bm_still = filename_raw_still.replace("raw_data", "brain_masks").replace(".mat", "_bm.nii")
        bm_still = np.where(nib.load(filename_bm_still).get_fdata()[10:-10][::-1, ::-1, :] < 0.5, 0, 1)[:, :, dataslice]
        filename_bm_still_noCSF = os.path.realpath(filename_bm_still).replace("_CSF", "")
        bm_still_noCSF = np.where(nib.load(filename_bm_still_noCSF).get_fdata()[10:-10][::-1, ::-1, :] < 0.5, 0, 1)[:, :, dataslice]


        data_move, sens_maps = data_move_[dataslice], sens_maps_move_[dataslice]
        coil_images = ifft2c(data_move)
        coil_images = np.roll(coil_images, shift=y_shift_move, axis=-2)
        coil_images = coil_images[:, :, :, 56:-56]
        sens_maps = np.nan_to_num(sens_maps / rss(sens_maps, 1)[:, None])
        img_move = np.sum(coil_images * np.conj(sens_maps), axis=1)
        filename_bm_move = filename_raw_move.replace("raw_data", "brain_masks").replace(".mat", "_bm.nii")
        bm_move = np.where(nib.load(filename_bm_move).get_fdata()[10:-10][::-1, ::-1, :] < 0.5, 0, 1)[:, :, dataslice]
        filename_bm_move_noCSF = os.path.realpath(filename_bm_move).replace("_CSF", "")
        bm_move_noCSF = np.where(nib.load(filename_bm_move_noCSF).get_fdata()[10:-10][::-1, ::-1, :] < 0.5, 0, 1)[:, :, dataslice]

        # apply binary erosion to the brainmasks to decrease size and remove misregistration errors:
        bm_still = binary_erosion(binary_erosion(bm_still))
        bm_still_noCSF = binary_erosion(binary_erosion(bm_still_noCSF))
        bm_move = binary_erosion(binary_erosion(bm_move))
        bm_move_noCSF = binary_erosion(binary_erosion(bm_move_noCSF))

        if np.sum(bm_still) == 0 or np.sum(bm_still_noCSF) == 0 or np.sum(bm_move) == 0 or np.sum(bm_move_noCSF) == 0:
            continue

        # look into fit errors:
        # brainmask with CSF
        T2star, A = t2star_linregr(abs(img_still), bm_still)
        mae, mape, ecc = t2_star_fit_error(abs(img_still), bm_still, T2star, A)
        MAE_still_withCSF.append(mae)
        MAPE_still_withCSF.append(mape)
        ECC_still_withCSF.append(ecc)
        T2star, A = t2star_linregr(abs(img_move), bm_move)
        mae, mape, ecc = t2_star_fit_error(abs(img_move), bm_move, T2star, A)
        MAE_move_withCSF.append(mae)
        MAPE_move_withCSF.append(mape)
        ECC_move_withCSF.append(ecc)

        # brainmask without CSF
        T2star, A = t2star_linregr(abs(img_still), bm_still_noCSF)
        mae, mape, ecc = t2_star_fit_error(abs(img_still), bm_still_noCSF, T2star, A)
        MAE_still_withoutCSF.append(mae)
        MAPE_still_withoutCSF.append(mape)
        ECC_still_withoutCSF.append(ecc)
        T2star, A = t2star_linregr(abs(img_move), bm_move_noCSF)
        mae, mape, ecc = t2_star_fit_error(abs(img_move), bm_move_noCSF, T2star, A)
        MAE_move_withoutCSF.append(mae)
        MAPE_move_withoutCSF.append(mape)
        ECC_move_withoutCSF.append(ecc)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Including CSF")
plt.violinplot([MAE_still_withCSF, MAE_move_withCSF], [0, 1], showmeans=True, showextrema=False, widths=0.2)
for s, m in zip(MAE_still_withCSF, MAE_move_withCSF):
    plt.plot([0, 1], [s, m], 'gray', linewidth=0.5)
plt.ylabel("MAE")
plt.ylim(25, 220)
plt.xticks([0, 1], ["Still", "Move"])
plt.subplot(1, 2, 2)
plt.title("Excluding CSF")
plt.violinplot([MAE_still_withoutCSF, MAE_move_withoutCSF], [0, 1], showmeans=True, showextrema=False, widths=0.2)
for s, m in zip(MAE_still_withoutCSF, MAE_move_withoutCSF):
    plt.plot([0, 1], [s, m], 'gray', linewidth=0.5)
plt.ylabel("MAE")
plt.xticks([0, 1], ["Still", "Move"])
plt.ylim(25, 220)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Including CSF")
plt.violinplot([MAPE_still_withCSF, MAPE_move_withCSF], [0, 1], showmeans=True, showextrema=False, widths=0.2)
for s, m in zip(MAPE_still_withCSF, MAPE_move_withCSF):
    plt.plot([0, 1], [s, m], 'gray', linewidth=0.5)
plt.ylabel("MAPE")
plt.ylim(0, 0.31)
plt.xticks([0, 1], ["Still", "Move"])
plt.subplot(1, 2, 2)
plt.title("Excluding CSF")
plt.violinplot([MAPE_still_withoutCSF, MAPE_move_withoutCSF], [0, 1], showmeans=True, showextrema=False, widths=0.2)
for s, m in zip(MAPE_still_withoutCSF, MAPE_move_withoutCSF):
    plt.plot([0, 1], [s, m], 'gray', linewidth=0.5)
plt.ylabel("MAPE")
plt.xticks([0, 1], ["Still", "Move"])
plt.ylim(0, 0.31)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Including CSF")
plt.violinplot([ECC_still_withCSF, ECC_move_withCSF], [0, 1], showmeans=True, showextrema=False, widths=0.2)
for s, m in zip(ECC_still_withCSF, ECC_move_withCSF):
    plt.plot([0, 1], [s, m], 'gray', linewidth=0.5)
plt.ylabel("ECC")
plt.ylim(0.9, 1)
plt.xticks([0, 1], ["Still", "Move"])
plt.subplot(1, 2, 2)
plt.title("Excluding CSF")
plt.violinplot([ECC_still_withoutCSF, ECC_move_withoutCSF], [0, 1], showmeans=True, showextrema=False, widths=0.2)
for s, m in zip(ECC_still_withoutCSF, ECC_move_withoutCSF):
    plt.plot([0, 1], [s, m], 'gray', linewidth=0.5)
plt.ylabel("ECC")
plt.xticks([0, 1], ["Still", "Move"])
plt.ylim(0.9, 1)
plt.show()
