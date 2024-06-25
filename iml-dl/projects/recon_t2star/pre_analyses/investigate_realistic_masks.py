""" To run this script: insert your file paths in lines 264, 266, 298, 301."""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import h5py as h5
import nibabel as nib
from medutils.mri import ifft2c, rss
from scipy.ndimage import binary_erosion
from transforms3d.affines import decompose, compose
from transforms3d.euler import mat2euler, euler2mat


def transform_sphere(dset_shape, motion_parameters, pixel_spacing, radius):
    """Rigidly transform a sphere with given motion parameters"""

    # get all voxels within sphere around isocenter:
    dim1, dim2, dim3 = dset_shape[-3:]
    zz, xx, yy = np.ogrid[:dim1, :dim2, :dim3]
    zz = zz * pixel_spacing[0]
    xx = xx * pixel_spacing[1]
    yy = yy * pixel_spacing[2]
    center = [np.mean(zz), np.mean(xx), np.mean(yy)]
    d2 = (zz - center[0]) ** 2 + (xx - center[1]) ** 2 + (yy - center[2]) ** 2
    mask = d2 <= radius ** 2
    z, x, y = np.nonzero(mask)
    coords = np.array(list(zip(z, x, y)))
    coords[:, 0] = coords[:, 0] * pixel_spacing[0]
    coords[:, 1] = coords[:, 1] * pixel_spacing[1]
    coords[:, 2] = coords[:, 2] * pixel_spacing[2]

    # reduce number of coordinates to speed up calculation:
    coords = coords[::100]

    # apply the transforms to the coordinates:
    centroids = []
    tr_coords = []
    for pars in motion_parameters:
        T = np.array(pars[0:3]) / np.array(pixel_spacing)
        R = np.array(pars[3:]) * np.pi / 180
        tr_coords_ = np.matmul(coords, euler2mat(*R).T)
        tr_coords_ = tr_coords_ + T
        tr_coords.append(tr_coords_)
        centroids.append(np.mean(tr_coords_, axis=0))

    return np.array(centroids), np.array(tr_coords)


def transf_from_parameters(T, R):
    """
    Use python module transforms3d to extract transformation matrix from
    translation and rotation parameters.

    Parameters
    ----------
    T : numpy array
        translation parameters.
    R : numpy array
        rotation angles in degrees.
    Returns
    -------
    A : numpy array (4x4)
        transformation matrix.
    """
    R_mat = euler2mat(R[2] * np.pi / 180, R[1] * np.pi / 180, R[0] * np.pi / 180)
    A = compose(T, R_mat, np.ones(3))

    return A


def parameters_from_transf(A):
    '''
    Use python module transforms3d to extract translation and rotation
    parameters from transformation matrix.
    Parameters
    ----------
    A : numpy array (4x4)
        transformation matrix.
    Returns
    -------
    T : numpy array
        translation parameters.
    R : numpy array
        rotation angles in degrees.
    '''
    T, R_, Z_, S_ = decompose(A)
    al, be, ga = mat2euler(R_)
    R = np.array([ga * 180 / np.pi, be * 180 / np.pi, al * 180 / np.pi])

    return np.array(T), R


class ImportMotionDataNpy:
    """Importing PCA-augmented fMRI motion data (Old version).

    fMRI motion data was decomposed with sklearn.decomposition.PCA and the
    first principal components (unit eigenvectors) can now be combined (weighted
    by corresponding eigenvalues) with the mean motion curves to generate new
    motion curves.

    Parameters
    ----------
    npy_file : str
        path to folder containing the results of the PC analysis. The folder
        should contain two text files (expl_var_<scenario>.txt and
        Mean_<scenario>.txt)) as well as a subfolder components_<scenario>,
        which contains text files with the unit eigenvectors
        (pc_00.txt, pc_o1.txt, ..).
    scan_length : int
        defining the necessary length of the motion curve.
    nr_curve : int
        number of the curve in the npy file
    random_start_time : bool
        whether a random time in the motion curve should be picked as start
        time if the curve is longer than scan_length
    reference_to_0 : bool
        whether the motion curve should be transformed so that the median
        position (reference) is at the zero-point.

    Atributes
    -------
    get_motion_data : numpy array
        outputs time points, translational and rotational parameters which are
        prepared as stated above.
    """

    def __init__(self, npy_file, scan_length, nr_curve,
                 random_start_time=True, reference_to_0=True):
        super().__init__()
        self.reference_to_0 = reference_to_0

        # load the relevant file and extract the data:
        npy_data = np.load(npy_file, allow_pickle=True)[()]
        self.seconds = npy_data['Time_seconds']
        self.motion_data = np.zeros((6, len(self.seconds)))
        for i, t in enumerate(['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']):
            self.motion_data[i] = npy_data[t][nr_curve]

        # only take time points within scan duration, sample randomly throughout time:
        if random_start_time:
            last_time = self.seconds[-1] - scan_length
            if last_time < 1:
                random_start = 0
                print("ERROR in ImportMotionDataPCAfMRI: the loaded motion data "
                      "is shorter than the needed scan length!")
            else:
                random_start = np.random.randint(0, last_time)
            ind = np.intersect1d(np.where(self.seconds >= random_start),
                                 np.where(self.seconds <= (random_start + scan_length)))
            self.motion_data = self.motion_data[:, ind]
            self.seconds = self.seconds[ind]
            self.seconds = self.seconds - self.seconds[0]
        else:
            self.motion_data = self.motion_data[:, self.seconds < scan_length]
            self.seconds = self.seconds[self.seconds < scan_length]


    def get_motion_data(self, dset_shape, pixel_spacing=[3.3, 2, 2], radius=64):
        """Get motion parameters.

        If self.reference_to_0 is set to True,
        the transformation parameters are first transformed so that the
        reference (median) position corresponds to a zero transformation.

        Parameters
        ----------
        dset_shape : numpy array
            shape of the dataset.
        pixel_spacing : list of length 3
            spacing of the voxels in z, x and y-direction
        radius : int or float
            radius of the sphere used for calculating reference position
        Returns
        -------
        seconds : numpy array
            time points in seconds
        T : numpy array
            translation parameters.
        R : numpy array
            rotation angles in degrees.
        """

        # get motion parameters (PCA output is already in mm and degree)
        T, R = self.motion_data[:3].T, self.motion_data[3:].T

        if not self.reference_to_0:
            return self.seconds, T, R

        else:
            # calculate reference through median of sphere's centroids
            tmp = np.array([T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
            centroids, _ = transform_sphere(dset_shape, tmp, pixel_spacing, radius)

            ind_median_centroid = np.argmin(np.sqrt(np.sum(
                (centroids - np.median(centroids, axis=0)) ** 2, axis=1)))

            # transform all matrices so that ind_median_centroid corresponds to identity:
            matrices = np.zeros((len(T), 4, 4))
            for i in range(len(T)):
                matrices[i] = transf_from_parameters(T[i], R[i])

            tr_matrices = np.matmul(np.linalg.inv(matrices[ind_median_centroid]),
                                    matrices)

            # get motion parameters
            T_reference0, R_reference0 = np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for i in range(len(T)):
                T_reference0[i], R_reference0[i] = parameters_from_transf(tr_matrices[i])

            return self.seconds, T_reference0, R_reference0


def create_mask_from_motion(motion_tracking, dataset, motion_thr=0.5, radius=64, pixel_spacing=[3.3, 2, 2],
                            nr_pe_steps=92, path_scan_order="path_to_scan_order/Scan_order.txt"):
    """
    Calculate a mask depending on the average displacement of a sphere that
    is transformed according to the motion curve

    The mask is 1, if the displacement is smaller/equal than the threshold
    (self.motion_thr), and 0 otherwise.
    """
    motion_parameters = motion_tracking[:, 1:]
    motion_times = motion_tracking[:, 0]
    temp = np.loadtxt(path_scan_order, unpack=True)
    acq_times, reps, echoes, slices, ys = temp

    centroids, tr_coords = transform_sphere(dataset.shape, motion_parameters,
                                            pixel_spacing, radius)

    # calculate reference through median
    ind_median_centroid = np.argmin(np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))

    # calculate average voxel displacement magnitude
    displ = tr_coords - tr_coords[ind_median_centroid]
    magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
    av_magn = np.mean(magn, axis=1)

    # 0 for motion > threshold, 1 for motion <= threshold (lines that can be included)
    motion_class = np.zeros(len(motion_times), dtype=int)
    motion_class[av_magn <= motion_thr] = 1
    print(
        '{}% of all time points <= {}mm displacement'.format(
            np.round(len(av_magn[av_magn <= motion_thr]) / len(av_magn) * 100, 3),
            motion_thr))

    max_slices = dataset.shape[1]
    mask = np.ones_like(dataset)

    not_acquired = np.amax(dataset.shape[-2]) - nr_pe_steps
    ys_sh = ys + int(not_acquired / 2)

    for t, r, e, s, y in zip(acq_times, reps, echoes, slices, ys_sh):
        if s < max_slices:
            idx = np.argmin(np.abs(motion_times - t))
            mask[int(e), int(s), int(y)] = motion_class[idx]
    mask = mask.astype(int)
    red_mask = mask[:, :, 0]  # only pick one value per pe line

    return av_magn, red_mask, mask


# Test curves:
npy_file = "path_to_real_motion_curves.npy"
nr_slices = 36
path_scan_order = "path_to/Scan_order_" + str(nr_slices) + ".txt"
scan_length = int(np.loadtxt(path_scan_order)[-1, 0]) + 1  # duration of a scan in seconds

masks = []
for i in range(0, 24):
    MotionImport = ImportMotionDataNpy(npy_file=npy_file, scan_length=scan_length,
                                       nr_curve=i, random_start_time=True, reference_to_0=True)
    times, T, R = MotionImport.get_motion_data((12, 36, 92, 112))
    motion = np.array([times, T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
    magn, mask, full_mask = create_mask_from_motion(motion_tracking=motion,
                                                    dataset=np.zeros((12, 36, 92, 112)),
                                                    motion_thr=0.5,
                                                    path_scan_order=path_scan_order)
    masks.append(full_mask)

    # mask_ = np.mean(full_mask, axis=0)[0]
    # plt.imshow(mask_.T, cmap='gray')
    # plt.axis('off')
    # plt.title("Test curves")
    # plt.show()


masks = np.mean(np.array(masks), axis=1)
aggr_masks = np.mean(masks, axis=(0, 1))

plt.imshow(aggr_masks.T, cmap='gray')
plt.axis('off')
plt.title("Test curves")
plt.show()


# Train curves:
npy_file = "path_to_real_motion_curves.npy"
nr_slices = 36
path_scan_order = "path_to/Scan_order_" + str(nr_slices) + ".txt"
scan_length = int(np.loadtxt(path_scan_order)[-1, 0]) + 1  # duration of a scan in seconds

masks = []
for i in range(0, 88):
    MotionImport = ImportMotionDataNpy(npy_file=npy_file, scan_length=scan_length,
                                       nr_curve=i, random_start_time=True, reference_to_0=True)
    times, T, R = MotionImport.get_motion_data((12, 36, 92, 112))
    motion = np.array([times, T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
    magn, mask, full_mask = create_mask_from_motion(motion_tracking=motion,
                                                    dataset=np.zeros((12, 36, 92, 112)),
                                                    motion_thr=0.5,
                                                    path_scan_order=path_scan_order)
    masks.append(full_mask)

masks = np.mean(np.array(masks), axis=1)
aggr_masks = np.mean(masks, axis=(0, 1))

plt.imshow(aggr_masks.T, cmap='gray')
plt.axis('off')
plt.title("Train curves")
plt.show()


random_mask = ["VarDensBlocks", 2, 3]  # type, acc rate, max width of blocks
npe = 92
std_scale = 5

for std_scale in [2, 3, 4, 5, 6, 7]:

    aggr_masks = np.zeros(npe)
    for i in range(0, 200):
        mask = np.ones(npe)
        nr_var1lines = npe-int(npe * 1 / random_mask[1])

        center = npe // 2

        # variable density via sampling from gaussian:
        count = 0
        while count < nr_var1lines:
            indx = int(np.round(np.random.normal(loc=center, scale=(npe - 1) / std_scale)))
            if indx < 46:
                indx += 46
            elif indx >= 46:
                indx -= 46

            width = np.random.randint(0, random_mask[2])
            if 0 <= indx < npe-width and mask[indx] == 1:
                for w in range(0, width+1):
                    mask[indx+w] = 0
                    count += 1


        aggr_masks += 1/200 *mask

    # reshape:
    nc = 32
    ne = 12
    nfe = 112
    full_mask = mask.reshape(1, 1, npe, 1).repeat(nc, axis=0).repeat(ne, axis=1).repeat(nfe, axis=3)
    full_aggr_mask = aggr_masks.reshape(1, 1, npe, 1).repeat(nc, axis=0).repeat(ne, axis=1).repeat(nfe, axis=3)

    plt.imshow(full_mask[0, 0].T, cmap='gray')
    plt.axis('off')
    plt.title("Example Std Scale {}".format(std_scale))
    plt.show()

    plt.imshow(full_aggr_mask[0, 0].T, cmap='gray')
    plt.axis('off')
    plt.title("Aggregated Std Scale {}".format(std_scale))
    plt.show()


print('Done')

