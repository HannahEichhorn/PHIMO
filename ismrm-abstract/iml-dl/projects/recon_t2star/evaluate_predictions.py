import argparse
import yaml
from scipy.ndimage import binary_dilation
from utils_plot import *
from utils import *


""" 
Evaluations performed in this script:

1) MAE of T2* values (with  registration to GT)
2) SSIM of T2* values (with  registration to GT)
3) average gray and white matter values
    3a) accuracy
    3b) precision (coefficient of variability)
4) Visual examples
5) SSIM and PSNR on images 
    5a) average over all echoes
    5b) average over last 3 echoes
"""

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--config_path',
                    type=str,
                    default='./configs/config_eval.yaml',
                    metavar='C',
                    help='path to configuration yaml file')
args = parser.parse_args()
with open(args.config_path, 'r') as stream_file:
    config_file = yaml.load(stream_file, Loader=yaml.FullLoader)

run_id = config_file["run_id"]
checkpoint_path = config_file["checkpoint_path"]
in_folder = config_file["in_folder"]
dataset = config_file["dataset"]
task = [t for t in ["val", "test"] if t in dataset][0]
out_dir = checkpoint_path.replace("downstream_metrics",
                                  "evaluation_downstream_metrics") + run_id


if (config_file["load_original"]
        or not os.path.exists(out_dir + "/t2star_maps/")):
    img_corr, slice_nrs, files_mat = load_predictions(checkpoint_path,
                                                      run_id,
                                                      task)
    (img_gt, img_uncorr, img_rdcorr, bm_noCSF,
     bm, gm_seg, wm_seg) = load_original_data(in_folder,
                                              dataset,
                                              files_mat,
                                              slice_nrs)

    # use dilated brainmask during fitting to avoid need for
    # registration for all except gt:
    bm_ = np.zeros_like(bm)
    for i in range(bm.shape[0]):
        bm_[i] = binary_dilation(binary_dilation(bm[i]))

    FitErrorGT = T2starFit(img_gt[:, None], bm)
    t2star_gt, _ = FitErrorGT.t2star_linregr()

    FitErrorUncorr = T2starFit(img_uncorr[:, None], bm_)
    t2star_uncorr, _ = FitErrorUncorr.t2star_linregr()

    FitErrorRDCorr = T2starFit(img_rdcorr[:, None], bm_)
    t2star_rdcorr, _ = FitErrorRDCorr.t2star_linregr()

    t2star_corr = {d: {} for d in img_corr.keys()}
    for d in img_corr.keys():
        for agg in img_corr[d].keys():
            FitErrorCorr = T2starFit(img_corr[d][agg][:, None], bm_)
            t2star_corr[d][agg], _ = FitErrorCorr.t2star_linregr()

    # register the images to gt:
    img_rdcorr_reg, img_uncorr_reg = [], []
    img_corr_reg = {d: {} for d in img_corr.keys()}
    t2star_rdcorr_reg, t2star_uncorr_reg = [], []
    t2star_corr_reg = {d: {} for d in t2star_corr.keys()}

    for i in range(len(img_gt)):
        tmp1, tmp2 = rigid_registration(abs(img_gt[i])[0],
                                        abs(img_uncorr[i])[0],
                                        abs(img_uncorr)[i],
                                        t2star_uncorr[i],
                                        numpy=True)
        img_uncorr_reg.append(tmp1)
        t2star_uncorr_reg.append(tmp2[0])
        tmp1, tmp2 = rigid_registration(abs(img_gt[i])[0],
                                        abs(img_rdcorr[i])[0],
                                        abs(img_rdcorr)[i],
                                        t2star_rdcorr[i],
                                        numpy=True)
        img_rdcorr_reg.append(tmp1)
        t2star_rdcorr_reg.append(tmp2[0])

        for d in t2star_corr.keys():
            for a in t2star_corr[d].keys():
                if a not in t2star_corr_reg[d].keys():
                    t2star_corr_reg[d][a] = []
                if a not in img_corr_reg[d].keys():
                    img_corr_reg[d][a] = []

                tmp1, tmp2 = rigid_registration(abs(img_gt[i])[0],
                                                abs(img_corr[d][a][i])[0],
                                                abs(img_corr[d][a])[i],
                                                t2star_corr[d][a][i],
                                                numpy=True)
                img_corr_reg[d][a].append(tmp1)
                t2star_corr_reg[d][a].append(tmp2[0])

    img_rdcorr_reg = np.array(img_rdcorr_reg)
    img_uncorr_reg = np.array(img_uncorr_reg)
    t2star_rdcorr_reg = np.array(t2star_rdcorr_reg) * bm
    t2star_uncorr_reg = np.array(t2star_uncorr_reg) * bm

    for d in t2star_corr.keys():
        for a in t2star_corr[d].keys():
            t2star_corr_reg[d][a] = np.array(t2star_corr_reg[d][a]) * bm
            img_corr_reg[d][a] = np.array(img_corr_reg[d][a])

    # save the T2star maps and segmentations:
    if not os.path.exists(out_dir + "/t2star_maps/"):
        os.makedirs(out_dir + "/t2star_maps/")

        np.save(out_dir + "/t2star_maps/gt.npy", t2star_gt)
        np.save(out_dir + "/t2star_maps/uncorr.npy", t2star_uncorr)
        np.save(out_dir + "/t2star_maps/rdcorr.npy", t2star_rdcorr)
        np.save(out_dir + "/t2star_maps/bm_noCSF.npy", bm_noCSF)
        np.save(out_dir + "/t2star_maps/bm.npy", bm)
        np.save(out_dir + "/t2star_maps/gm_seg.npy", gm_seg)
        np.save(out_dir + "/t2star_maps/wm_seg.npy", wm_seg)

        for d in t2star_corr.keys():
            create_dir(out_dir + "/t2star_maps/corr/" + d)
            for a in t2star_corr[d].keys():
                np.save(out_dir + "/t2star_maps/corr/" + d + "/" + a + ".npy",
                        t2star_corr[d][a])

        # save the registered images and T2star maps:
        create_dir(out_dir + "/t2star_maps_reg/")
        np.save(out_dir + "/t2star_maps_reg/uncorr.npy", t2star_uncorr_reg)
        np.save(out_dir + "/t2star_maps_reg/rdcorr.npy", t2star_rdcorr_reg)

        for d in t2star_corr.keys():
            create_dir(out_dir + "/t2star_maps_reg/corr/" + d)
            for a in t2star_corr[d].keys():
                np.save(
                    out_dir + "/t2star_maps_reg/corr/" + d + "/" + a + ".npy",
                    t2star_corr_reg[d][a])

        create_dir(out_dir + "/images_reg/")
        np.save(out_dir + "/images_reg/gt.npy", img_gt)
        np.save(out_dir + "/images_reg/uncorr.npy", img_uncorr_reg)
        np.save(out_dir + "/images_reg/rdcorr.npy", img_rdcorr_reg)

        for d in t2star_corr.keys():
            create_dir(out_dir + "/images_reg/corr/" + d)
            for a in t2star_corr[d].keys():
                np.save(out_dir + "/images_reg/corr/" + d + "/" + a + ".npy",
                        img_corr_reg[d][a])

        # save slice numbers and filenames:
        np.save(out_dir + "/slice_nrs.npy", slice_nrs)
        np.save(out_dir + "/filenames.npy", files_mat)

    else:
        print(out_dir + " already exists. Not saving the T2* maps.")
else:
    t2star_gt = np.load(out_dir + "/t2star_maps/gt.npy")
    t2star_uncorr = np.load(out_dir + "/t2star_maps/uncorr.npy")
    t2star_rdcorr = np.load(out_dir + "/t2star_maps/rdcorr.npy")
    bm_noCSF = np.load(out_dir + "/t2star_maps/bm_noCSF.npy")
    bm = np.load(out_dir + "/t2star_maps/bm.npy")
    gm_seg = np.load(out_dir + "/t2star_maps/gm_seg.npy")
    wm_seg = np.load(out_dir + "/t2star_maps/wm_seg.npy")
    t2star_uncorr_reg = np.load(out_dir + "/t2star_maps_reg/uncorr.npy")
    t2star_rdcorr_reg = np.load(out_dir + "/t2star_maps_reg/rdcorr.npy")
    img_gt = np.load(out_dir + "/images_reg/gt.npy")
    img_uncorr_reg = np.load(out_dir + "/images_reg/uncorr.npy")
    img_rdcorr_reg = np.load(out_dir + "/images_reg/rdcorr.npy")

    t2star_corr, t2star_corr_reg, img_corr_reg = {}, {}, {}
    for d in os.listdir(out_dir + "/t2star_maps/corr/"):
        t2star_corr[d], t2star_corr_reg[d], img_corr_reg[d] = {}, {}, {}
        for a in os.listdir(out_dir + "/t2star_maps/corr/" + d + "/"):
            t2star_corr[d][a.replace(".npy", "")] = np.load(
                out_dir + "/t2star_maps/corr/" + d + "/" + a
            )
            t2star_corr_reg[d][a.replace(".npy", "")] = np.load(
                out_dir + "/t2star_maps_reg/corr/" + d + "/" + a
            )
            img_corr_reg[d][a.replace(".npy", "")] = np.load(
                out_dir + "/images_reg/corr/" + d + "/" + a
            )

    slice_nrs = np.load(out_dir + "/slice_nrs.npy")
    files_mat = np.load(out_dir + "/filenames.npy")

t2star_gt = t2star_gt[:, 0]
t2star_uncorr = t2star_uncorr[:, 0]
t2star_rdcorr = t2star_rdcorr[: ,0]
for d in t2star_corr.keys():
    for a in t2star_corr[d].keys():
        t2star_corr[d][a] = t2star_corr[d][a][:, 0]
        img_corr_reg[d][a] = abs(img_corr_reg[d][a])
img_gt = abs(img_gt)
img_uncorr_reg = abs(img_uncorr_reg)
img_rdcorr_reg = abs(img_rdcorr_reg)

# only consider indices where slice_nr > min_slice_num:
ind = np.where(slice_nrs >= 0) #np.where(slice_nrs > 15)
min_slice_num = {"SQ-struct-33": 13,
                 "SQ-struct-38": 12,
                 "SQ-struct-39": 15,
                 "SQ-struct-43": 14,
                 "SQ-struct-45": 14,
                 "SQ-struct-46": 10} # >=!!!
subjects = [f[:12] for f in files_mat]
ind = [i for i in ind[0] if slice_nrs[i] >= min_slice_num[subjects[i]]]

# # look at all slices from all subjects:
# for subj in ["SQ-struct-33", "SQ-struct-38", "SQ-struct-39", "SQ-struct-43", "SQ-struct-45", "SQ-struct-46"]:
#     ind_sub = [i for i in range(0, len(files_mat)) if files_mat[i].startswith(subj)]
#     nr = 1
#     plt.figure(figsize=(10, 10))
#     for i in ind_sub:
#         plt.subplot(6, 6, slice_nrs[i]+1)
#         plt.imshow(t2star_uncorr_reg[i].T, vmin=0, vmax=200)
#         plt.title(str(slice_nrs[i]))
#         plt.axis('off')
#         plt.tight_layout()
#         nr += 1
#     plt.suptitle(subj)
#     plt.show()

# divide into minor and stronger
ind_minor = []
for sub in ["SQ-struct-33", "SQ-struct-39", "SQ-struct-45"]:
    ind_minor.extend([i for i in ind if files_mat[i].startswith(sub)])
ind_stronger = []
for sub in ["SQ-struct-38", "SQ-struct-43", "SQ-struct-46"]:
    ind_stronger.extend([i for i in ind if files_mat[i].startswith(sub)])

print("Number of slices with minor motion: {}, "
      "number of slices with stronger motion: {}".format(
    len(ind_minor), len(ind_stronger))
)


""" 1) MAE of T2* values 
    1b) with registration to GT: """
MAE_T2s = {i: {} for i in ["stronger", "minor"]}
for ind_, motion_type in zip([ind_stronger, ind_minor], ["stronger", "minor"]):
    for mask, mask_name, col in zip([bm_noCSF, gm_seg, wm_seg],
                                    ["Brainmask without CSF", "Gray matter",
                                     "White matter"],
                                    ["tab:green", "tab:gray", "tab:blue"]):
        MAE_T2s[motion_type][mask_name] = {
            "uncorr": calc_masked_MAE(t2star_uncorr_reg[ind_],
                                      t2star_gt[ind_],
                                      mask[ind_]),
            "rdcorr": calc_masked_MAE(t2star_rdcorr_reg[ind_],
                                      t2star_gt[ind_],
                                      mask[ind_])
        }

        for d in t2star_corr.keys():
            num = int(d.replace("T2StarMotionCorrection", ""))
            for a in t2star_corr[d].keys():
                descr = a.replace("-", "") + "-" + str(num)
                MAE_T2s[motion_type][mask_name][descr] = calc_masked_MAE(
                    t2star_corr_reg[d][a][ind_], t2star_gt[ind_], mask[ind_]
                )

# violin plot:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
else:
    sorted_keys = sorted(MAE_T2s["stronger"]["Gray matter"].keys(), key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("MAE T2s ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, MAE_T2s[motion_type][mask_name]
        )

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=MAE_T2s,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[15, 7.5],
                  ylim=[2.8, 20],
                  ylabel="MAE [ms]",
                  bar_top=[True, True],
                  leg_loc=False,
                  save=False)


""" 1) SSIM of T2* maps 
    1b) with registration to GT: """
SSIM_T2s = {i: {} for i in ["stronger", "minor"]}
for ind_, motion_type in zip([ind_stronger, ind_minor], ["stronger", "minor"]):
    for mask, mask_name, col in zip([bm_noCSF, gm_seg, wm_seg],
                                    ["Brainmask without CSF", "Gray matter",
                                     "White matter"],
                                    ["tab:green", "tab:gray", "tab:blue"]):
        SSIM_T2s[motion_type][mask_name] = {
            "uncorr": calc_masked_SSIM(t2star_uncorr_reg[ind_],
                                       t2star_gt[ind_],
                                       mask[ind_]),
            "rdcorr": calc_masked_SSIM(t2star_rdcorr_reg[ind_],
                                       t2star_gt[ind_],
                                       mask[ind_])}

        for d in t2star_corr.keys():
            num = int(d.replace("T2StarMotionCorrection", ""))
            for a in t2star_corr[d].keys():
                descr = a.replace("-", "") + "-" + str(num)
                SSIM_T2s[motion_type][mask_name][descr] = calc_masked_SSIM(
                    t2star_corr_reg[d][a][ind_], t2star_gt[ind_], mask[ind_]
                )

# violin plot:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
else:
    sorted_keys = sorted(SSIM_T2s["stronger"]["Gray matter"].keys(), key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("SSIM T2s ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, SSIM_T2s[motion_type][mask_name]
        )

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=SSIM_T2s,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[0.5, 0.46],
                  ylim=[0.2, 0.87],
                  ylabel="SSIM",
                  bar_top=[False, False],
                  leg_loc=False,
                  save=False)


""" 2) average gray and white matter values
    2b) accuracy"""
# calculate regional averages of T2* maps:
T2s_av = {i: {} for i in ["stronger", "minor"]}
for mask, mask_name in zip([gm_seg, wm_seg],
                           ["Gray matter", "White matter"]):
    for ind_, motion_type in zip([ind_stronger, ind_minor],
                                 ["stronger", "minor"]):
        T2s_av[motion_type][mask_name] = {
            "motion-free": calc_regional_average(t2star_gt[ind_],
                                                 mask[ind_]),
            "uncorr": calc_regional_average(t2star_uncorr_reg[ind_],
                                            mask[ind_]),
            "rdcorr": calc_regional_average(t2star_rdcorr_reg[ind_],
                                            mask[ind_])
        }

        for d in t2star_corr.keys():
            num = int(d.replace("T2StarMotionCorrection", ""))
            for a in t2star_corr[d].keys():
                descr = a.replace("-", "") + "-" + str(num)
                T2s_av[motion_type][mask_name][descr] = calc_regional_average(
                    t2star_corr_reg[d][a][ind_], mask[ind_]
                )

# violin plot:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
    sorted_keys.append("motion-free")
else:
    sorted_keys = sorted(T2s_av["stronger"]["Gray matter"].keys(),
                         key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("T2s Averages ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, T2s_av[motion_type][mask_name]
        )

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=T2s_av,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[61, 58],
                  ylim=[42, 67],
                  ylabel="T2* average [ms]",
                  bar_top=[True, True],
                  leg_loc=False,
                  save=False)


""" 2) average gray and white matter values
    2b) precision (coefficient of variability)"""
T2s_cv = {i: {} for i in ["stronger", "minor"]}
for mask, mask_name in zip([gm_seg, wm_seg],
                           ["Gray matter", "White matter"]):
    for ind_, motion_type in zip([ind_stronger, ind_minor],
                                 ["stronger", "minor"]):
        T2s_cv[motion_type][mask_name] = {
            "motion-free": calc_regional_coeff_var(t2star_gt[ind_],
                                                   mask[ind_]),
            "uncorr": calc_regional_coeff_var(t2star_uncorr_reg[ind_],
                                              mask[ind_]),
            "rdcorr": calc_regional_coeff_var(t2star_rdcorr_reg[ind_],
                                              mask[ind_])}

        for d in t2star_corr.keys():
            num = int(d.replace("T2StarMotionCorrection", ""))
            for a in t2star_corr[d].keys():
                descr = a.replace("-", "") + "-" + str(num)
                T2s_cv[motion_type][mask_name][descr] = calc_regional_coeff_var(
                    t2star_corr_reg[d][a][ind_], mask[ind_]
                )

# violin plot:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
    sorted_keys.append("motion-free")
else:
    sorted_keys = sorted(T2s_cv["stronger"]["Gray matter"].keys(),
                         key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("T2s variation ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, T2s_cv[motion_type][mask_name]
        )

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=T2s_cv,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[0.5, 0.4],
                  ylim=[0.05, 0.6],
                  ylabel="T2* std/mean",
                  bar_top=[True, True],
                  leg_loc=False,
                  save=False)


""" Example T2* maps """
plot = False
if plot:
    # stronger motion examples:
    # i = 39 is also a good sample!!!

    for i in [105, 39]: #[105, 106]: #[51, 50, 17]:
        i_str = np.where(np.array(ind_stronger)==i)[0][0]
        plt.imshow(t2star_uncorr_reg[i].T, vmin=0, vmax=200)
        #plt.title("Motion-corrupted\n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['stronger']['Gray matter']['uncorr'][i_str], 1),
            np.round(MAE_T2s['stronger']['White matter']['uncorr'][i_str], 1)
        ), color="w", fontsize=17)
        plt.show()

        plt.imshow(
            t2star_corr_reg['T2StarMotionCorrection1000']['moco_mean'][i].T,
            vmin=0, vmax=200
        )
        #plt.title("Mean of 1000 \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['stronger']['Gray matter']['moco_mean-1000'][i_str], 1),
            np.round(MAE_T2s['stronger']['White matter']['moco_mean-1000'][i_str], 1)
        ), color="w", fontsize=17)
        plt.show()

        plt.imshow(
            t2star_corr_reg['T2StarMotionCorrection1000']['moco_best-'][i].T,
            vmin=0, vmax=200
        )
        #plt.title("PHIMO \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['stronger']['Gray matter']['moco_best-1000'][i_str], 1),
            np.round(MAE_T2s['stronger']['White matter']['moco_best-1000'][i_str], 1)
        ), color="w", fontsize=17
                 )
        plt.show()

        plt.imshow(t2star_rdcorr_reg[i].T, vmin=0, vmax=200)
        #plt.title("Correction with HR & Q\n(acqu. 6 min 25s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['stronger']['Gray matter']['rdcorr'][i_str], 1),
            np.round(MAE_T2s['stronger']['White matter']['rdcorr'][i_str], 1)
        ), color="w", fontsize=17
                 )
        plt.show()

        plt.imshow(t2star_gt[i].T, vmin=0, vmax=200)
        #plt.title("Motion-free \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # minor motion examples:
    for i in [13]:
        i_str = np.where(np.array(ind_minor)==i)[0][0]
        plt.imshow(t2star_uncorr_reg[i].T, vmin=0, vmax=200)
        #plt.title("Motion-corrupted\n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['minor']['Gray matter']['uncorr'][i_str], 1),
            np.round(MAE_T2s['minor']['White matter']['uncorr'][i_str], 1)
        ), color="w", fontsize=17
                 )
        plt.show()

        plt.imshow(t2star_corr_reg['T2StarMotionCorrection1000']['moco_mean'][i].T, vmin=0, vmax=200)
        #plt.title("Mean of 1000 \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['minor']['Gray matter']['moco_mean-1000'][i_str], 1),
            np.round(MAE_T2s['minor']['White matter']['moco_mean-1000'][i_str], 1)
        ), color="w", fontsize=17
                 )
        plt.show()

        plt.imshow(t2star_corr_reg['T2StarMotionCorrection1000']['moco_best-'][i].T,
                   vmin=0,
                   vmax=200)
        #plt.title("PHIMO \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['minor']['Gray matter']['moco_best-1000'][i_str], 1),
            np.round(MAE_T2s['minor']['White matter']['moco_best-1000'][i_str], 1)
        ), color="w", fontsize=17
                 )
        plt.show()

        plt.imshow(t2star_rdcorr_reg[i].T, vmin=0, vmax=200)
        #plt.title("Correction with HR & Q\n(acqu. 6 min 25s)")
        plt.axis('off')
        plt.tight_layout()
        plt.text(1, 12, "GM: {} ms\nWM: {} ms".format(
            np.round(MAE_T2s['minor']['Gray matter']['rdcorr'][i_str], 1),
            np.round(MAE_T2s['minor']['White matter']['rdcorr'][i_str], 1)
        ), color="w", fontsize=17
                 )
        plt.show()

        plt.imshow(t2star_gt[i].T, vmin=0, vmax=200)
        #plt.title("Motion-free \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    # compare 1 subject with and without registration
    for i in [13]: #[105]: #[51, 50, 17]:
        plt.imshow(t2star_uncorr_reg[i].T, vmin=0, vmax=200)
        #plt.title("Motion-corrupted\n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_rdcorr_reg[i].T, vmin=0, vmax=200)
        #plt.title("Correction with HR & Q\n(acqu. 6 min 25s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_corr_reg['T2StarMotionCorrection1000']['moco_mean'][i].T,
                   vmin=0, vmax=200)
        #plt.title("Mean of 1000 \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_corr_reg['T2StarMotionCorrection1000']['moco_best-'][i].T,
                   vmin=0, vmax=200)
        #plt.title("PHIMO \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_gt[i].T, vmin=0, vmax=200)
        #plt.title("Motion-free \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_uncorr[i].T, vmin=0, vmax=200)
        #plt.title("Motion-corrupted\n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


        plt.imshow(t2star_corr['T2StarMotionCorrection1000']['moco_mean'][i].T,
                   vmin=0, vmax=200)
        #plt.title("Mean of 1000 \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_corr['T2StarMotionCorrection1000']['moco_best-'][i].T,
                   vmin=0, vmax=200)
        #plt.title("PHIMO \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_rdcorr[i].T, vmin=0, vmax=200)
        #plt.title("Correction with HR & Q\n(acqu. 6 min 25s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.imshow(t2star_gt[i].T, vmin=0, vmax=200)
        #plt.title("Motion-free \n(acqu. 3 min 39s)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


""" Look at image quality metrics: """
""" 1) SSIM and PSNR of T2* weighted images 
    1b) with registration to GT: """
SSIM_imgs = {i: {} for i in ["stronger", "minor"]}
PSNR_imgs = {i: {} for i in ["stronger", "minor"]}
for ind_, motion_type in zip([ind_stronger, ind_minor], ["stronger", "minor"]):
    for mask, mask_name, col in zip([bm_noCSF, gm_seg, wm_seg],
                                    ["Brainmask without CSF", "Gray matter",
                                     "White matter"],
                                    ["tab:green", "tab:gray", "tab:blue"]):
        SSIM_imgs[motion_type][mask_name] = {
            "uncorr": calc_masked_SSIM_4D(img_uncorr_reg[ind_], img_gt[ind_],
                                          mask[ind_]),
            "rdcorr": calc_masked_SSIM_4D(img_rdcorr_reg[ind_], img_gt[ind_],
                                          mask[ind_])
        }
        PSNR_imgs[motion_type][mask_name] = {
            "uncorr": calc_masked_PSNR_4D(img_uncorr_reg[ind_], img_gt[ind_],
                                          mask[ind_]),
            "rdcorr": calc_masked_PSNR_4D(img_rdcorr_reg[ind_], img_gt[ind_],
                                          mask[ind_])
        }

        for d in t2star_corr.keys():
            num = int(d.replace("T2StarMotionCorrection", ""))
            for a in t2star_corr[d].keys():
                descr = a.replace("-", "") + "-" + str(num)
                SSIM_imgs[motion_type][mask_name][descr] = calc_masked_SSIM_4D(
                    img_corr_reg[d][a][ind_], img_gt[ind_], mask[ind_])
                PSNR_imgs[motion_type][mask_name][descr] = calc_masked_PSNR_4D(
                    img_corr_reg[d][a][ind_], img_gt[ind_], mask[ind_])

# violin plot SSIM:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
else:
    sorted_keys = sorted(SSIM_imgs["stronger"]["Gray matter"].keys(),
                         key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("MAE T2s ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, SSIM_imgs[motion_type][mask_name])

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=SSIM_imgs,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[0.5, 0.4],
                  ylim=[0.3, 1.0],
                  ylabel="SSIM",
                  bar_top=[False, False],
                  leg_loc=False,
                  save=False)

# violin plot PSNR:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
else:
    sorted_keys = sorted(PSNR_imgs["stronger"]["Gray matter"].keys(),
                         key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("MAE T2s ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, PSNR_imgs[motion_type][mask_name])

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=PSNR_imgs,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[30, 31],
                  ylim=[12, 35],
                  ylabel="PSNR",
                  bar_top=[True, True],
                  leg_loc=False,
                  save=False)



""" 1) SSIM and PSNR of T2* weighted images (later echoes)
    1b) with registration to GT: """
SSIM_imgs_l3e = {i: {} for i in ["stronger", "minor"]}
PSNR_imgs_l3e = {i: {} for i in ["stronger", "minor"]}
for ind_, motion_type in zip([ind_stronger, ind_minor], ["stronger", "minor"]):
    for mask, mask_name, col in zip([bm_noCSF, gm_seg, wm_seg],
                                    ["Brainmask without CSF", "Gray matter",
                                     "White matter"],
                                    ["tab:green", "tab:gray", "tab:blue"]):
        SSIM_imgs_l3e[motion_type][mask_name] = {
            "uncorr": calc_masked_SSIM_4D(img_uncorr_reg[ind_], img_gt[ind_],
                                          mask[ind_], av_echoes=False,
                                          later_echoes=3),
            "rdcorr": calc_masked_SSIM_4D(img_rdcorr_reg[ind_], img_gt[ind_],
                                          mask[ind_], av_echoes=False,
                                          later_echoes=3)}
        PSNR_imgs_l3e[motion_type][mask_name] = {
            "uncorr": calc_masked_PSNR_4D(img_uncorr_reg[ind_], img_gt[ind_],
                                          mask[ind_], av_echoes=False,
                                          later_echoes=3),
            "rdcorr": calc_masked_PSNR_4D(img_rdcorr_reg[ind_], img_gt[ind_],
                                          mask[ind_], av_echoes=False,
                                          later_echoes=3)}

        for d in t2star_corr.keys():
            num = int(d.replace("T2StarMotionCorrection", ""))
            for a in t2star_corr[d].keys():
                descr = a.replace("-", "") + "-" + str(num)
                SSIM_imgs_l3e[motion_type][mask_name][descr] = calc_masked_SSIM_4D(
                    img_corr_reg[d][a][ind_], img_gt[ind_],
                    mask[ind_], av_echoes=False, later_echoes=3)
                PSNR_imgs_l3e[motion_type][mask_name][descr] = calc_masked_PSNR_4D(
                    img_corr_reg[d][a][ind_], img_gt[ind_],
                    mask[ind_], av_echoes=False, later_echoes=3)

# violin plot SSIM:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
else:
    sorted_keys = sorted(SSIM_imgs_l3e["stronger"]["Gray matter"].keys(),
                         key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("SSIM_imgs_l3e ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, SSIM_imgs_l3e[motion_type][mask_name])

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=SSIM_imgs_l3e,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[0.5, 0.4],
                  ylim=[0.2, 0.95],
                  ylabel="SSIM",
                  bar_top=[False, False],
                  leg_loc=False,
                  save=False)

# violin plot PSNR:
if 'sorted_keys' in config_file.keys():
    sorted_keys = config_file['sorted_keys'].copy()
else:
    sorted_keys = sorted(PSNR_imgs_l3e["stronger"]["Gray matter"].keys(),
                         key=sort_key)

combinations, p_values = ({k: {} for k in ["stronger", "minor"]},
                          {k: {} for k in ["stronger", "minor"]})
for motion_type in ["stronger", "minor"]:
    for mask_name in ["Gray matter", "White matter"]:
        print("PSNR_imgs_l3e ", motion_type, mask_name)
        (combinations[motion_type][mask_name],
         p_values[motion_type][mask_name]) = statistical_testing(
            sorted_keys, PSNR_imgs_l3e[motion_type][mask_name])

make_violin_plots(motion_types=["stronger", "minor"],
                  keys=sorted_keys,
                  masks=["Gray matter", "White matter"],
                  metric_dict=PSNR_imgs_l3e,
                  p_vals=p_values,
                  combs=combinations,
                  bar_values=[21, 20],
                  ylim=[13, 34],
                  ylabel="PSNR",
                  bar_top=[False, False],
                  leg_loc=False,
                  save=False)
