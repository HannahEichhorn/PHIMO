import logging
import os.path
import numpy as np
import torch.nn
import wandb
import merlinth
from time import time
from dl_utils import *
from core.DownstreamEvaluator import DownstreamEvaluator
from data.t2star_loader import RawMotionBootstrapSamples
from projects.recon_t2star.utils import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """Downstream Tasks"""

    def __init__(self,
                 name,
                 model,
                 device,
                 test_data_dict,
                 checkpoint_path,
                 task="recon",
                 aggregation=None,
                 include_brainmask=False,
                 nr_bootstrap_samples=False,
                 save_predictions=False):
        super(PDownstreamEvaluator, self).__init__(
            name, model, device, test_data_dict, checkpoint_path
        )

        self.task = task
        if aggregation is None:
            self.aggregation = ["mean"]
        else:
            self.aggregation = aggregation
        self.include_brainmask = include_brainmask
        self.nr_bootstrap_samples = nr_bootstrap_samples
        self.save_predictions = save_predictions

        self._check_input_arguments()

    def start_task(self, global_model):
        """Function to perform analysis after training is finished."""

        if self.task == "recon":
            self.test_reconstruction(global_model)
        elif self.task == "moco":
            self.test_moco(global_model)
        else:
            logging.info("[DownstreamEvaluator::ERROR]: This task is not "
                         "implemented.")

    def test_reconstruction(self, global_model):
        """Validation of reconstruction downstream task."""

        logging.info("################ Reconstruction test #################")
        self.model.load_state_dict(global_model)
        self.model.eval()

        keys = ["SSIM_magn_pred", "SSIM_magn_zf", "PSNR_magn_pred", "PSNR_magn_zf",
                "MSE_magn_pred", "MSE_magn_zf", "SSIM_phase_pred", "SSIM_phase_zf",
                "PSNR_phase_pred", "PSNR_phase_zf", "MSE_phase_pred", "MSE_phase_zf"]
        metrics = {k: [] for k in keys}

        for dataset_key in self.test_data_dict.keys():
            logging.info('DATASET: {}'.format(dataset_key))
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {k: [] for k in metrics.keys()}

            for idx, data in enumerate(dataset):
                with torch.no_grad():
                    (img_cc_zf, kspace_zf, mask, sens_maps,
                     img_cc_fs, brain_mask, A) = process_input_data(
                        self.device, data
                    )

                    prediction = self.model(img_cc_zf, kspace_zf, mask,
                                            sens_maps)

                    for i in range(len(prediction)):
                        count = str(idx * len(prediction) + i)

                        metrics_pred = calculate_img_metrics(
                            target=img_cc_fs[i],
                            data=prediction[i],
                            bm=brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=[k for k in test_metrics.keys()
                                                if "pred" in k],
                            include_brainmask=self.include_brainmask)

                        metrics_zf = calculate_img_metrics(
                            target=img_cc_fs[i],
                            data=img_cc_zf[i],
                            bm=brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=[k for k in test_metrics.keys()
                                                if "zf" in k],
                            include_brainmask=self.include_brainmask)

                        test_metrics = update_metrics_dict(metrics_pred,
                                                           test_metrics)
                        test_metrics = update_metrics_dict(metrics_zf,
                                                           test_metrics)

                        if idx % 2 == 0 and i % 10 == 0:
                            log_images_to_wandb(
                                prepare_for_logging(prediction[i]),
                                prepare_for_logging(img_cc_fs[i]),
                                prepare_for_logging(img_cc_zf[i]),
                                logging=dataset_key,
                                logging_2=count
                            )

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(
                    metric,
                    np.nanmean(test_metrics[metric]),
                    np.nanstd(test_metrics[metric]))
                )
                metrics[metric].append(test_metrics[metric])

    def test_moco(self, global_model):
        """Validation of motion correction downstream task."""

        logging.info("################ MoCo test (Bootstrap with random masks)"
                     " #################")
        self.model.load_state_dict(global_model)
        self.model.eval()

        if not self.nr_bootstrap_samples:
            print("[DownstreamEvaluator::ERROR]: Number of bootstrap samples"
                  " not specified")

        # Different metrics:
        keys = ["T2s_MAE_noCSF", "SSIM_magn", "PSNR_magn", "MSE_magn"]
        metrics = {}
        for descr in ["uncorr", "hrqrcorr"]:
            metrics[descr] = {k: [] for k in keys}
        for descr in self.aggregation:
            if descr.startswith("best-"):
                metrics["best-"] = {k: [] for k in keys}
            else:
                metrics[descr] = {k: [] for k in keys}

        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            test_metrics = {descr: {k: [] for k in metrics[descr].keys()}
                            for descr in metrics.keys()}

            for idx, data in enumerate(dataset):
                with ((torch.no_grad())):
                    self._process_motion_input_data(self.device, data)

                    self._run_bootstrap_aggregation()

                    if self.save_predictions:
                        if dataset_key in ["val", "test"]:
                            self._save_bootstrap_results(dataset_key)

                    self.t2star_gt = self._calculate_t2star_map(
                        self.img_gt, self.brain_mask
                    )
                    self.t2star_uncorr = self._calculate_t2star_map(
                        self.img_uncorr, self.brain_mask
                    )
                    self.t2star_hrqrcorr = self._calculate_t2star_map(
                        self.img_hrqrcorr, self.brain_mask
                    )
                    self.t2star_corr = {}
                    for agg in self.img_corr.keys():
                        self.t2star_corr[agg] = self._calculate_t2star_map(
                            self.img_corr[agg], self.brain_mask
                        )

                    for i in range(len(self.img_uncorr)):
                        count = str(idx * len(self.img_uncorr) + i)

                        self._register_to_gt(batch_nr=i)

                        img_metrics = [k for k in test_metrics["uncorr"].keys()
                                       if "T2s" not in k]
                        metrics_uncorr = calculate_img_metrics(
                            target=self.img_gt[i],
                            data=self.img_uncorr_reg,
                            bm=self.brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=img_metrics,
                            include_brainmask=self.include_brainmask
                        )
                        test_metrics["uncorr"] = update_metrics_dict(
                            metrics_uncorr, test_metrics["uncorr"]
                        )
                        metrics_hrqrcorr = calculate_img_metrics(
                            target=self.img_gt[i],
                            data=self.img_hrqrcorr_reg,
                            bm=self.brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=img_metrics,
                            include_brainmask=self.include_brainmask
                        )
                        test_metrics["hrqrcorr"] = update_metrics_dict(
                            metrics_hrqrcorr, test_metrics["hrqrcorr"]
                        )
                        for agg in self.img_corr.keys():
                            metrics_corr = calculate_img_metrics(
                                target=self.img_gt[i],
                                data=self.img_corr_reg[agg],
                                bm=self.brain_mask[i][None].repeat(12, 1, 1),
                                metrics_to_be_calc=img_metrics,
                                include_brainmask=self.include_brainmask
                            )
                            test_metrics[agg] = update_metrics_dict(
                                metrics_corr, test_metrics[agg]
                            )

                        t2s_metrics = [k for k in test_metrics["uncorr"].keys()
                                       if "T2s" in k]
                        metrics_uncorr = calculate_t2star_metrics(
                            target=self.t2star_gt[i],
                            data=self.t2star_uncorr_reg,
                            bm=self.brain_mask_noCSF[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=t2s_metrics,
                            include_brainmask=self.include_brainmask
                        )
                        test_metrics["uncorr"] = update_metrics_dict(
                            metrics_uncorr, test_metrics["uncorr"]
                        )
                        metrics_hrqrcorr = calculate_t2star_metrics(
                            target=self.t2star_gt[i],
                            data=self.t2star_hrqrcorr_reg,
                            bm=self.brain_mask_noCSF[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=t2s_metrics,
                            include_brainmask=self.include_brainmask
                        )
                        test_metrics["hrqrcorr"] = update_metrics_dict(
                            metrics_hrqrcorr, test_metrics["hrqrcorr"]
                        )
                        for agg in self.img_corr.keys():
                            metrics_corr = calculate_t2star_metrics(
                                target=self.t2star_gt[i],
                                data=self.t2star_corr_reg[agg],
                                bm=self.brain_mask_noCSF[i][None].repeat(12, 1, 1),
                                metrics_to_be_calc=t2s_metrics,
                                include_brainmask=self.include_brainmask
                            )
                            test_metrics[agg] = update_metrics_dict(
                                metrics_corr, test_metrics[agg]
                            )

                        # log a few example images in wandb
                        if idx % 2 == 0 and i % 2 == 0:  # plot some images
                            for agg in self.img_corr.keys():
                                log_images_to_wandb(
                                    prepare_for_logging(self.img_corr[agg][i]),
                                    prepare_for_logging(self.img_gt[i]),
                                    prepare_for_logging(self.img_uncorr[i]),
                                    mask_example=abs(
                                        detach_torch(self.aggr_masks[agg][i])
                                    ),
                                    logging_0=self.name,
                                    logging=agg+"_"+dataset_key,
                                    logging_2=count
                                )

                                # track quantitative parameter maps:
                                plot2wandb(
                                    self.t2star_uncorr_reg[i],
                                    self.t2star_hrqrcorr_reg[i],
                                    self.t2star_corr_reg[agg][i],
                                    self.t2star_gt[i, 0],
                                    ["Motion-corrupted",
                                     "Correction w\n HR \& QR",
                                     "PHIMO\n (Proposed)",
                                     "Motion-free"],
                                    "{}/Quantitative_MoCo_Examples_"
                                    "{}_{}/{}".format(agg,
                                                      self.name,
                                                      dataset_key,
                                                      str(count)))

                        # log example bootstrap masks for one slice:
                        if idx == 0 and i == 0:
                            for agg in self.img_corr.keys():
                                if agg in ['weighted', 'best-']:
                                    masks_examples = abs(
                                        detach_torch(self.masks[agg][i])
                                    )
                                    for n in range(masks_examples.shape[0]):
                                        mask_ = np.repeat(
                                            masks_examples[n][:, None],
                                            112, 1
                                        )
                                        mask_ = wandb.Image(
                                            np.swapaxes(mask_, -2, -1),
                                            caption=""
                                        )
                                        wandb.log({"{}/MoCo_Examples_Bootstrap"
                                                   "_Masks_{}_{}/Batch0_Slice0_"
                                                   "Nr{}".format(self.name,
                                                                 agg,
                                                                 dataset_key,
                                                                 n):
                                                   mask_}
                                                  )
                                aggr_masks_examples = abs(
                                    detach_torch(self.aggr_masks[agg][i])
                                )[:, None]
                                aggr_masks_examples = np.repeat(
                                    aggr_masks_examples,
                                    112, 1
                                )
                                aggr_mask_ = wandb.Image(
                                    np.swapaxes(aggr_masks_examples, -2, -1),
                                    caption=""
                                )
                                wandb.log({"{}/MoCo_Examples_Bootstrap_Masks"
                                           "_{}_{}/Batch0_Slice0_Aggreg. "
                                           "mask".format(self.name,
                                                         agg,
                                                         dataset_key):
                                               aggr_mask_}
                                          )

            for descr in test_metrics.keys():
                logging.info('##### {} #####'.format(descr))
                for metric in test_metrics[descr]:
                    logging.info(
                        '{} mean rounded: {} +/- {}, ''full: {} +/- {}'
                        .format(
                            metric,
                            np.round(np.nanmean(test_metrics[descr][metric]), 2),
                            np.round(np.nanstd(test_metrics[descr][metric]), 2),
                            np.nanmean(test_metrics[descr][metric]),
                            np.nanstd(test_metrics[descr][metric])
                        )
                    )
                    metrics[descr][metric].append(test_metrics[descr][metric])

        for descr in metrics.keys():
            for metric in metrics[descr]:
                for i, dataset in enumerate(self.test_data_dict.keys()):
                    folder = (self.checkpoint_path + "/" + self.name
                              + "/moco_" + descr + "/")
                    create_dir(folder)
                    np.savetxt(folder + metric + "_" + dataset + ".txt",
                               np.array(metrics[descr][metric][i]).T,
                               header="Metric values for metric and dataset "
                                      "specified in filename")

    def _check_input_arguments(self):
        if len([item for item in self.aggregation
                if item.startswith("best-")]
               ) > 1:
            print("ERROR: more than one best- sample cannot be processed. "
                  "Only processing first entry with best-.")

        if self.task == "moco":
            for agg in self.aggregation:
                create_dir("{}/{}/moco_{}".format(self.checkpoint_path,
                                                  self.name, agg))

            if ("mean" not in self.aggregation
                    and len([item for item in self.aggregation
                             if item.startswith("best-")]) == 0):
                logging.info("[DownstreamEvaluator::test_moco] ERROR: This "
                             "aggregation method is not implemented. "
                             "Using mean aggregation.")
                self.aggregation = ["mean"]

    def _process_motion_input_data(self, device, data):
        """Processes input data for training."""

        self.img_uncorr = data[0].to(device)
        self.sens_maps = data[1].to(device)
        self.img_gt = data[2].to(device)
        self.img_hrqrcorr = data[3].to(device)
        self.brain_mask = data[4].to(device)
        self.brain_mask_noCSF = data[5].to(device)
        self.filename, self.slice_num = data[6:8]
        self.random_mask = data[8]
        self.A = merlinth.layers.mri.MulticoilForwardOp(
            center=True,
            channel_dim_defined=False
        )

    def _initialize_bootstrap_results(self):
        """Initialize results for bootstrap aggregation."""

        self.img_corr, self.masks, self.aggr_masks = {}, {}, {}
        self.weights, self.predictions = {}, {}
        self.best_fit_errors, self.mean_fit_error = {}, {}

        if "mean" in self.aggregation:
            self.img_corr["mean"] = torch.zeros_like(self.img_uncorr)
            self.aggr_masks["mean"] = torch.zeros_like(
                self.img_uncorr[:, 0, :, 0]
            )

        if any(item.startswith("best-") for item in self.aggregation):
            n = int([item.replace("best-", "") for item in self.aggregation if
                     item.startswith("best-")][0])
            self.predictions["best-"] = torch.zeros(
                size=(self.img_uncorr.shape[0], n, *self.img_uncorr.shape[1:]),
                dtype=self.img_uncorr.dtype)
            self.masks["best-"] = torch.zeros(
                size=(self.img_uncorr.shape[0], n, self.img_uncorr.shape[2]),
                dtype=self.img_uncorr.dtype)
            self.best_fit_errors["best-"] = np.zeros((self.img_uncorr.shape[0],
                                                      n))
            self.mean_fit_error["best-"] = np.zeros(self.img_uncorr.shape[0])

    def _reconstruct_bootstrap_sample(self):
        """Reconstruct a single bootstrap sample using trained model."""

        BootstrapSampler = RawMotionBootstrapSamples(1,
                                                     self.random_mask)
        mask, img_cc_zf = BootstrapSampler.apply_random_masks(
            detach_torch(self.img_uncorr), detach_torch(self.sens_maps)
        )

        img_cc_zf = img_cc_zf[:, 0].to(self.device)
        mask = mask[:, 0].to(self.device)

        kspace_zf = self.A(img_cc_zf, mask, self.sens_maps)
        prediction = self.model(img_cc_zf, kspace_zf, mask, self.sens_maps)

        return mask, prediction

    def _collect_bootstrap_results(self, prediction, mask):
        """Collect results from a single bootstrap sample."""

        if "mean" in self.aggregation:
            self.img_corr["mean"] += prediction * 1 / self.nr_bootstrap_samples
            self.aggr_masks["mean"] += mask[:, 0, 0, :,
                                  0] * 1 / self.nr_bootstrap_samples

        if any(item.startswith("best-") for item in self.aggregation):
            FitError = T2starFit(
                detach_torch(prediction[:, None]),
                detach_torch(self.brain_mask_noCSF)
            )
            _, fit_error = FitError.weight_fit_error()

            # check if better fit:
            for b in range(0, prediction.shape[0]):
                ind_min = np.argmin(self.best_fit_errors["best-"][b])
                if fit_error[b] > self.best_fit_errors["best-"][b, ind_min]:
                    self.best_fit_errors["best-"][b, ind_min] = fit_error[b]
                    self.masks["best-"][b, ind_min] = mask[b, 0, 0, :, 0]
                    self.predictions["best-"][b, ind_min] = prediction[b]
                self.mean_fit_error["best-"][b] += fit_error[
                                         b, 0] / self.nr_bootstrap_samples

    def _aggregate_bootstrap_results(self):
        """Aggregate results from all bootstrap samples for "best-"."""

        self.weights["best-"] = normalize_values(self.best_fit_errors["best-"])

        self.img_corr["best-"] = torch.sum(
            self.predictions["best-"]
            * self.weights["best-"][:, :, None, None, None],
            dim=1
        ).to(self.device).to(self.predictions["best-"].dtype)

        self.aggr_masks["best-"] = torch.sum(
            self.masks["best-"]
            * self.weights["best-"][:, :, None],
            dim=1
        )

    def _run_bootstrap_aggregation(self):
        """Run bootstrap aggregation for motion correction downstream task."""

        start_time = time()
        self._initialize_bootstrap_results()


        for i in range(self.nr_bootstrap_samples):
            mask, prediction = self._reconstruct_bootstrap_sample()

            self._collect_bootstrap_results(prediction, mask)

        if any(item.startswith("best-") for item in self.aggregation):
            self._aggregate_bootstrap_results()

        end_time = time()
        logging.info(
            "Bootstrap aggregation took {} seconds, corresponding to {} seconds per slice.".format(
                end_time - start_time,
                (end_time - start_time) / self.img_uncorr.shape[0]))

    def _save_bootstrap_results(self, dataset_key):
        """Save results from bootstrap aggregation."""

        for b in range(0, self.img_uncorr.shape[0]):
            ind_subj = os.path.basename(self.filename[b]).find("SQ-struct")
            subj = os.path.basename(self.filename[b])[ind_subj: ind_subj + 12]

            if "mean" in self.aggregation:
                base_path = os.path.join(self.checkpoint_path,
                                         self.name,
                                         "moco_mean")
                folder_masks = os.path.join(base_path,
                                            f"{dataset_key}_masks",
                                            subj)
                folder_imgs = os.path.join(base_path,
                                           f"{dataset_key}_predictions",
                                           subj)
                create_dir(folder_masks)
                create_dir(folder_imgs)

                base_filename = os.path.basename(self.filename[b])[:-4]
                slice_filename = f"{base_filename}_slice-{self.slice_num[b]}.npy"

                np.save(os.path.join(folder_imgs, slice_filename),
                        detach_torch(self.img_corr["mean"][b]))
                np.save(os.path.join(folder_masks, slice_filename),
                        detach_torch(self.aggr_masks["mean"][b]))

            if any(item.startswith("best-") for item in self.aggregation):
                agg = "best-"
                base_path = os.path.join(self.checkpoint_path,
                                         self.name,
                                         f"moco_{agg}")
                folder_masks = os.path.join(base_path,
                                            f"{dataset_key}_best_masks",
                                            subj)
                folder_imgs = os.path.join(base_path,
                                           f"{dataset_key}_best_images",
                                           subj)
                folder_errors = os.path.join(base_path,
                                             f"{dataset_key}_best_fit_errors",
                                             subj)
                create_dir(folder_masks)
                create_dir(folder_imgs)
                create_dir(folder_errors)

                base_filename = os.path.basename(self.filename[b])[:-4]
                slice_filename = f"{base_filename}_slice-{self.slice_num[b]}.npy"
                error_filename = f"{base_filename}_slice-{self.slice_num[b]}.txt"

                np.save(os.path.join(folder_imgs, slice_filename),
                        detach_torch(self.predictions["best-"][b]))
                np.save(os.path.join(folder_masks, slice_filename),
                        detach_torch(self.masks["best-"][b]))
                np.savetxt(os.path.join(folder_errors, error_filename),
                           self.best_fit_errors["best-"][b])

    def _calculate_t2star_map(self, img, bm):
        """Calculate T2* maps from complex-valued T2*-weighted images."""

        FitError = T2starFit(detach_torch(img[:, None]),
                             detach_torch(bm))
        t2star, _ = FitError.t2star_linregr()

        return t2star

    def _register_to_gt(self, batch_nr):
        """Register magnitude images and t2* maps to ground truth."""

        self.img_uncorr_reg, self.t2star_uncorr_reg = rigid_registration(
            abs(self.img_gt[batch_nr])[0],
            abs(self.img_uncorr[batch_nr])[0],
            abs(self.img_uncorr)[batch_nr],
            self.t2star_uncorr[batch_nr])

        self.img_hrqrcorr_reg, self.t2star_hrqrcorr_reg = rigid_registration(
            abs(self.img_gt[batch_nr])[0],
            abs(self.img_hrqrcorr[batch_nr])[0],
            abs(self.img_hrqrcorr)[batch_nr],
            self.t2star_hrqrcorr[batch_nr])

        self.img_corr_reg, self.t2star_corr_reg = {}, {}
        for agg in self.img_corr.keys():
            self.img_corr_reg[agg], self.t2star_corr_reg[
                agg] = rigid_registration(
                abs(self.img_gt[batch_nr])[0],
                abs(self.img_corr[agg][batch_nr])[0],
                abs(self.img_corr[agg])[batch_nr],
                self.t2star_corr[agg][batch_nr])
