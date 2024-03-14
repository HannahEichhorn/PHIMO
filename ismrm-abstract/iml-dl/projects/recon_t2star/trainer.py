import os.path
from core.Trainer import Trainer
from time import time
import wandb
import logging
import numpy as np
from torchinfo import summary
from projects.recon_t2star.utils import *


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(
            training_params, model, data, device, log_wandb)

        self.loss_domain = training_params.get('loss_domain', 'I')
        self.mask_image = training_params.get('mask_image', False)

        for s in self.train_ds:
            input_size = [
                s[0].numpy().shape, s[2].numpy().shape,
                s[2].numpy().shape, s[3].numpy().shape
            ]
            break

        dtypes = [torch.complex64, torch.complex64, torch.complex64,
                  torch.complex64]
        print(f'Input size of summary is: {input_size}')
        summary(model, input_size, dtypes=dtypes)

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Trains the local client with the option to initialize the model and
        optimizer states.

        Parameters
        ----------
        model_state : dict, optional
            Weights of the global model. If provided, the local model is
            initialized with these weights.
        opt_state : dict, optional
            State of the optimizer. If provided, the local optimizer is
            initialized with this state.
        start_epoch : int, optional
            Starting epoch for training.

        Returns
        -------
        dict
            The state dictionary of the trained local model.
        """

        if model_state is not None:
            self.model.load_state_dict(model_state)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        self.early_stop = False
        self.model.train()

        for epoch in range(self.training_params['nr_epochs']):
            print('Epoch: ', epoch)
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info(
                    "[Trainer::test]: ################ Finished  training "
                    "(early stopping) ################"
                )
                break

            start_time = time()
            batch_loss = {"combined": 0, "component_1": 0, "component_2": 0}
            count_images = 0

            for data in self.train_ds:
                (img_cc_zf, kspace_zf, mask, sens_maps,
                 img_cc_fs, brain_mask, A) = process_input_data(
                    self.device, data
                )
                count_images += img_cc_zf.shape[0]

                # Forward Pass
                self.optimizer.zero_grad()
                prediction = self.model(img_cc_zf, kspace_zf, mask, sens_maps)

                # Reconstruction Loss
                mask = (brain_mask[:, None].repeat(1, 12, 1, 1)
                        if self.mask_image else None)
                if self.loss_domain == 'I':
                    losses = self.criterion_rec(img_cc_fs, prediction,
                                                output_components=True,
                                                mask=mask)
                elif self.loss_domain == 'k':
                    kspace_pred = A(prediction, torch.ones_like(mask),
                                    sens_maps)
                    kspace_fs = A(img_cc_fs, torch.ones_like(mask),
                                  sens_maps)
                    losses = self.criterion_rec(kspace_fs, kspace_pred,
                                                output_components=True,
                                                mask=mask)
                else:
                    logging.info(
                        "[Trainer::train]: This loss domain is not "
                        "implemented."
                    )

                # Backward Pass
                losses[0].backward()
                self.optimizer.step()
                batch_loss = self._update_batch_loss(
                    batch_loss, losses, img_cc_zf.size(0)
                )

            self._track_epoch_loss(epoch, batch_loss, start_time, count_images)

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val',
                      self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None,
             epoch=0):
        """
        Tests the local client.

        Parameters
        ----------
        model_weights : dict
            Weights of the global model.
        test_data : DataLoader
            Test data for evaluation.
        task : str, optional
            Task identifier (default is 'Val').
        opt_weights : dict, optional
            Optimal weights (default is None).
        epoch : int, optional
            Current epoch number (default is 0).
        """

        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {task + '_loss_': 0}
        test_total = 0

        val_image_available = False
        with torch.no_grad():
            for data in test_data:
                # Input
                (img_cc_zf, kspace_zf, mask, sens_maps,
                 img_cc_fs, brain_mask, A) = process_input_data(self.device,
                                                                data)
                filename, slice_num = data[5:]
                test_total += img_cc_zf.shape[0]

                # Forward Pass
                prediction = self.test_model(img_cc_zf, kspace_zf, mask,
                                             sens_maps)
                prediction_track = prediction.clone().detach()
                gt_track = img_cc_fs.clone().detach()
                zf_track = img_cc_zf.clone().detach()

                # Reconstruction Loss
                mask = (brain_mask[:, None].repeat(1, 12, 1, 1)
                        if self.mask_image else None)
                if self.loss_domain == 'I':
                    loss_ = self.criterion_rec(img_cc_fs, prediction,
                                               output_components=False,
                                               mask=mask)
                elif self.loss_domain == 'k':
                    kspace_pred = A(prediction, torch.ones_like(mask),
                                    sens_maps)
                    kspace_fs = A(img_cc_fs, torch.ones_like(mask),
                                  sens_maps)
                    loss_ = self.criterion_rec(kspace_fs, kspace_pred,
                                               output_components=False,
                                               mask=mask)
                else:
                    logging.info(
                        "[Trainer::test]: This loss domain is not "
                        "implemented."
                    )
                metrics[task + '_loss_'] += loss_.item() * img_cc_zf.size(0)

                if task == 'Val':
                    search_string = ('SQ-struct-40_nr_09082023_1643103_4_2_'
                                     'wip_t2s_air_sg_fV4.mat_15')
                    (gt_, prediction_,
                     zf_, val_image_available) = self._find_validation_image(
                        search_string, filename, slice_num, prediction_track,
                        gt_track, zf_track
                    )

            if task == 'Val':
                if not val_image_available:
                    print('[Trainer - test] ERROR: No validation image can be '
                          'tracked, since the required filename is '
                          'not in the validation set.\nUsing the last '
                          'available example instead')
                    gt_ = gt_track[0]
                    prediction_ = prediction_track[0]
                    zf_ = zf_track[0]

                log_images_to_wandb(
                    prepare_for_logging(prediction_),
                    prepare_for_logging(gt_),
                    prepare_for_logging(zf_),
                    logging=task
                )

            for metric_key in metrics.keys():
                metric_name = task + '/' + str(metric_key)
                metric_score = metrics[metric_key] / test_total
                wandb.log({metric_name: metric_score, '_step_': epoch})
            wandb.log({
                'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch}
            )

            if task == 'Val':
                epoch_val_loss = metrics[task + '_loss_'] / test_total
                print(
                    'Epoch: {} \tValidation Loss: {:.6f} , computed for {} '
                    'samples'.format(epoch, epoch_val_loss, test_total)
                )
                if epoch_val_loss < self.min_val_loss:
                    self.min_val_loss = epoch_val_loss
                    torch.save({'model_weights': model_weights,
                                'optimizer_weights': opt_weights,
                                'epoch': epoch},
                               self.client_path + '/best_model.pt')
                    self.best_weights = model_weights
                    self.best_opt_weights = opt_weights
                self.early_stop = self.early_stopping(epoch_val_loss)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch_val_loss)

    def _update_batch_loss(self, batch_loss, losses, batch_size):
        """Update batch losses with current losses"""

        batch_loss["combined"] += losses[0].item() * batch_size
        batch_loss["component_1"] += losses[1].item() * batch_size
        batch_loss["component_2"] += losses[2].item() * batch_size
        return batch_loss


    def _track_epoch_loss(self, epoch, batch_loss, start_time, count_images):
        """Track and log epoch loss."""

        epoch_loss = {
            "combined": (batch_loss["combined"] / count_images
                         if count_images > 0 else batch_loss["combined"]),
            "component_1": (batch_loss["component_1"] / count_images
                            if count_images > 0 else batch_loss["component_1"]),
            "component_2": (batch_loss["component_2"] / count_images
                            if count_images > 0 else batch_loss["component_2"])
        }

        end_time = time()
        loss_msg = (
            'Epoch: {} \tTraining Loss: {:.6f} , computed in {} '
            'seconds for {} samples').format(
            epoch, epoch_loss["combined"], end_time - start_time,
            count_images
        )
        print(loss_msg)

        wandb.log({"Train/Loss_": epoch_loss["combined"], '_step_': epoch})
        wandb.log(
            {"Train/Loss_Comp1_": epoch_loss["component_1"], '_step_': epoch})
        wandb.log(
            {"Train/Loss_Comp2_": epoch_loss["component_2"], '_step_': epoch})

    def _find_validation_image(self, search_string, filename, slice_num,
                               prediction_track, gt_track, zf_track):
        """Search for a specific validation image and retrieve the data."""
        # Search for a specific validation image
        search = [os.path.basename(f) + '_' + str(s.numpy()) for f, s in
                  zip(filename, slice_num)]

        # Check if the search string is in the list
        try:
            ind = search.index(search_string)
            val_image_available = True
        except ValueError:
            ind = -1
            val_image_available = False

        # Retrieve data for the found image or provide default values
        gt_ = gt_track[ind] if val_image_available else torch.tensor([])
        prediction_ = prediction_track[
            ind] if val_image_available else torch.tensor([])
        zf_ = zf_track[ind] if val_image_available else torch.tensor([])

        return gt_, prediction_, zf_, val_image_available
