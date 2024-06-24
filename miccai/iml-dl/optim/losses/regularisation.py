import torch
import logging
import numpy as np


class EnforceDiverseDCWeights:
    """ Regulariser to enforce diverse DC weights for different
    acceleration rates. """

    def __init__(self):
        super(EnforceDiverseDCWeights, self).__init__()

    def __call__(self, model, device, min_acc=0.05, max_acc=0.5):
        """
        Compute the difference between the predicted weights of the
        DC component for different exclusion rates.

        Parameters
        ----------
        weights : torch.Tensor
            Weights tensor.

        Returns
        -------
        loss : torch.Tensor
            EnforceDiverseDCWeights loss for the given weights.
        """

        exclusion_rates = sorted([np.random.uniform(min_acc, max_acc)
                                  for _ in range(12)])
        dc_weights = []
        for e in exclusion_rates:
            excl_rate = torch.tensor([e], device=device).to(
                torch.float32)
            dc_weights.append([
                model.predict_parameters(hyper_input=excl_rate)[k]
                for k in model.predict_parameters(hyper_input=excl_rate).keys()
                if 'DC' in k
            ])

        dc_weights = torch.stack([
            torch.stack(sublist) for sublist in dc_weights
        ])
        squared_diff = torch.sum(
            torch.abs(dc_weights[0:6] - dc_weights[6:]) ** 2,
            dim=1
        )
        max_weight = torch.amax(torch.cat([
            torch.abs(dc_weights[0:6]) ** 2,
            torch.abs(dc_weights[6:]) ** 2], dim=1), dim=1)

        return 1 - torch.mean(squared_diff / (max_weight * dc_weights.shape[1]))


class EnforcedSmallExclRates:
    """ Regulariser to enforce small exclusion rates. """

    def __init__(self):
        super(EnforcedSmallExclRates, self).__init__()

    def __call__(self, mask):

        return torch.mean(1 - torch.sum(mask, dim=1) / mask.shape[1])


class MaskVariabilityAcrossSlices:
    """  Regulariser to penalise variability of the masks across
    adjacent slices.
    """

    def __init__(self):
        super(MaskVariabilityAcrossSlices, self).__init__()


    def __call__(self, mask, slice_num):
        """
        Compute the agreement between masks of adjacent slices,
        using sum of absolute differences.

        Note: According to the interleaved multi-slice acquisition scheme,
        first all even slices are acquired, followed by all odd slices,
        so the difference between adjacent slices is 2.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor.
        slice_num : torch.Tensor
            Slice number tensor.

        Returns
        -------
        loss : torch.Tensor
            MaskVariabilityAcrossSlices loss for the given mask.
        """

        if slice_num.shape[0] > 1:
            slice_diff_next = slice_num[1:] - slice_num[:-1]
            adj_slices = abs(slice_diff_next) == 2
            adj_slices = torch.cat((
                adj_slices, torch.tensor([[False]]).to(adj_slices.device)
            ))
            abs_diff = torch.sum(torch.abs(
                mask[adj_slices.expand_as(mask)]
                - mask[torch.roll(adj_slices, shifts=1).expand_as(mask)]
            ))
            return abs_diff / (torch.sum(adj_slices)*mask.shape[1])
        else:
            logging.info(
                "[Trainer::train]: Regularisation on mask variation "
                "can only be calculated for batch sizes > 1")
            return None
