import torch
from torch.nn import L1Loss, MSELoss
from merlinth.losses.pairwise_loss import mse


class L2:
    def __init__(self, mask_image=False):
        super(L2, self).__init__()
        self.mask_image = mask_image
        if not self.mask_image:
            self.loss_ = MSELoss()
        else:
            self.loss_ = MSELoss(reduction='none')

    def __call__(self, x, x_recon, mask=None):
        if not self.mask_image:
            return self.loss_(x, x_recon)
        else:
            return torch.mean(self.loss_(x, x_recon)[mask == 1])


class L1:
    def __init__(self):
        super(L1, self).__init__()
        self.loss_ = L1Loss()

    def __call__(self, x, x_recon, z=None):
        return self.loss_(x, x_recon)


class RealImagL2:
    """
    Custom loss function for L2 norm on real and imaginary components
    separately.

    Parameters
    ----------
    mask_image : bool, optional
        Flag indicating whether to use masking during loss computation.
        Default is False.

    Methods
    -------
    __call__(gt, pred, output_components=False, mask=None)
        Calculate the L2 loss between ground truth (gt) and predictions (pred).

        Parameters
        ----------
        gt : torch.Tensor
            Ground truth tensor.
        pred : torch.Tensor
            Prediction tensor.
        output_components : bool, optional
            Flag to output individual loss components (real and imaginary).
            Default is False.
        mask : torch.Tensor, optional
            Mask tensor indicating regions to include in the loss calculation.
            Required if mask_image is True.

        Returns
        -------
        loss : torch.Tensor
            Total L2 loss.
        re_loss : torch.Tensor, optional
            L2 loss on the real component.
        im_loss : torch.Tensor, optional
            L2 loss on the imaginary component.

    Notes
    -----
    - If mask_image is True, the loss is computed only on the regions
    specified by the mask.
    - The output can include individual loss components (real and imaginary)
    if output_components is True.
    """

    def __init__(self, mask_image=False):
        super(RealImagL2, self).__init__()
        self.loss_ = MSELoss()
        self.mask_image = mask_image
        if not self.mask_image:
            self.loss_ = MSELoss()
        else:
            self.loss_ = MSELoss(reduction='none')

    def __call__(self, gt, pred, output_components=False, mask=None):
        """
        Calculate the L2 loss between ground truth (gt) and predictions (pred).

        Parameters
        ----------
        gt : torch.Tensor
            Ground truth tensor.
        pred : torch.Tensor
            Prediction tensor.
        output_components : bool, optional
            Flag to output individual loss components (real and imaginary).
            Default is False.
        mask : torch.Tensor, optional
            Mask tensor indicating regions to include in the loss calculation.
            Required if mask_image is True.

        Returns
        -------
        loss : torch.Tensor
            Total L2 loss.
        re_loss : torch.Tensor, optional
            L2 loss on the real component (if output_components is True).
        im_loss : torch.Tensor, optional
            L2 loss on the imaginary component (if output_components is True).
        """

        if not self.mask_image:
            if not output_components:
                return (self.loss_(torch.real(gt), torch.real(pred))
                        + self.loss_(torch.imag(gt), torch.imag(pred))
                        )
            else:
                re_loss = self.loss_(torch.real(gt), torch.real(pred))
                im_loss = self.loss_(torch.imag(gt), torch.imag(pred))
                return re_loss + im_loss, re_loss, im_loss

        else:
            if not output_components:
                return (torch.mean(self.loss_(torch.real(gt),
                                              torch.real(pred))[mask == 1])
                        + torch.mean(self.loss_(torch.imag(gt),
                                                torch.imag(pred))[mask == 1])
                        )
            else:
                re_loss = torch.mean(self.loss_(torch.real(gt),
                                                torch.real(pred))[mask == 1])
                im_loss = torch.mean(self.loss_(torch.imag(gt),
                                                torch.imag(pred))[mask == 1])
                return re_loss + im_loss, re_loss, im_loss
