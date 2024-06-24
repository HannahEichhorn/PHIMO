import torch
import numpy as np
import merlinth


class SSIM:
    """
    Structural Similarity Index (SSIM) loss function.

    Parameters
    ----------
    select_echo : bool
        Flag to select whether to calculate SSIM for multi-echo or
        single-echo data.
    mask_image : bool, optional
        Flag indicating whether to use masking during SSIM computation.
        Default is False.

    Methods
    -------
    __call__(gt, pred, output_components=False, mask=None)
        Calculate the SSIM loss between ground truth (gt) and
        predictions (pred).

        Parameters
        ----------
        gt : torch.Tensor
            Ground truth tensor.
        pred : torch.Tensor
            Predicted tensor.
        output_components : bool, optional
            Flag to output individual SSIM components (real and imaginary).
            Default is False.
        mask : torch.Tensor, optional
            Mask tensor indicating regions to include in the SSIM calculation.
            Required if mask_image is True.

        Returns
        -------
        loss : torch.Tensor
            Total SSIM loss.
        re_loss : torch.Tensor, optional
            SSIM loss on the real component (if output_components is True).
        im_loss : torch.Tensor, optional
            SSIM loss on the imaginary component (if output_components is
            True).
    """

    def __init__(self, select_echo, mask_image=False):
        super(SSIM, self).__init__()
        if not select_echo:
            self.loss_ = merlinth.losses.SSIM(channel=12)
        else:
            self.loss_ = merlinth.losses.SSIM(channel=1)
        self.mask_image = mask_image

    def __call__(self, gt, pred, output_components=False, mask=None):
        """
        Calculate the SSIM loss between ground truth (gt) and
        predictions (pred).

        Parameters
        ----------
        gt : torch.Tensor
            Ground truth tensor.
        pred : torch.Tensor
            Predicted tensor.
        output_components : bool, optional
            Flag to output individual SSIM components (real and imaginary).
            Default is False.
        mask : torch.Tensor, optional
            Mask tensor indicating regions to include in the SSIM calculation.
            Required if mask_image is True.

        Returns
        -------
        loss : torch.Tensor
            Total SSIM loss.
        re_loss : torch.Tensor, optional
            SSIM loss on the real component (if output_components is True).
        im_loss : torch.Tensor, optional
            SSIM loss on the imaginary component (if output_components is
            True).
        """

        if not self.mask_image:
            if not output_components:
                return (1 - self.loss_(torch.real(gt), torch.real(pred)) +
                        1 - self.loss_(torch.imag(gt), torch.imag(pred))
                        )
            else:
                re_loss = 1 - self.loss_(torch.real(gt), torch.real(pred))
                im_loss = 1 - self.loss_(torch.imag(gt), torch.imag(pred))
                return re_loss + im_loss, re_loss, im_loss
        else:
            if not output_components:
                _, tmp1 = self.loss_(torch.real(gt), torch.real(pred),
                                     full=True)
                _, tmp2 = self.loss_(torch.imag(gt), torch.imag(pred),
                                     full=True)
                return ((1 - torch.mean(tmp1[mask == 1]))
                        + (1 - torch.mean(tmp2[mask == 1]))
                        )
            else:
                _, tmp1 = self.loss_(torch.real(gt), torch.real(pred),
                                     full=True)
                _, tmp2 = self.loss_(torch.imag(gt), torch.imag(pred),
                                     full=True)
                re_loss = 1 - torch.mean(tmp1[mask == 1])
                im_loss = 1 - torch.mean(tmp2[mask == 1])
                return re_loss + im_loss, re_loss, im_loss


class SSIM_Magn:
    """
    Structural Similarity Index (SSIM) loss function for magnitude images.

    Parameters
    ----------
    mask_image : bool, optional
        Flag indicating whether to use masking during SSIM computation.
        Default is False.

    Methods
    -------
    __call__(x, x_recon, mask=None)
        Calculate the SSIM loss between input magnitude images (x) and
        reconstructed magnitude images (x_recon).

        Parameters
        ----------
        x : torch.Tensor
            Input magnitude images tensor.
        x_recon : torch.Tensor
            Reconstructed magnitude images tensor.
        mask : torch.Tensor, optional
            Mask tensor indicating regions to include in the SSIM calculation.
            Required if mask_image is True.

        Returns
        -------
        loss : torch.Tensor
            SSIM loss between input and reconstructed magnitude images.
    """

    def __init__(self, mask_image=False):
        super(SSIM_Magn, self).__init__()
        self.mask_image = mask_image
        self.loss_ = merlinth.losses.SSIM(channel=12)

    def __call__(self, x, x_recon, mask=None):
        """
        Calculate the SSIM loss between input magnitude images (x) and
        reconstructed magnitude images (x_recon).

        Parameters
        ----------
        x : torch.Tensor
            Input magnitude images tensor.
        x_recon : torch.Tensor
            Reconstructed magnitude images tensor.
        mask : torch.Tensor, optional
            Mask tensor indicating regions to include in the SSIM calculation.
            Required if mask_image is True.

        Returns
        -------
        loss : torch.Tensor
            SSIM loss between input and reconstructed magnitude images.
        """

        if not self.mask_image:
            return self.loss_(x, x_recon)
        else:
            _, tmp = self.loss_(x, x_recon, full=True)
            return torch.mean(tmp[mask == 1])


class PSNR_Magn:
    """
    Peak Signal-to-Noise Ratio (PSNR) loss function for magnitude images.

    Parameters
    ----------
    mask_image : bool, optional
        Flag indicating whether to use masking during PSNR computation.
        Default is False.

    Methods
    -------
    __call__(x, x_recon, mask)
        Calculate the PSNR loss between input magnitude images (x) and
        reconstructed magnitude images (x_recon).

        Parameters
        ----------
        x : torch.Tensor
            Input magnitude images tensor.
        x_recon : torch.Tensor
            Reconstructed magnitude images tensor.
        mask : torch.Tensor
            Mask tensor indicating regions to include in the
            PSNR calculation.

        Returns
        -------
        loss : torch.Tensor
            PSNR loss between input and reconstructed magnitude images.
    """

    def __init__(self, mask_image=False):
        super(PSNR_Magn, self).__init__()
        self.mask_image = mask_image

    def __call__(self, x, x_recon, mask):
        """
        Calculate the PSNR loss between input magnitude images (x) and
        reconstructed magnitude images (x_recon).

        Parameters
        ----------
        x : torch.Tensor
            Input magnitude images tensor.
        x_recon : torch.Tensor
            Reconstructed magnitude images tensor.
        mask : torch.Tensor
            Mask tensor indicating regions to include in the
            PSNR calculation.

        Returns
        -------
        loss : torch.Tensor
            PSNR loss between input and reconstructed magnitude images.
        """

        data_range = x.max()
        mse = (abs(x - x_recon) ** 2) + 1e-15
        psnr_val = 10 * torch.log10(data_range ** 2 / mse)

        if self.mask_image:
            return torch.mean(psnr_val[mask == 1])
        else:
            return torch.mean(psnr_val)
