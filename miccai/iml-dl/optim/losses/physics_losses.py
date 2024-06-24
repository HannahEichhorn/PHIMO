import torch


class ModelFitError:
    """
    Computes the ModelFitError as averaged residuals of a least squares fit
    (model: exponential decay) to the input data.
    """

    def __init__(self, mask_image=True, te1=5, d_te=5,
                 error_type='squared_residuals'):
        super(ModelFitError, self).__init__()
        self.mask_image = mask_image
        self.te1 = te1
        self.d_te = d_te
        self.error_type = error_type

    def _least_squares_fit(self, data):
        """ Fit a linear model to the input data.

        Use torch.linalg.lstsq solve AX-B for
            X with shape (n, k) = (2, 1),
        given
            B with shape (m, k) = (12, 1),
            A with shape (m, n) = (12, 2).
        """

        echo_times = torch.arange(self.te1, data.shape[1] * self.te1 + 1,
                                  self.d_te, dtype=data.dtype,
                                  device=data.device, requires_grad=True)
        echo_times = echo_times.unsqueeze(0).repeat(data.shape[0], 1)

        self.B = data.unsqueeze(-1)
        self.A = torch.cat((echo_times.unsqueeze(-1),
                            torch.ones_like(echo_times).unsqueeze(-1)),
                           dim=-1)
        return torch.linalg.lstsq(self.A, self.B).solution

    def _calc_residuals(self, data, solution):
        """ Calculate the residuals of the least squares fit. """

        return data - (self.A @ solution)[..., 0]

    def _calc_emp_corr(self, data, solution):
        """
        Calculate the empirical correlation coefficient (= Pearson)
        between the original and the fitted signal intensities.

        Note: Values of the correlation coefficient are between -1 and 1.
        The error is calculated as 1-emp_corr to have a value
        between 0 and 1.
        """

        orig_signal = torch.exp(data)
        fit_signal = torch.exp(self.A @ solution)[..., 0]

        mean_orig = torch.mean(orig_signal, dim=1)
        mean_fit = torch.mean(fit_signal, dim=1)

        numerator = torch.sum(
            (orig_signal - mean_orig.unsqueeze(1))
            * (fit_signal - mean_fit.unsqueeze(1)),
            dim=1
        )
        denominator = torch.sqrt(
            torch.sum((orig_signal - mean_orig.unsqueeze(1))**2, dim=1) *
            torch.sum((fit_signal - mean_fit.unsqueeze(1))**2, dim=1)
        )
        return 1 - torch.mean(numerator/denominator)

    def __call__(self, img, mask=None):
        """
        Calculate squared residuals of a least squares fit or
        empirical correlation coefficient as ModelFitError
        for input magnitude images (x) and image mask (mask).

        Note: currently, the T2* values are not clipped to a maximum value.

        Parameters
        ----------
        img : torch.Tensor
            Input magnitude images.
        mask : torch.Tensor
            Mask tensor indicating regions to include in the loss.

        Returns
        -------
        loss : torch.Tensor
            ModelFitError of the input magnitude images.
        """

        if self.mask_image and mask is None:
            print("ERROR: Masking is enabled but no mask is provided.")

        if not self.mask_image:
            mask = torch.ones_like(img)

        mask = mask.permute(1, 0, 2, 3)
        img = img.permute(1, 0, 2, 3)
        mask = mask[0] > 0

        # Convert data to solve a linear equation
        # A small constant is added to avoid taking the logarithm of zero
        data = torch.log(torch.abs(img[:, mask]) + 1e-9).T

        solution = self._least_squares_fit(data)

        if self.error_type == 'squared_residuals':
            residuals = self._calc_residuals(data, solution)
            error = torch.mean(residuals**2)
        elif self.error_type == 'emp_corr':
            error = self._calc_emp_corr(data, solution)
        else:
            raise ValueError('Invalid error type.')
        return error


class T2StarDiff:
    """
    Computes the T2StarDiff as mean absolute error between T2star maps
    resulting from predicted and ground truth images.
    """

    def __init__(self, mask_image=True, te1=5, d_te=5):
        super(T2StarDiff, self).__init__()
        self.mask_image = mask_image
        self.te1 = te1
        self.d_te = d_te

    @staticmethod
    def _linearize_input(img, mask):
        """
        Convert input data to solve a linear equation.

        A small constant is added to avoid taking the logarithm of zero.
        """

        return torch.log(torch.abs(img[:, mask]) + 1e-9).T

    def _least_squares_fit(self, data):
        """
        Fit a linear model to the input data.

        Use torch.linalg.lstsq solve AX-B for
            X with shape (n, k) = (2, 1),
        given
            B with shape (m, k) = (12, 1),
            A with shape (m, n) = (12, 2).
        """

        echo_times = torch.arange(self.te1, data.shape[1] * self.te1 + 1,
                                  self.d_te, dtype=data.dtype,
                                  device=data.device, requires_grad=True)
        echo_times = echo_times.unsqueeze(0).repeat(data.shape[0], 1)

        self.B = data.unsqueeze(-1)
        self.A = torch.cat((echo_times.unsqueeze(-1),
                            torch.ones_like(echo_times).unsqueeze(-1)),
                           dim=-1)
        return torch.linalg.lstsq(self.A, self.B).solution

    @staticmethod
    def _calc_t2star(solution):
        """
        Calculate T2* relaxation times from the least squares fit solution.

        Signal model:   img(t) = A_0 * exp(-t/T2*)
        Linearized:       log(img(t)) = log(A_0) - t/T2*
                                B = X[0] * t +X[1] = AX
        with    X = [- 1/T2*, log(A_0),],
                   B = log(img(t)),
                   A = [t, 1].
        """

        return - 1 / solution[..., 0]

    def __call__(self, img_gt, img_pred, mask=None):
        """
        Calculate mean absolute error between T2star maps resulting from
        predicted and ground truth images.

        Note: currently, the T2* values are clipped between 0 and 200.

        Parameters
        ----------
        img_gt : torch.Tensor
            Input  ground truth images.
        img_pred : torch.Tensor
            Input predicted images.
        mask : torch.Tensor
            Mask tensor indicating regions to include in the loss.

        Returns
        -------
        loss : torch.Tensor
            T2StarDiff of the input ground truth and predicted images.
        """

        if self.mask_image and mask is None:
            print("ERROR: Masking is enabled but no mask is provided.")

        if not self.mask_image:
            mask = torch.ones_like(img_pred)

        mask = mask.permute(1, 0, 2, 3)
        img_pred = img_pred.permute(1, 0, 2, 3)
        img_gt = img_gt.permute(1, 0, 2, 3)
        mask = mask[0] > 0

        data_gt = self._linearize_input(img_gt, mask)
        data_pred = self._linearize_input(img_pred, mask)

        t2star_gt = self._calc_t2star(self._least_squares_fit(data_gt))
        t2star_pred = self._calc_t2star(self._least_squares_fit(data_pred))

        t2star_gt = torch.clamp(t2star_gt, min=0, max=200)
        t2star_pred = torch.clamp(t2star_pred, min=0, max=200)

        return torch.mean(torch.abs(t2star_gt - t2star_pred))
