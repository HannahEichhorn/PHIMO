import torch


class T2StarFit:
    """ Computes the T2star maps with least_squares_fit. """

    def __init__(self, dim=3, te1=5, d_te=5,
                 exclude_last_echoes=0):
        super(T2StarFit, self).__init__()
        self.te1 = te1
        self.d_te = d_te
        self.dim = dim
        self.exclude_last_echoes = exclude_last_echoes

    def _least_squares_fit(self, data):
        """
        Fit a linear model to the input data.

        Use torch.linalg.lstsq solve AX-B for
            X with shape (n, k) = (2, 1),
        given
            B with shape (m, k) = (12, 1),
            A with shape (m, n) = (12, 2).
        """

        echo_times = torch.arange(self.te1, data.shape[-1] * self.te1 + 1,
                                  self.d_te, dtype=data.dtype,
                                  device=data.device, requires_grad=True)
        if self.dim == 3:
            echo_times = echo_times.unsqueeze(0).unsqueeze(0).repeat(
                data.shape[0], data.shape[1], 1
            )
        if self.dim == 4:
            echo_times = echo_times.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
                data.shape[0], data.shape[1], data.shape[2], 1
            )

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

        return - 1 / solution[..., 0, 0]

    def __call__(self, img, mask=None):

        if len(img.shape) != self.dim:
            print("This image has different dimensions than expected: ",
                  len(img.shape), self.dim)

        if self.dim == 3:
            img = torch.log(torch.abs(img.permute(1, 2, 0)) + 1e-9)
        if self.dim == 4:
            img = torch.log(torch.abs(img.permute(0, 2, 3, 1)) + 1e-9)

        if self.exclude_last_echoes > 0:
            img = img[..., :-self.exclude_last_echoes]

        t2star = self._calc_t2star(self._least_squares_fit(img))
        t2star = torch.clamp(t2star, min=0, max=200)

        if mask is not None:
            t2star = t2star * mask.to(t2star.device)

        return t2star