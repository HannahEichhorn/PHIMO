import torch
import merlinth
from merlinth.layers import ComplexConv2d, ComplexConv3d
from merlinth.layers.complex_act import cReLU
from merlinth.layers.module import ComplexModule
from merlinth.complex import complex2real, real2complex


class ComplexUnrolledNetwork(ComplexModule):
    """ Unrolled network for iterative reconstruction

    Input to the network are zero-filled, coil-combined images, corresponding
    undersampling masks and coil sensitivtiy maps. Output is a reconstructed
    coil-combined image.

    """
    def __init__(self,
                 nr_iterations=10,
                 dc_method="GD",
                 denoiser_method="ComplexCNN",
                 weight_sharing=True,
                 partial_weight_sharing=False,
                 select_echo=False,
                 nr_filters=64,
                 kernel_size=3,
                 nr_layers=5,
                 activation="relu",
                 **kwargs):
        super(ComplexUnrolledNetwork, self).__init__()

        self.nr_iterations = nr_iterations
        self.dc_method = dc_method
        self.T = 1 if weight_sharing else nr_iterations
        input_dim = 12 if select_echo is False else 1
        self.partial_weight_sharing = partial_weight_sharing

        # create layers
        if denoiser_method == "Real2chCNN":
            if not partial_weight_sharing:
                self.denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=input_dim * 2,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=nr_layers,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    **kwargs
                ) for _ in range(self.T)])
            else:
                # share the first nr_layers-1 layers between iterations
                self.shared_denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=input_dim * 2,
                    output_dim=nr_filters,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=nr_layers - 1,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    last_activation=True,
                    **kwargs
                )])
                self.individual_denoiser = torch.nn.ModuleList([Real2chCNN(
                    dim='2D',
                    input_dim=nr_filters,
                    output_dim=input_dim * 2,
                    filters=nr_filters,
                    kernel_size=kernel_size,
                    num_layer=1,
                    activation=activation,
                    use_bias=True,
                    normalization=None,
                    **kwargs
                ) for _ in range(self.T)])
        else:
            print("This denoiser method is not implemented yet.")

        A = merlinth.layers.mri.MulticoilForwardOp(center=True,
                                                   channel_dim_defined=False)
        AH = merlinth.layers.mri.MulticoilAdjointOp(center=True,
                                                    channel_dim_defined=False)
        if self.dc_method == "GD":
            self.DC = torch.nn.ModuleList([
                merlinth.layers.data_consistency.DCGD(A, AH, weight_init=1e-4)
                for _ in range(self.T)
            ])
        elif self.dc_method == "PM":
            self.DC = torch.nn.ModuleList([
                merlinth.layers.data_consistency.DCPM(A, AH, weight_init=1e-2)
                for _ in range(self.T)
            ])
        elif self.dc_method == "None":
            self.DC = []
        else:
            print("This DC Method is not implemented.")

        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,
                                                             torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, img, y, mask, smaps):
        x = img
        for i in range(self.nr_iterations):
            ii = i % self.T
            if not self.partial_weight_sharing:
                x = x + real2complex(self.denoiser[ii](complex2real(x)))
            else:
                x = x + real2complex(
                    self.individual_denoiser[ii](
                        self.shared_denoiser[0](
                            complex2real(x)
                        )
                    )
                )
            if self.dc_method != "None":
                x = self.DC[ii]([x, y, mask, smaps])
        return x


class Real2chCNN(ComplexModule):
    """adapted from merlinth.models.cnn.Real2chCNN"""

    def __init__(
            self,
            dim="2D",
            input_dim=1,
            output_dim=False,
            filters=64,
            kernel_size=3,
            num_layer=5,
            activation="relu",
            use_bias=True,
            normalization=None,
            last_activation=False,
            **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if dim == "2D":
            conv_layer = torch.nn.Conv2d
        elif dim == "3D":
            conv_layer = torch.nn.Conv3d
        else:
            raise RuntimeError(f"Convolutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = torch.nn.ReLU

        if output_dim is False:
            output_dim = input_dim

        padding = kernel_size // 2
        # create layers
        self.ops = []

        if num_layer == 1:
            self.ops.append(
                conv_layer(
                    input_dim,
                    output_dim,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if last_activation:
                if normalization is not None:
                    self.ops.append(normalization())
                self.ops.append(act_layer(inplace=True))

        else:
            self.ops.append(
                conv_layer(
                    input_dim,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization())

            self.ops.append(act_layer(inplace=True))

            for _ in range(num_layer - 2):
                self.ops.append(
                    conv_layer(
                        filters,
                        filters,
                        kernel_size,
                        padding=padding,
                        bias=use_bias,
                        **kwargs,
                    )
                )
                if normalization is not None:
                    self.ops.append(normalization())
                self.ops.append(act_layer(inplace=True))

            self.ops.append(
                conv_layer(
                    filters,
                    output_dim,
                    kernel_size,
                    bias=False,
                    padding=padding,
                    **kwargs,
                )
            )

            if last_activation:
                if normalization is not None:
                    self.ops.append(normalization())
                self.ops.append(act_layer(inplace=True))

        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,
                                                             torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        return x


class MerlinthComplexCNN(ComplexModule):
    """
    This is a copy of merlinth.models.cnn.ComplexCNN since the module could not
    be loaded due to problems with incomplete optox installation.
    """

    def __init__(
        self,
        dim="2D",
        input_dim=1,
        filters=64,
        kernel_size=3,
        num_layer=5,
        activation="relu",
        use_bias=True,
        normalization=None,
        weight_std=False,
        **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if dim == "2D":
            conv_layer = ComplexConv2d
        elif dim == "3D":
            conv_layer = ComplexConv3d
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = cReLU

        padding = kernel_size // 2
        # create layers
        self.ops = []
        self.ops.append(
            conv_layer(
                input_dim,
                filters,
                kernel_size,
                padding=padding,
                bias=use_bias,
                weight_std=weight_std,
                **kwargs,
            )
        )
        if normalization is not None:
            self.ops.append(normalization())

        self.ops.append(act_layer())

        for _ in range(num_layer - 2):
            self.ops.append(
                conv_layer(
                    filters,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization())
            self.ops.append(act_layer())

        self.ops.append(
            conv_layer(
                filters,
                input_dim,
                kernel_size,
                bias=False,
                padding=padding,
                **kwargs,
            )
        )
        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,
                                                             torch.nn.Linear):
            # equivalent to tf.layers.xavier_initalization()
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        return x
