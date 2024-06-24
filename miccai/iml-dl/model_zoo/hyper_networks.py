import torch
import torch.nn as nn
import hyperlight as hl
import merlinth


class HyperUnrolled(nn.Module):
    """ HyperNetwork setting the weights of Unrolled Reconstruction Network """

    def __init__(self, mainnet, input_dim=1, hidden_sizes=None,
                 hypernetize_only_last=False, constrain_dc=False,
                 input_embedding=False):
        super().__init__()
        self.hidden_sizes = (hidden_sizes
                             if hidden_sizes is not None
                             else [16, 64, 128])
        self.input_dim = input_dim
        self.constrain_dc = constrain_dc
        self.relu = nn.ReLU()
        self.input_embedding = input_embedding
        if self.input_embedding:
            self.input_dim = self.input_dim * 3
            self.embedding = torch.nn.Embedding(93, self.input_dim)

        # Use HyperLight convenience functions to select relevant modules
        modules = hl.find_modules_of_type(
            mainnet, [nn.Conv2d,
                                  merlinth.layers.data_consistency.DCGD]
        )
        if hypernetize_only_last:
            modules = {k: v for k, v in modules.items()
                       if "DC" in k
                       or k.endswith(".ops.8")
                       or k.startswith("individual_denoiser")
                       }

        self.mainnet = hl.hypernetize(mainnet, modules=modules)
        parameter_shapes = self.mainnet.external_shapes()

        encoding = None if self.input_embedding else "cos|sin"
        self.hypernet = hl.HyperNet(
            input_shapes={'h': (self.input_dim,)},
            output_shapes=parameter_shapes,
            hidden_sizes=self.hidden_sizes,
            encoding=encoding
        )
        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, hyper_input, main_input):
        if self.input_embedding:
            hyper_input = self.embedding((hyper_input*92).long())
            hyper_input = hyper_input.reshape(hyper_input.shape[0], -1)

        parameters = self.hypernet(h=hyper_input)

        if self.constrain_dc:
            for p in parameters.keys():
                if "DC" in p:
                    parameters[p] = self.relu(parameters[p])

        with self.mainnet.using_externals(parameters):
            prediction = self.mainnet(*main_input)

        return prediction

    def predict_parameters(self, hyper_input):
        if self.input_embedding:
            hyper_input = self.embedding((hyper_input * 92).long())
            hyper_input = hyper_input.reshape(hyper_input.shape[0], -1)

        parameters = self.hypernet(h=hyper_input)
        if self.constrain_dc:
            for p in parameters.keys():
                if "DC" in p:
                    parameters[p] = self.relu(parameters[p])
        return parameters

    def get_main_network(self):
        return self.mainnet

    def get_hyper_network(self):
        return self.hypernet
