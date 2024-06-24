import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size=1, output_size=92, hidden_sizes=None,
                 compress_output=False, input_embedding=False,
                 init_bias_last_layer=1):
        super(MLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64]
        self.init_bias_last_layer = init_bias_last_layer
        self.compress_output = compress_output
        if self.compress_output:
            output_size = output_size // 2

        self.input_embedding = input_embedding
        if self.input_embedding:
            input_size = input_size * 3
            self.embedding = torch.nn.Embedding(36, input_size)

        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        layers.append(torch.nn.ReLU())

        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        layers.append(torch.nn.Sigmoid())

        self.layers = torch.nn.Sequential(*layers)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                if module == self.layers[-2]:
                    module.bias.data.fill_(self.init_bias_last_layer)
                else:
                    module.bias.data.fill_(0)

    def forward(self, x):
        if self.input_embedding:
            x = self.embedding(x.long())
            x = x.reshape(x.shape[0], -1)

        x = self.layers(x)

        if self.compress_output:
            x = x.repeat_interleave(2, dim=1)

        return x
