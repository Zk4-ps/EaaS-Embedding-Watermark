import torch.nn as nn


class PreLayers(nn.Module):
    """
    Building pre layers used in autoencoder model.
    Is a sequence of FC, ReLU activation and FC.
    """

    def __init__(self, dimensions, num_layers, ratio):
        super(PreLayers, self).__init__()

        # input dims -> input dims // ratio
        layer_dims = dimensions
        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.add_module(f'fc_in_{_}', nn.Linear(layer_dims, layer_dims // ratio))
            layer_dims = layer_dims // ratio

    def forward(self, x):
        output_x = self.layers(x)
        return output_x


class PostLayers(nn.Module):
    """
    Building post layers used in autoencoder model.
    Is a sequence of FC, Batch Normalization, ReLU activation and FC.
    """

    def __init__(self, dimensions, num_layers, ratio):
        super(PostLayers, self).__init__()

        # input dims -> input dims * ratio
        self.layers = nn.Sequential()
        layer_dims = dimensions
        for _ in range(num_layers):
            self.layers.add_module(f'fc_in_{_}', nn.Linear(layer_dims, layer_dims * ratio))
            layer_dims = layer_dims * ratio

    def forward(self, x):
        output_x = self.layers(x)

        return output_x


if __name__ == '__main__':
    test_model = PreLayers(dimensions=1536, num_layers=2, ratio=4)
    print(test_model)
    test_model = PostLayers(dimensions=96, num_layers=2, ratio=4)
    print(test_model)
