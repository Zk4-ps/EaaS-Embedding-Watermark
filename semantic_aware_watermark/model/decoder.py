import torch
import torch.nn as nn
from model.options import ModelConfiguration
from model.basic_layers import PreLayers, PostLayers


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked emb and extracts the watermark.
    """
    def __init__(self, config: ModelConfiguration):
        super(Decoder, self).__init__()

        self.emb_dim = config.emb_dim
        self.num_layers = config.decoder_layers
        self.message_length = config.message_length
        self.ratio = config.decoder_ratio

        # only need the pre layers
        layer_dim = config.emb_dim
        self.layers = PreLayers(layer_dim, self.num_layers, self.ratio)

        # fully connected layers
        for i in range(self.num_layers):
            layer_dim = layer_dim // self.ratio
        self.linear = nn.Linear(layer_dim, self.message_length)

    def forward(self, emb_with_wm):
        # go with pre layers
        x = self.layers(emb_with_wm)
        x = self.linear(x)

        # return decoded messages with sigmoid
        return torch.sigmoid(x)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("CUDA version:", torch.version.cuda)
        print("Number of CUDA devices:", torch.cuda.device_count())
    else:
        print("CUDA is not available.")

    test_config = ModelConfiguration(
        emb_dim=1536, message_length=24,
        encoder_layers=2, decoder_layers=1,
        encoder_ratio=4, decoder_ratio=8,
        use_discriminator=False,
        discriminator_layers=0,
        decoder_loss=0,
        encoder_loss=0,
        adversarial_loss=0
    )

    test_model = Decoder(test_config)
    print(test_model)
