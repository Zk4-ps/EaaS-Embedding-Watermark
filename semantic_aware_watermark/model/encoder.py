from typing import final

import torch
import torch.nn as nn
from model.options import ModelConfiguration
from model.basic_layers import PreLayers, PostLayers


class Encoder(nn.Module):
    """
    Inserts a watermark into an embedding.
    """
    def __init__(self, config: ModelConfiguration):
        super(Encoder, self).__init__()

        self.emb_dim = config.emb_dim
        self.num_layers = config.encoder_layers
        self.message_length = config.message_length
        self.ratio = config.encoder_ratio

        # pre and post layers in autoencoder model
        pre_layer_dim = config.emb_dim
        post_layer_dim = config.emb_dim
        for i in range(self.num_layers):
            post_layer_dim = post_layer_dim // self.ratio

        self.encode_layers = PreLayers(pre_layer_dim, self.num_layers, self.ratio)
        self.encode_relu = nn.ReLU(inplace=False)
        self.decode_layers = PostLayers(post_layer_dim, self.num_layers, self.ratio)

    def forward(self, embedding):
        # deal with the embedding
        encoded_emb = self.encode_layers(embedding)
        middle_emb = self.encode_relu(encoded_emb)
        decoded_emb = self.decode_layers(middle_emb)

        return decoded_emb


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

    test_model = Encoder(test_config)
    print(test_model)
