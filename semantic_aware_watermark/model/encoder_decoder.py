import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.options import ModelConfiguration


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Decoder into single pipeline.
    The input is the org emb and the watermark message. The module inserts the watermark into the emb
    Then passes the wm_emb to the Decoder which tries to recover the watermark.
    The module outputs a two-tuple: (encoded_emb, decoded_message)
    """
    def __init__(self, config: ModelConfiguration):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, embedding):
        encoded_emb = self.encoder(embedding)
        decoded_message = self.decoder(encoded_emb)

        return encoded_emb, decoded_message


if __name__ == '__main__':
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

    test_model = EncoderDecoder(test_config)
    print(test_model)
