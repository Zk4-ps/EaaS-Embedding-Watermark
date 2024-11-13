import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.options import ModelConfiguration
from model.encoder_decoder import EncoderDecoder
from tensorboard_logger import TensorBoardLogger


class AdaptiveWM:
    def __init__(self, configuration: ModelConfiguration, device: torch.device, tb_logger):
        """
        :param configuration: Configuration for the net
        :param device: torch.device object, CPU or GPU
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(AdaptiveWM, self).__init__()

        self.discriminator = None
        self.optimizer_discrim = None
        self.encoder_decoder = EncoderDecoder(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.encoder.parameters())

        self.config = configuration
        self.device = device

        self.bce_loss = nn.BCELoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.cos_sim = nn.CosineSimilarity(dim=-1).to(device)

        self.tb_logger = tb_logger
        if tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))

            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of embedding and message
        :param batch: batch of training data, in the form [embedding, message]
        :return: dictionary of error metrics from EncoderDecoder on the current batch
        """
        embeddings, messages = batch
        batch_size = embeddings.shape[0]

        self.encoder_decoder.train()
        with ((torch.enable_grad())):
            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            encoded_embs, decoded_messages = self.encoder_decoder(embeddings)
            # print(encoded_embs.shape, encoded_embs)
            # print(decoded_messages.shape, decoded_messages)

            loss_enc = self.mse_loss(encoded_embs, embeddings)
            loss_dec = self.bce_loss(decoded_messages, messages)
            loss = self.config.encoder_loss * loss_enc + self.config.decoder_loss * loss_dec

            loss.backward()
            self.optimizer_enc_dec.step()

        # let decoded message become [0, 1] vector, compute the Bit Error Rate
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) /
                           (batch_size * messages.shape[0]))

        # add the cos similarity and L2 distance of encoded emb
        embs_cosine_similarities = self.cos_sim(encoded_embs.detach().cpu(), embeddings.detach().cpu())
        mean_embs_cosine_similarity = embs_cosine_similarities.mean()

        embs_l2_distances = torch.norm(encoded_embs.detach().cpu() - embeddings.detach().cpu(), p=2, dim=1)
        mean_embs_l2_distance = embs_l2_distances.mean()

        # add the cos similarity and L2 distance of message(watermark)
        messages_cosine_similarities = self.cos_sim(decoded_messages.detach().cpu(), messages.detach().cpu())
        mean_messages_cosine_similarity = messages_cosine_similarities.mean()

        messages_l2_distances = torch.norm(decoded_messages.detach().cpu() - messages.detach().cpu(), p=2, dim=1)
        mean_messages_l2_distance = messages_l2_distances.mean()

        losses = {
            'loss           ': loss.item(),
            'encoder_mse    ': loss_enc.item(),
            'encoder_cos        ': mean_embs_cosine_similarity.item(),
            'encoder_l2        ': mean_embs_l2_distance.item(),
            'decoder_bce        ': loss_dec.item(),
            'decoder_cos        ': mean_messages_cosine_similarity.item(),
            'decoder_l2        ': mean_messages_l2_distance.item(),
            'bitwise-error  ': bitwise_avg_err,
        }

        return losses, (encoded_embs, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Run validation on a single batch of data consisting of embs and messages
        :param batch: batch of validation data, in form [embs, messages]
        :return: dictionary of error metrics from EncoderDecoder on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)

            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)

        embeddings, messages = batch
        batch_size = embeddings.shape[0]

        self.encoder_decoder.eval()
        with torch.no_grad():
            encoded_embs, decoded_messages = self.encoder_decoder(embeddings)

            loss_enc = self.mse_loss(encoded_embs, embeddings)
            loss_dec = self.bce_loss(decoded_messages, messages)
            loss = self.config.encoder_loss * loss_enc + self.config.decoder_loss * loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) /
                           (batch_size * messages.shape[0]))

        # add the cos similarity and L2 distance of encoded emb
        embs_cosine_similarities = self.cos_sim(encoded_embs.detach().cpu(), embeddings.detach().cpu())
        mean_embs_cosine_similarity = embs_cosine_similarities.mean()

        embs_l2_distances = torch.norm(encoded_embs.detach().cpu() - embeddings.detach().cpu(), p=2, dim=1)
        mean_embs_l2_distance = embs_l2_distances.mean()

        # add the cos similarity and L2 distance of message(watermark)
        messages_cosine_similarities = self.cos_sim(decoded_messages.detach().cpu(), messages.detach().cpu())
        mean_messages_cosine_similarity = messages_cosine_similarities.mean()

        messages_l2_distances = torch.norm(decoded_messages.detach().cpu() - messages.detach().cpu(), p=2, dim=1)
        mean_messages_l2_distance = messages_l2_distances.mean()

        losses = {
            'loss           ': loss.item(),
            'encoder_mse    ': loss_enc.item(),
            'encoder_cos        ': mean_embs_cosine_similarity.item(),
            'encoder_l2        ': mean_embs_l2_distance.item(),
            'decoder_bce        ': loss_dec.item(),
            'decoder_cos        ': mean_messages_cosine_similarity.item(),
            'decoder_l2        ': mean_messages_l2_distance.item(),
            'bitwise-error  ': bitwise_avg_err,
        }

        return losses, (encoded_embs, decoded_messages)

    def to_string(self):
        return str(self.encoder_decoder)
