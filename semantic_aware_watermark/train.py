import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from model.options import *
from model.adaptive_wm import AdaptiveWM
from average_loss import AverageLoss


def train(model: AdaptiveWM,
          device: torch.device,
          model_config: ModelConfiguration,
          train_options: TrainingOptions,
          result_folder: str,
          tb_logger):
    """
    Trains the encoder-decoder model
    :param model: The model
    :param device: usually this is GPU (if avaliable), otherwise CPU.
    :param model_config: The network configuration
    :param train_options: The training settings
    :param result_folder: The parent folder for the current training run to store logs.
    :param tb_logger:
    TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
    Pass None to disable TensorboardX logging
    """

    train_dataloader, val_dataloader = utils.get_data_loaders(train_options)
    file_count = len(train_dataloader.dataset)

    # confirm the training steps each epoch
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    # TODO
    print_steps = 100
    print_epochs = 50

    messages = torch.Tensor(np.random.choice([0, 1], size=model_config.message_length)).to(device)
    # expand to the batch size
    messages = messages.unsqueeze(0).expand(train_options.batch_size, -1)
    # set up with random messages
    # messages = torch.randint(0, 2, size=messages.shape).float().to(device)
    logging.info('Watermark Message: \n{}'.format(messages[0]))

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))

        # make the value with the same type
        training_losses = defaultdict(AverageLoss)
        epoch_start_time = time.time()
        step = 1

        for idx, embedding in train_dataloader:
            embedding = embedding.to(device)
            losses, _ = model.train_on_batch([embedding, messages])

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_steps == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch)
                    )
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start_time
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(result_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        validation_losses = defaultdict(AverageLoss)
        val_enc_emb = {}
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for idx, embedding in val_dataloader:
            embedding = embedding.to(device)
            losses, (encoded_embs, decoded_messages) = model.validate_on_batch([embedding, messages])
            for i in range(len(idx)):
                val_enc_emb[idx[i]] = encoded_embs[i].tolist()
            for name, loss in losses.items():
                validation_losses[name].update(loss)

        # save validation encoded embedding of the last epoch
        if (epoch == train_options.number_of_epochs or
                epoch % print_epochs == 0):
            utils.save_val_emb(val_enc_emb, train_options.experiment_name, epoch,
                               os.path.join(result_folder, 'val_emb'))
            utils.save_checkpoint(model, train_options.experiment_name, epoch,
                                  os.path.join(result_folder, 'checkpoints'))

        # save the validation losses
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.write_losses(os.path.join(result_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start_time)


if __name__ == '__main__':
    test_config = ModelConfiguration(
        emb_dim=1536, message_length=24,
        encoder_layers=2, decoder_layers=1,
        encoder_ratio=4, decoder_ratio=8,
        use_discriminator=False,
        discriminator_layers=0,
        decoder_loss=1,
        encoder_loss=0.7,
        adversarial_loss=0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model = AdaptiveWM(test_config, device, None)
    test_options = TrainingOptions(
        batch_size=4,
        number_of_epochs=2,
        train_file='data/enron/train_emb.json', validation_file='data/enron/test_emb.json', runs_folder='',
        start_epoch=1, experiment_name='enron'
    )
    result_folder = f'result/{test_options.experiment_name}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    train(test_model, device, test_config, test_options, result_folder, None)
