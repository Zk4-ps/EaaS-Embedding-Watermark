import os
import re
import csv
import json
import time
import pickle
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data

from model.options import ModelConfiguration, TrainingOptions
from model.adaptive_wm import AdaptiveWM
from model.custom_dataset import CustomDataset


# embedding to tensor: 1-dim
def emb_to_tensor(embedding):
    """
    Transforms an emb(both numpy and list) into torch tensor: 1-dim
    """
    if isinstance(embedding, list):
        emb_tensor = torch.tensor(embedding)
    elif isinstance(embedding, np.ndarray):
        emb_tensor = torch.tensor(embedding)
    else:
        emb_tensor = torch.empty(())  # return the empty tensor

    return emb_tensor


def tensor_to_emb(emb_tensor, emb_type):
    """
    Transforms a torch tensor into list or ndarray
    """
    if emb_type == 'numpy':
        embedding = emb_tensor.numpy()
    elif emb_type == 'list':
        embedding = emb_tensor.tolist()
    else:
        embedding = []  # return the empty emb

    return embedding


def save_val_emb(emb_dict: dict, experiment_name: str, epoch: int, log_file_folder: str):
    if not os.path.exists(log_file_folder):
        os.makedirs(log_file_folder)

    val_emb_file = f'{experiment_name}--val-emb-{epoch}.json'
    val_emb_file = os.path.join(log_file_folder, val_emb_file)
    logging.info('Saving validation embedding to {}'.format(val_emb_file))

    with open(val_emb_file, 'w') as f:
        json.dump(emb_dict, f)


def sorted_nicely(l):
    """
    Sort the given iterable in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# find the last checkpoint from the folder
def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: AdaptiveWM, experiment_name: str, epoch: int, checkpoint_folder: str):
    """
    Saves a checkpoint at the end of an epoch.
    """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))

    # seperate the encoder and decoder
    if model.discriminator:
        checkpoint = {
            'enc-model': model.encoder_decoder.encoder.state_dict(),
            'dec-model': model.encoder_decoder.decoder.state_dict(),
            'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
            'discrim-model': model.discriminator.state_dict(),
            'discrim-optim': model.optimizer_discrim.state_dict(),
            'epoch': epoch
        }
    else:
        checkpoint = {
            'enc-model': model.encoder_decoder.encoder.state_dict(),
            'dec-model': model.encoder_decoder.decoder.state_dict(),
            'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
            'epoch': epoch
        }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file, weights_only=False)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(adaptive_wm_net, checkpoint):
    """ Restores the AdaptiveWM object from a checkpoint object """
    if adaptive_wm_net.discriminator:
        adaptive_wm_net.encoder_decoder.encoder.load_state_dict(checkpoint['enc-model'])
        adaptive_wm_net.encoder_decoder.decoder.load_state_dict(checkpoint['dec-model'])
        adaptive_wm_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
        adaptive_wm_net.discriminator.load_state_dict(checkpoint['discrim-model'])
        adaptive_wm_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])
    else:
        adaptive_wm_net.encoder_decoder.encoder.load_state_dict(checkpoint['enc-model'])
        adaptive_wm_net.encoder_decoder.decoder.load_state_dict(checkpoint['dec-model'])
        adaptive_wm_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])


def load_options(options_file_name) -> (TrainingOptions, ModelConfiguration, dict):
    """ Loads the training options and model configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        model_config = pickle.load(f)

    return train_options, model_config


# TODO FLAG
def get_data_loaders(train_options: TrainingOptions):
    """
    Get torch data loaders for training and validation. The data file maybe the json file,
    transform embedding into tensor.
    """
    train_dataset = CustomDataset(train_options.train_file)
    train_sampler = data.BatchSampler(data.RandomSampler(train_dataset),
                                      batch_size=train_options.batch_size,
                                      drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_sampler,
                                               num_workers=4)

    validation_dataset = CustomDataset(train_options.validation_file)
    validation_sampler = data.BatchSampler(data.RandomSampler(validation_dataset),
                                           batch_size=train_options.batch_size,
                                           drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_sampler=validation_sampler,
                                                    num_workers=4)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, -0.5, 0.5)
        init.uniform_(m.bias, -0.5, 0.5)
    elif isinstance(m, nn.BatchNorm1d):
        init.uniform_(m.weight, -0.5, 0.5)
        init.uniform_(m.bias, -0.5, 0.5)
