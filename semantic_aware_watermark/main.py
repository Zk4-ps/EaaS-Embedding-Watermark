import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from model.options import *
from model.adaptive_wm import AdaptiveWM
from train import train


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of AdaptiveWM Model.')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-Parser for commands.')
    # judge if the run is new run
    new_run_parser = subparsers.add_parser('new', help='Starts a new run.')
    new_run_parser.add_argument('--data-dir', '-d',
                                required=True, type=str, help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b',
                                required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e',
                                default=100, type=int, help='Number of epochs.')
    new_run_parser.add_argument('--name',
                                required=True, type=str, help='The name of the experiment.')
    new_run_parser.add_argument('--emb_size', '-s',
                                default=1536, type=int,
                                help='The size of the embedding.')
    new_run_parser.add_argument('--message', '-m',
                                default=24, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--continue-from-folder', '-c',
                                default='', type=str,
                                help='The folder to continue a previous run. '
                                     'Leave blank if you are starting a new experiment.')

    new_run_parser.add_argument('--tensorboard',
                                action='store_true', help='Use to switch on Tensorboard logging.')
    new_run_parser.set_defaults(tensorboard=False)

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run.')
    continue_parser.add_argument('--folder', '-f',
                                 required=True, type=str, help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. '
                                      'Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                 help='Number of epochs to run the simulation. '
                                      'Specify a value only if you want to override the previous value.')

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == 'continue':
        result_folder = args.folder
        # load the training options and model config
        options_file = os.path.join(result_folder, 'options-and-config.pickle')
        train_options, model_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(result_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1

        # train file and test file
        if args.data_dir is not None:
            train_options.train_file = os.path.join(args.data_dir, 'train_emb.json')
            train_options.validation_file = os.path.join(args.data_dir, 'test_emb.json')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_file=os.path.join(args.data_dir, 'your_train_embedding_file'),
            validation_file=os.path.join(args.data_dir, 'your_test_embedding_file'),
            runs_folder='result',
            start_epoch=start_epoch,
            experiment_name=args.name)
        model_config = ModelConfiguration(
            emb_dim=args.emb_size,
            message_length=args.message,
            encoder_layers=4, decoder_layers=1,
            encoder_ratio=1, decoder_ratio=8,
            use_discriminator=False,
            discriminator_layers=0,
            decoder_loss=0.0001, encoder_loss=1, adversarial_loss=0
        )

        result_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(result_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(model_config, f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(result_folder, f'{train_options.experiment_name}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # TODO: review tensorboard logger
    if (args.command == 'new' and args.tensorboard) or \
            (args.command == 'continue' and os.path.isdir(os.path.join(result_folder, 'tb-logs'))):
        logging.info('Tensorboard is enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(result_folder, 'tb-logs'))
    else:
        tb_logger = None

    # random initialize the decoder parameters, set requires_grad = False
    model = AdaptiveWM(model_config, device, tb_logger)
    model.encoder_decoder.decoder.apply(utils.init_weights)
    for param in model.encoder_decoder.decoder.parameters():
        param.requires_grad = False

    if args.command == 'continue':
        # if training are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('-' * 40)
    logging.info('AdaptiveWM Model: \n{}'.format(model.to_string()))
    logging.info('-' * 40)
    logging.info('Model Configuration:')
    logging.info(pprint.pformat(vars(model_config)))
    logging.info('-' * 40)
    logging.info('Train Options:')
    logging.info(pprint.pformat(vars(train_options)))
    logging.info('-' * 40)

    train(model, device, model_config, train_options, result_folder, tb_logger)


if __name__ == '__main__':
    main()
