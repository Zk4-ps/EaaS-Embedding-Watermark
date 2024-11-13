class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 train_file: str, validation_file: str, runs_folder: str,
                 start_epoch: int, experiment_name: str):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        self.train_file = train_file
        self.validation_file = validation_file
        self.runs_folder = runs_folder

        self.start_epoch = start_epoch
        self.experiment_name = experiment_name


class ModelConfiguration:
    """
    The adaptive_wm encoder-decoder model network configuration
    """

    def __init__(self, emb_dim, message_length: int,
                 encoder_layers: int, decoder_layers: int,
                 encoder_ratio: int, decoder_ratio: int,
                 use_discriminator: bool,
                 discriminator_layers: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float):
        self.emb_dim = emb_dim
        self.message_length = message_length

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_ratio = encoder_ratio
        self.decoder_ratio = decoder_ratio

        self.use_discriminator = use_discriminator
        self.discriminator_layers = discriminator_layers

        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
