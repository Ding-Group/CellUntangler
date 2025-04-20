import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = None
    config.device = torch.device("cpu")

    # Hyperparameters for dataset
    # The batch size to use when training the model
    config.batch_size = 128
    
    # Hyperparameters for model
    # True to use a single parameter for the covariance matrix, False otherwise
    config.scalar_parametrization = False
    # If the dataset has no batches, 1 or [1]
    # Otherwise, it should be a list of integers, one integer for each batch
    # The integer inidicates how many of each batch there are
    # E.g., [3, 12] means there are two batches, one with 3 labels and one with 12 labels
    config.n_batch = [1]
    # True to not pass the batch to the encoder, False otherwise
    config.batch_invariant = True
    # True to use the batch with z1, if False, batch set to 0s and not used
    config.use_z1_batch = True
    # True to use the batch with z2, if False, batch set to 0s and not used
    config.use_z2_batch = True
    # The initialization for the model weights
    # May be default initialization in which case config.init = None
    # Alternatively, it may be "normal", "xavier_uniform", "xavier_normal",
    # "he_uniform", "custom", or "custom_xavier_normal"
    # We use config.init = "custom" when trying to capture the cell cycle
    config.init = None
    # Used when config.init is set to "xavier_normal" or "xavier_normal"
    config.gain = 1.0
    # The activation function for the encoder and decoder
    # May be "relu", "leaky_relu", or "tanh"
    # Otherwise, it is the GELU activation function
    config.activation = "gelu"
    # The size of the last layer of the encoder
    config.h_dim = 32
    # True to use stop gradient, False otherwise
    config.use_z2_no_grad = False
    # The epoch to start using stop gradient if config.use_z2_no_grad = True
    config.start_z2_no_grad = 0
    # The epoch to end using stop gradient if config.use_z2_no_grad = True
    config.end_z2_no_grad = 500

    # Hyperparmeters for training
    # The total number of epochs to use for training the model
    config.max_epochs = 500
    # Betas to weight the KL-divergence
    config.start = 1.0
    config.end = 1.0
    config.end_epoch = 1
    config.epochs = 200
    # Batch normalization
    # True to use batch normalization, False otherwise
    config.use_batch_norm = False
    # Momentum and eps for batch normalization
    config.momentum = 0.99
    config.eps = 0.001
    # Hyperparameters for the optimizer
    # True to use the AdamW oprimizer, False otherwise
    config.use_adamw = True
    # Weight decay for the optimizer
    config.weight_decay = None
    config.learning_rate = 0.001
    # True to use the fixed curvature, False to learn the curvature
    config.fixed_curvature = True

    return config
