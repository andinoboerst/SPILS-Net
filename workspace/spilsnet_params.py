from spilsnet import MinMaxScaler


# -----------------------------------------------------------------------------
# Benchmark configuration
# -----------------------------------------------------------------------------
SPILSNET_BENCHMARK_CONFIG = {
    "dimension": 2,
    "input_size": 42,
    "internal_state_size": 1,
    "encoder_structure": [
        {"k": 3, "s": 2, "p": 2, "out": 4},
        {"k": 3, "s": 2, "p": 2, "out": 5},
    ],
    "bottleneck_pool_size": 1,
    "skip_target_nodes": 5,
    "latent_dim": 16,
    "latent_encoder_mlp": [8, 16],
    "gru_hidden_size": 16,
    "gru_layers": 1,
    "internal_input_mlp": [8],
    "internal_output_mlp": [16, 8],
    "latent_decoder_structure": [32, 16],
    "smoothing_kernel_size": 3,
    "dropout_rate": 0.2,
}


SPILSNET_BENCHMARK_HYPERPARAMETERS = {
    "learning_rate": 0.01,
    "batch_size": 2048,
    "num_epochs": 5000,
    "weight_decay": 0.01,
    "early_stop_patience": 50,
    "loss_alpha": 0.995,
    "loss_beta": 0.005,
    "loss_gamma": 0.0,
}


SPILSNET_BENCHMARK_PARAMETERS = {
    "model_config": SPILSNET_BENCHMARK_CONFIG,
    "hyperparameters": SPILSNET_BENCHMARK_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}


# -----------------------------------------------------------------------------
# Scaled Configuration
# -----------------------------------------------------------------------------
SPILSNET_SCALED_CONFIG = {
    "dimension": 2,
    "input_size": 102,
    "internal_state_size": 1,
    "encoder_structure": [
        {"k": 3, "s": 2, "p": 2, "out": 16},
        {"k": 3, "s": 2, "p": 2, "out": 32},
        {"k": 3, "s": 2, "p": 1, "out": 32},
    ],
    "bottleneck_pool_size": 2,
    "skip_target_nodes": 4,
    "latent_dim": 64,
    "latent_encoder_mlp": [64],
    "gru_hidden_size": 64,
    "gru_layers": 1,
    "internal_input_mlp": [32],
    "internal_output_mlp": [32],
    "latent_decoder_structure": [128, 128],
    "smoothing_kernel_size": 3,
    "dropout_rate": 0.05,
}


SPILSNET_SCALED_HYPERPARAMETERS = {
    "learning_rate": 0.01,
    "batch_size": 2048,
    "num_epochs": 5000,
    "weight_decay": 0.01,
    "early_stop_patience": 50,
    "loss_alpha": 0.9,
    "loss_beta": 0.1,
    "loss_gamma": 0.3,
}


SPILSNET_SCALED_PARAMETERS = {
    "model_config": SPILSNET_SCALED_CONFIG,
    "hyperparameters": SPILSNET_SCALED_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}
