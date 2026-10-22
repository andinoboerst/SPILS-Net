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


# -----------------------------------------------------------------------------
# 3D GNN Configuration (Unstructured 2D interface in 3D domain)
# -----------------------------------------------------------------------------
SPILSNET_3D_GNN_CONFIG = {
    "dimension": 3,
    "input_size": 453,  # 151 interface nodes * 3 DOFs
    "internal_state_size": 1,
    "conv_type": "SAGE",
    "encoder_structure": [
        {"out": 32},
        {"out": 64},
        {"out": 64},
    ],
    "skip_target_nodes": 6,
    "latent_dim": 64,
    "latent_encoder_mlp": [64],
    "gru_hidden_size": 64,
    "gru_layers": 1,
    "internal_input_mlp": [32],
    "internal_output_mlp": [32],
    "latent_decoder_structure": [128, 128],
    "use_decoder_conv": True,
    "dropout_rate": 0.05,
}


SPILSNET_3D_GNN_HYPERPARAMETERS = {
    "learning_rate": 0.01,
    "batch_size": 2048,
    "num_epochs": 5000,
    "weight_decay": 0.01,
    "early_stop_patience": 50,
    "loss_alpha": 0.9,
    "loss_beta": 0.1,
    "loss_gamma": 0.0,
}


SPILSNET_3D_GNN_PARAMETERS = {
    "model_config": SPILSNET_3D_GNN_CONFIG,
    "hyperparameters": SPILSNET_3D_GNN_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}


# -----------------------------------------------------------------------------
# 3D Inverse Incremental GNN Configuration ("inc")
# Inputs per node: [f^n, u^n, u^n - u^{n-1}]; output per node: u^{n+1} - u^n
# -----------------------------------------------------------------------------
SPILSNET_3D_GNN_INC_CONFIG = {
    **SPILSNET_3D_GNN_CONFIG,
    "input_size": 1359,  # 151 interface nodes * 9 features
    "input_node_features": 9,
}


SPILSNET_3D_GNN_INC_PARAMETERS = {
    "model_config": SPILSNET_3D_GNN_INC_CONFIG,
    "hyperparameters": SPILSNET_3D_GNN_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}


# -----------------------------------------------------------------------------
# 3D GNN Configuration with Interface Node Positions (Coordinates-aware GNN)
# -----------------------------------------------------------------------------
SPILSNET_3D_GNN_POS_CONFIG = {
    "dimension": 3,
    "input_size": 453,  # 151 interface nodes * 3 DOFs
    "internal_state_size": 1,
    "use_node_positions": True,
    "node_pos_dim": 3,
    "conv_type": "SAGE",
    "encoder_structure": [
        {"out": 32},
        {"out": 64},
        {"out": 64},
    ],
    "skip_target_nodes": 6,
    "latent_dim": 64,
    "latent_encoder_mlp": [64],
    "gru_hidden_size": 64,
    "gru_layers": 1,
    "internal_input_mlp": [32],
    "internal_output_mlp": [32],
    "latent_decoder_structure": [128, 128],
    "use_decoder_conv": True,
    "dropout_rate": 0.05,
}


SPILSNET_3D_GNN_POS_HYPERPARAMETERS = {
    "learning_rate": 0.01,
    "batch_size": 2048,
    "num_epochs": 5000,
    "weight_decay": 0.01,
    "early_stop_patience": 50,
    "loss_alpha": 0.9,
    "loss_beta": 0.1,
    "loss_gamma": 0.0,
}


SPILSNET_3D_GNN_POS_PARAMETERS = {
    "model_config": SPILSNET_3D_GNN_POS_CONFIG,
    "hyperparameters": SPILSNET_3D_GNN_POS_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}


# -----------------------------------------------------------------------------
# 3D Discontinuous Interface GNN Configuration ("disc")
# -----------------------------------------------------------------------------
SPILSNET_3D_DISC_GNN_CONFIG = {
    "dimension": 3,
    "input_size": 450,  # Default, dynamically adapted to n_interface_nodes * 3
    "internal_state_size": 1,
    "conv_type": "SAGE",
    "encoder_structure": [
        {"out": 32},
        {"out": 64},
        {"out": 64},
    ],
    "skip_target_nodes": 6,
    "latent_dim": 64,
    "latent_encoder_mlp": [64],
    "gru_hidden_size": 64,
    "gru_layers": 1,
    "internal_input_mlp": [32],
    "internal_output_mlp": [32],
    "latent_decoder_structure": [128, 128],
    "use_decoder_conv": True,
    "dropout_rate": 0.05,
}


SPILSNET_3D_DISC_GNN_HYPERPARAMETERS = {
    "learning_rate": 0.01,
    "batch_size": 2048,
    "num_epochs": 5000,
    "weight_decay": 0.01,
    "early_stop_patience": 50,
    "loss_alpha": 0.9,
    "loss_beta": 0.1,
    "loss_gamma": 0.0,
}


SPILSNET_3D_DISC_GNN_PARAMETERS = {
    "model_config": SPILSNET_3D_DISC_GNN_CONFIG,
    "hyperparameters": SPILSNET_3D_DISC_GNN_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}



