from spilsnet import MinMaxScaler


def create_sequential_edge_index(num_nodes: int = 51) -> list:
    """
    Creates a bi-directional edge_index for a 1D sequential mesh using pure Python lists.
    Returns a list of lists: [sources, targets]
    """
    sources = []
    targets = []

    for i in range(num_nodes - 1):
        # Forward connection: i -> i+1
        sources.append(i)
        targets.append(i + 1)

        # Backward connection: i+1 -> i
        sources.append(i + 1)
        targets.append(i)

    return [sources, targets]


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


SPILSNETGRAPH_BENCHMARK_CONFIG = {
    "dimension": 2,
    "input_size": 42,  # 21 nodes * 2 dimensions
    "internal_state_size": 1,
    # 1. ENCODER SIMPLIFICATION
    "encoder_structure": [
        {"out": 4},
        {"out": 5},
    ],
    "skip_target_nodes": 3,
    "latent_dim": 16,
    "latent_encoder_mlp": [8, 16],
    "gru_hidden_size": 16,
    "gru_layers": 1,
    "internal_input_mlp": [8],
    "internal_output_mlp": [16, 8],
    "latent_decoder_structure": [32, 16],
    "dropout_rate": 0.2,
    # 2. THE TOPOLOGY
    "edge_index": create_sequential_edge_index(num_nodes=21),
}


SPILSNETGRAPH_HIERARCHICAL_BENCHMARK_CONFIG = {
    "dimension": 2,
    "input_size": 42,           # 21 nodes × 2 dims
    "internal_state_size": 1,

    # GNN encoder: 21 → 7 (4 ch) → 3 (4 ch)
    # coarse_flat_size = 3 × 4 = 12
    "encoder_structure": [
        {"out": 4, "nodes": 7},
        {"out": 4, "nodes": 3},
    ],

    # Decoder mirrors the encoder channel depths at each level.
    # Step 1 (3→7):  input channels = 4 (unpool) + 4 (enc skip) = 8
    # Step 2 (7→21): input channels = 4 (unpool) + 4 (enc skip) = 8
    "decoder_structure": [
        {"out": 4},
        {"out": 4},
    ],

    # latent_enc: 12 → 8 → 8
    "latent_dim": 8,
    "latent_encoder_mlp": [8],

    # GRU
    "gru_hidden_size": 8,
    "gru_layers": 1,

    # bottleneck_mixer: (8 gru + 12 spatial) = 20 → 16 → 12
    "bottleneck_mlp": [16],

    # Internal state MLPs
    "internal_input_mlp":  [4],
    "internal_output_mlp": [4],

    "dropout_rate": 0.2,
    "edge_index": create_sequential_edge_index(num_nodes=21),
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

SPILSNETGRAPH_BENCHMARK_PARAMETERS = {
    "model_config": SPILSNETGRAPH_HIERARCHICAL_BENCHMARK_CONFIG,
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


SPILSNETGRAPH_SCALED_CONFIG = {
    "dimension": 2,
    "input_size": 102,
    "internal_state_size": 1,
    "encoder_structure": [
        {"out": 16},
        {"out": 32},
        {"out": 32},
    ],
    "skip_target_nodes": 4,
    "latent_dim": 64,
    "latent_encoder_mlp": [64],
    "gru_hidden_size": 64,
    "gru_layers": 1,
    "internal_input_mlp": [32],
    "internal_output_mlp": [32],
    "latent_decoder_structure": [128, 128],
    "dropout_rate": 0.05,
    "edge_index": create_sequential_edge_index(num_nodes=51),
}


SPILSNETGRAPH_SCALED_PARAMETERS = {
    "model_config": SPILSNETGRAPH_SCALED_CONFIG,
    "hyperparameters": SPILSNET_SCALED_HYPERPARAMETERS,
    "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}
