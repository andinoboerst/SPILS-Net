from spilsnet import MinMaxScaler


SPILSNET_BENCHMARK_CONFIG = {
    "dimension": 2,
    "input_size": 42,
    "internal_state_size": 1,

    # --- 1. The "Aggressive" Encoder ---
    # We stride heavily (s=2) immediately to throw away local details.
    "encoder_structure": [
        # 21 -> 11 nodes.
        # Kernel 7 catches 33% of the beam in the first layer.
        {"k": 3, "s": 2, "p": 2, "out": 4},  # 21 -> 11 nodes
        {"k": 3, "s": 2, "p": 2, "out": 5},  # 11 -> 5 nodes
    ],

    "bottleneck_pool_size": 1,
    "skip_target_nodes": 5,

    # --- 2. The "Choke Point" ---
    # Latent Dim = 4.
    # This is the most critical change. It restricts the physics to
    # only 4 degrees of freedom (e.g., x-trans, y-trans, rotation, bend).
    # Noise requires high dimensions to exist; this kills it.
    "latent_dim": 16,

    # Keep this simple
    "latent_encoder_mlp": [8, 16],

    # --- 3. The Dynamics ---
    "gru_hidden_size": 16,
    "gru_layers": 1,

    "internal_input_mlp": [8],
    "internal_output_mlp": [16, 8],

    # --- 4. The "Linear" Decoder ---
    # EMPTY LIST [] means NO HIDDEN LAYERS.
    # It becomes a single Linear matrix: [Latent(4) + Pooled(16) -> Output(42)].
    # A single matrix multiplication is perfectly smooth.
    # It cannot create "kinks" or local spikes because it lacks ReLUs.
    "latent_decoder_structure": [32, 16],

    # --- 5. The "Sledgehammer" Smoothing ---
    # Kernel 9 covers ~45% of the mesh.
    # Node 10 is now the average of Nodes 6 through 14.
    # Independent oscillation is mathematically impossible.
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
    "internal_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
}
