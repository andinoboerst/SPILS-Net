"""
Smoke tests for SPILS-Net ML predictor modules.

These tests do NOT require FEniCSx (dolfinx, mpi4py, petsc4py) or
spilsnet-torch to be installed. They verify that the pure-Python /
PyTorch ML utilities work correctly with synthetic data.

Run with:
    conda activate spils-net
    pytest tests/smoke_test.py -v
"""
import sys
import os

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Make sure workspace/ on the path so we can import nn_predictors etc.
# when running from the repository root.
# ------------------------------------------------------------------ #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.join(REPO_ROOT, "workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)


# ================================================================== #
# Helpers
# ================================================================== #

def make_synthetic_data(n_sims: int = 22, n_steps: int = 30, n_interface_dofs: int = 22):
    """
    Create synthetic training data with the same shape as the real datasets.

    Training input X shape: (n_sims, n_steps, n_interface_dofs * 3)
      — concatenation of displacements (x), velocities (v), accelerations (a)
    Training output Y shape: (n_sims, n_steps, n_interface_dofs)
      — interface tractions
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_sims, n_steps, n_interface_dofs * 3))
    Y = rng.standard_normal((n_sims, n_steps, n_interface_dofs))
    return X, Y


# ================================================================== #
# Tests: data utilities (nn_predictors.misc)
# ================================================================== #

class TestMiscUtilities:
    def test_train_val_test_split_returns_correct_indices(self):
        from nn_predictors.misc import train_val_test_split_tct

        train, val, test = train_val_test_split_tct()

        # Indices must be non-overlapping
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

        # Together they span 0..21 (22 simulations)
        all_indices = sorted(train + val + test)
        assert all_indices == list(range(22))

    def test_scale_data_range(self):
        from nn_predictors.misc import scale_data, train_val_test_split_tct

        X, _ = make_synthetic_data()
        train_sims, val_sims, test_sims = train_val_test_split_tct()

        scaler, train_scaled, val_scaled, test_scaled = scale_data(
            X, [train_sims, val_sims, test_sims]
        )

        # Training data should be scaled to [-1, 1]
        assert train_scaled.min() >= -1.0 - 1e-9
        assert train_scaled.max() <= 1.0 + 1e-9

    def test_scale_data_shape_preserved(self):
        from nn_predictors.misc import scale_data, train_val_test_split_tct

        X, _ = make_synthetic_data()
        train_sims, val_sims, test_sims = train_val_test_split_tct()

        scaler, train_scaled, val_scaled, test_scaled = scale_data(
            X, [train_sims, val_sims, test_sims]
        )

        assert train_scaled.shape == (len(train_sims), X.shape[1], X.shape[2])
        assert val_scaled.shape == (len(val_sims), X.shape[1], X.shape[2])
        assert test_scaled.shape == (len(test_sims), X.shape[1], X.shape[2])

    def test_simulation_dataset_length(self):
        from nn_predictors.misc import SimulationDataset

        X, Y = make_synthetic_data(n_sims=16, n_steps=30, n_interface_dofs=22)

        dataset = SimulationDataset(X, Y, device="cpu")
        assert len(dataset) == 16

    def test_simulation_dataset_item_shapes(self):
        from nn_predictors.misc import SimulationDataset

        X, Y = make_synthetic_data(n_sims=16, n_steps=30, n_interface_dofs=22)

        dataset = SimulationDataset(X, Y, device="cpu")
        x_item, y_item = dataset[0]

        assert x_item.shape == (30, 66)  # n_steps × (n_interface_dofs * 3)
        assert y_item.shape == (30, 22)  # n_steps × n_interface_dofs

    def test_simulation_dataset_append(self):
        from nn_predictors.misc import SimulationDataset
        import torch

        X, Y = make_synthetic_data(n_sims=16, n_steps=30, n_interface_dofs=22)

        dataset = SimulationDataset(X, Y, device="cpu")
        extra_X = torch.randn(16, 30, 66, dtype=torch.float64)
        extra_Y = torch.randn(16, 30, 22, dtype=torch.float64)
        dataset.append(extra_X, extra_Y)

        assert len(dataset) == 32


# ================================================================== #
# Tests: LSTM network (nn_predictors.lstm)
# ================================================================== #

class TestLSTMNetwork:
    def test_instantiation(self):
        from nn_predictors.lstm import LSTMNetwork

        X, Y = make_synthetic_data()
        net = LSTMNetwork(X.shape[2], Y.shape[2])

        assert net.input_size == X.shape[2]
        assert net.output_size == Y.shape[2]

    def test_trainable_param_count_is_positive(self):
        from nn_predictors.lstm import LSTMNetwork

        X, Y = make_synthetic_data()
        net = LSTMNetwork(X.shape[2], Y.shape[2])

        n_params = net.get_trainable_params()
        assert n_params > 0

    def test_lstm_model_forward_shape(self):
        """Check that the raw LSTMModel produces correct output shapes."""
        import torch
        from nn_predictors.lstm import LSTMModel

        batch, seq_len, in_dim, hidden, out_dim = 4, 30, 66, 64, 22
        model = LSTMModel(in_dim, hidden, out_dim, num_layers=2)
        x = torch.randn(batch, seq_len, in_dim, dtype=torch.float64)
        out = model(x)

        assert out.shape == (batch, seq_len, out_dim)

    def test_hyperparameters_set(self):
        from nn_predictors.lstm import LSTMNetwork

        X, Y = make_synthetic_data()
        net = LSTMNetwork(X.shape[2], Y.shape[2])
        net.set_hyperparameters()

        assert hasattr(net, "learning_rate")
        assert hasattr(net, "num_epochs")
        assert hasattr(net, "batch_size")

    def test_gaussian_noise_module(self):
        import torch
        from nn_predictors.lstm import GaussianNoise

        noise = GaussianNoise(noise_std=0.1)
        x = torch.zeros(10, 5, dtype=torch.float64)
        out = noise(x)

        assert out.shape == x.shape
        # With noise, output should not be all zeros (with overwhelming probability)
        assert not torch.all(out == 0)


# ================================================================== #
# Tests: LSTMModel — hidden state pass-through
# ================================================================== #

class TestLSTMModelHiddenState:
    def test_return_hidden(self):
        import torch
        from nn_predictors.lstm import LSTMModel

        model = LSTMModel(input_size=66, hidden_size=64, output_size=22, num_layers=2)
        x = torch.randn(1, 30, 66, dtype=torch.float64)

        out, hidden = model(x, return_hidden=True)
        h, c = hidden

        assert out.shape == (1, 30, 22)
        assert h.shape == (2, 1, 64)
        assert c.shape == (2, 1, 64)


# ================================================================== #
# Tests: SPILSNet Integration (from spilsnet-torch package)
# ================================================================== #

class TestSPILSNetIntegration:
    """
    Tests that the external spilsnet-torch package works correctly
    when integrated into this repository.
    """

    @pytest.fixture
    def minimal_config(self):
        return {
            "dimension": 2,
            "input_size": 10,
            "internal_state_size": 2,
            "encoder_structure": [
                {"out": 8, "k": 3, "s": 1, "p": 1},
            ],
            "bottleneck_pool_size": 2,
            "latent_dim": 4,
            "gru_hidden_size": 16,
            "latent_encoder_mlp": [8],
            "internal_input_mlp": [8],
            "internal_output_mlp": [8],
            "dropout_rate": 0.1,
        }

    def test_spilsnet_instantiation(self, minimal_config):
        from spilsnet import SPILSNet
        net = SPILSNet(model_config=minimal_config)
        assert net.input_size == 10
        assert net.get_trainable_params() > 0

    def test_spilsnet_fit_predict(self, minimal_config):
        from spilsnet import SPILSNet
        # Use enough simulations so the automatic 70/15/15 split gives
        # at least 1 validation simulation (requires n >= 7).
        n_sims = 10
        I = np.random.randn(n_sims, 10, 2)
        X_dummy = np.random.randn(n_sims, 10, 10)
        Y_dummy = np.random.randn(n_sims, 10, 10)

        net = SPILSNet(model_config=minimal_config)
        net.num_epochs = 1
        net.fit(X_dummy, Y_dummy, I)

        net.initialize_memory_variables()
        x_step = np.random.randn(10)
        y_pred = net.predict(x_step)
        assert y_pred.shape == (10,)

    def test_spilsnet_serialization(self, minimal_config, tmp_path):
        from spilsnet import SPILSNet
        import os

        save_path = str(tmp_path / "test_spils")
        net = SPILSNet(model_config=minimal_config, save_path=save_path)
        
        # Check that it saves as .safetensors
        net.save(save_path)
        assert os.path.exists(f"{save_path}.safetensors")

        # Load back
        loaded = SPILSNet.load(save_path)
        assert loaded.model_config == minimal_config
