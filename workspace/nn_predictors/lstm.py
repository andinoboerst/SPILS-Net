import numpy as np
import logging

import torch
import torch.nn as nn
from lion_pytorch import Lion
from torch.utils.data import DataLoader

from nn_predictors.misc import scale_data, train_val_test_split_tct, SimulationDataset


logger = logging.getLogger(__name__)


# Set random seed for reproducibility
np.random.seed(8)
torch.manual_seed(8)


class LSTMNetwork():
    def __init__(self, input_size: int, output_size: int, save_path: str = "lstm_model.pkl") -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.save_path = save_path

        self.curr_epoch = 0
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.optimizer_state_dict = None

        self.set_hyperparameters()

        # Placeholders for data preparation
        self.input_scaler = None
        self.output_scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Model parameters
        self.input_size = input_size
        self.hidden_size = 64
        self.output_size = output_size

        self.num_layers = 2

        self._model = LSTMModel(self.input_size, self.hidden_size, self.output_size, self.num_layers)

    def set_hyperparameters(self) -> None:
        self.learning_rate = 6.3e-5
        self.num_epochs = 4000
        self.weight_decay = 1.2
        self.batch_size = 7

    def prep_data(self, X, Y, train_indices=None, val_indices=None, test_indices=None, gaussian_noise: list[float] | None = None) -> None:
        if train_indices is None or val_indices is None or test_indices is None:
            train_sims, val_sims, test_sims = train_val_test_split_tct()
        else:
            train_sims, val_sims, test_sims = train_indices, val_indices, test_indices

        if gaussian_noise is None:
            gaussian_noise = [0.1, 0.2, 0.4, 0.6]

        self.input_scaler, train_input_data, val_input_data, test_input_data = scale_data(X, [train_sims, val_sims, test_sims])
        self.output_scaler, train_target_data, val_target_data, test_target_data = scale_data(Y, [train_sims, val_sims, test_sims])

        train_dataset = SimulationDataset(train_input_data, train_target_data, device=self.device)
        val_dataset = SimulationDataset(val_input_data, val_target_data, device=self.device)
        test_dataset = SimulationDataset(test_input_data, test_target_data, device=self.device)

        for std in gaussian_noise:
            train_dataset.append(GaussianNoise(noise_std=std)(train_input_data), train_target_data)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

    def validate_epoch(self, epoch, num_epochs, total_train_loss, criterion, optimizer, early_stop_patience, patience_counter) -> tuple[bool, int, float]:
        # Validation phase
        self._model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in self.val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(X_val_batch.device), y_val_batch.to(y_val_batch.device)
                val_outputs = self._model(X_val_batch)
                total_val_loss += criterion(val_outputs, y_val_batch).item()

        total_train_loss /= len(self.train_loader)
        total_val_loss /= len(self.val_loader)

        self.optimizer_state_dict = optimizer.state_dict()
        self.save(f"{self.save_path}")

        # Early stopping (optional)
        stop = False
        if total_val_loss < self.best_val_loss:
            patience_counter = 0
            self.best_val_loss = total_val_loss
            self.best_epoch = epoch
            self.save(f"{self.save_path}_best")

        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                stop = True

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss (MSE): {total_train_loss:.8f}, Val Loss (MSE): {total_val_loss:.8f}, Best Epoch: {self.best_epoch + 1}, Best Val Loss: {self.best_val_loss:.8f}")

        torch.cuda.empty_cache()  # Free unused memory

        return stop, patience_counter, total_val_loss

    def calculate_test_metrics(self):
        self._model.eval()
        test_criterion = nn.L1Loss()
        test_loss = 0.0
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(X_test_batch.device), y_test_batch.to(y_test_batch.device)
                test_outputs = self._model(X_test_batch)

                test_outputs = self.output_scaler.inverse_transform(test_outputs.reshape(-1, test_outputs.shape[-1]).cpu().numpy())
                y_test_batch = self.output_scaler.inverse_transform(y_test_batch.reshape(-1, y_test_batch.shape[-1]).cpu().numpy())

                test_loss += test_criterion(
                    torch.from_numpy(test_outputs),
                    torch.from_numpy(y_test_batch)
                ).item()

        test_loss /= len(self.test_loader)
        logger.info(f"Final Test Loss (L1): {test_loss:.8f}")

    def save(self, path: str) -> None:
        import os
        import json
        from safetensors.torch import save_file
        from spilsnet.utils import serialize_scaler

        path = str(path)
        if not path.endswith(".safetensors"):
            path = f"{path}.safetensors"

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        metadata = {
            "model_config": json.dumps({
                "input_size": self.input_size,
                "hidden_size": getattr(self, "hidden_size", 64),
                "output_size": self.output_size,
                "num_layers": getattr(self, "num_layers", 2),
            }),
            "hyperparameters": json.dumps({
                "learning_rate": getattr(self, "learning_rate", 6.3e-5),
                "num_epochs": getattr(self, "num_epochs", 4000),
                "weight_decay": getattr(self, "weight_decay", 1.2),
                "batch_size": getattr(self, "batch_size", 7),
            }),
            "training_state": json.dumps({
                "best_val_loss": getattr(self, "best_val_loss", float('inf')),
                "best_epoch": getattr(self, "best_epoch", 0),
                "curr_epoch": getattr(self, "curr_epoch", 0),
            }),
            "scalers": json.dumps({
                "input": serialize_scaler(self.input_scaler) if getattr(self, "input_scaler", None) is not None else None,
                "output": serialize_scaler(self.output_scaler) if getattr(self, "output_scaler", None) is not None else None,
            })
        }

        tensors = {k: v.contiguous() for k, v in self._model.state_dict().items()}

        if getattr(self, "optimizer_state_dict", None) is not None:
            opt_state = self.optimizer_state_dict
            for i, group in enumerate(opt_state.get("param_groups", [])):
                for key, val in group.items():
                    if isinstance(val, torch.Tensor):
                        tensors[f"optimizer.param_groups.{i}.{key}"] = val.contiguous()
            for key, val in opt_state.get("state", {}).items():
                for subkey, subval in val.items():
                    if isinstance(subval, torch.Tensor):
                        tensors[f"optimizer.state.{key}.{subkey}"] = subval.contiguous()

        save_file(tensors, path, metadata=metadata)
        logger.info(f"Model saved to {path} (Safetensors format)")

    @classmethod
    def load(cls, path: str) -> "LSTMNetwork":
        import os
        import json
        from safetensors.torch import load_file
        from safetensors import safe_open
        from spilsnet.utils import deserialize_scaler
        import pickle

        path = str(path)
        st_path = path if path.endswith(".safetensors") else f"{path}.safetensors"

        if os.path.exists(st_path):
            tensors = load_file(st_path)
            with safe_open(st_path, framework="pt") as f_safe:
                metadata = f_safe.metadata()

            if not metadata:
                raise ValueError(f"No metadata found in {st_path}")

            model_config = json.loads(metadata["model_config"])
            hyperparameters = json.loads(metadata["hyperparameters"])
            training_state = json.loads(metadata["training_state"])
            scalers_data = json.loads(metadata["scalers"])

            instance = cls(
                input_size=model_config["input_size"],
                output_size=model_config["output_size"],
                save_path=os.path.splitext(st_path)[0]
            )
            instance.hidden_size = model_config.get("hidden_size", 64)
            instance.num_layers = model_config.get("num_layers", 2)

            # Recreate model with correct parameters
            instance._model = LSTMModel(instance.input_size, instance.hidden_size, instance.output_size, instance.num_layers)

            instance.learning_rate = hyperparameters.get("learning_rate", 6.3e-5)
            instance.num_epochs = hyperparameters.get("num_epochs", 4000)
            instance.weight_decay = hyperparameters.get("weight_decay", 1.2)
            instance.batch_size = hyperparameters.get("batch_size", 7)

            instance.best_val_loss = training_state.get("best_val_loss", float('inf'))
            instance.best_epoch = training_state.get("best_epoch", 0)
            instance.curr_epoch = training_state.get("curr_epoch", 0)

            instance.input_scaler = deserialize_scaler(scalers_data.get("input"))
            instance.output_scaler = deserialize_scaler(scalers_data.get("output"))

            model_tensors = {}
            optimizer_tensors = {}
            for k, v in tensors.items():
                if k.startswith("optimizer."):
                    optimizer_tensors[k] = v
                else:
                    model_tensors[k] = v

            instance._model.load_state_dict(model_tensors)
            instance._model.to(instance.device)

            if optimizer_tensors:
                opt_state = {"param_groups": [], "state": {}}
                group_indices = sorted(list(set([int(k.split(".")[2]) for k in optimizer_tensors if "param_groups" in k])))
                for i in group_indices:
                    group = {}
                    prefix = f"optimizer.param_groups.{i}."
                    for k, v in optimizer_tensors.items():
                        if k.startswith(prefix):
                            group[k[len(prefix):]] = v
                    opt_state["param_groups"].append(group)

                state_ids = sorted(list(set([k.split(".")[2] for k in optimizer_tensors if "state" in k])))
                for sid in state_ids:
                    state_id = int(sid)
                    opt_state["state"][state_id] = {}
                    prefix = f"optimizer.state.{sid}."
                    for k, v in optimizer_tensors.items():
                        if k.startswith(prefix):
                            opt_state["state"][state_id][k[len(prefix):]] = v
                instance.optimizer_state_dict = opt_state

            logger.info(f"Model loaded from {st_path} (Safetensors)")
            return instance

        # Fallback to legacy
        pkl_path = f"{path}.pkl" if not path.endswith(".pkl") else path
        if os.path.exists(pkl_path):
            logger.warning(f"Safetensors not found. Falling back to legacy format: {pkl_path}")
            with open(pkl_path, "rb") as f:
                instance = pickle.load(f)
                return instance

        raise FileNotFoundError(f"Could not find model at {path} in either .safetensors or legacy .pkl format.")

    def fit(self, X: np.ndarray, Y: np.ndarray, train_indices=None, val_indices=None, test_indices=None, gaussian_noise: list[float] | None = None) -> None:
        self._model.to(self.device)

        if self.train_loader is None:
            self.prep_data(X, Y, train_indices, val_indices, test_indices, gaussian_noise)

        patience_counter = 0
        early_stop_patience = 50

        criterion = nn.MSELoss()

        optimizer = Lion(self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_state_dict is not None:
            self.optimizer_state_dict['param_groups'][0]['initial_lr'] = self.learning_rate
            optimizer.load_state_dict(self.optimizer_state_dict)

        # noise = GaussianNoise()

        scheduler = WarmupExponentialScheduler(optimizer, last_epoch=self.curr_epoch - 1)

        for epoch in range(self.curr_epoch, self.num_epochs):
            self.curr_epoch = epoch

            total_train_loss = 0
            self._model.train()
            for batch_inputs, batch_targets in self.train_loader:
                batch_inputs, batch_targets = batch_inputs.to(batch_inputs.device), batch_targets.to(batch_targets.device)
                optimizer.zero_grad()

                # Forward pass: process the entire sequence for the batch
                outputs = self._model(batch_inputs)

                # Compute loss: compare the entire output sequence with the target sequence
                loss = criterion(outputs, batch_targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                scheduler.step()

            stop_early, patience_counter, total_val_loss = self.validate_epoch(epoch, self.num_epochs, total_train_loss, criterion, optimizer, early_stop_patience, patience_counter)

            if stop_early:
                break

        logger.info('Training finished!')

        self.calculate_test_metrics()

    def initialize_memory_variables(self) -> None:
        self._model.eval()
        self.hidden_state = None

    def predict(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        x_input = np.concatenate((x, v), axis=0)

        x_normalized = self.input_scaler.transform(x_input.reshape(1, -1))

        x_tensor = torch.tensor(x_normalized, dtype=torch.float64)
        with torch.no_grad():
            predicted, self.hidden_state = self._model(x_tensor[None, :, :], self.hidden_state, return_hidden=True)

        predicted_denormalized = self.output_scaler.inverse_transform(predicted.detach().numpy()[-1].reshape(1, -1))[0]

        return predicted_denormalized

    def get_trainable_params(self):
        return sum(p.numel() for p in self._model.parameters())


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dtype=torch.float64)
        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float64)

    def forward(self, x, hidden=None, return_hidden=False):
        # If hidden and cell states are not provided, initialize them as zeros
        if hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(self.layer_size, batch_size, self.hidden_size, dtype=torch.float64).to(x.device)
            c0 = torch.zeros(self.layer_size, batch_size, self.hidden_size, dtype=torch.float64).to(x.device)
            hidden = (h0, c0)

        # Forward pass through LSTM
        out, hidden_n = self.lstm(x, hidden)
        out = self.fc(out)
        if return_hidden:
            return out, hidden_n
        else:
            return out


class GaussianNoise(nn.Module):
    def __init__(self, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        noise = torch.randn_like(x) * self.noise_std
        return x + noise


class WarmupExponentialScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=4000, decay_steps=10000, gamma=0.99, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            decay_step = self.last_epoch - self.warmup_steps
            return [base_lr * (self.gamma ** (decay_step / self.decay_steps)) for base_lr in self.base_lrs]
