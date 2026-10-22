import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def scale_data(data, split_indices: list, concatenate=False):

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Combine all training simulations into a single NumPy array (for fitting)
    combined_data = np.concatenate(data[split_indices[0]], axis=0)

    scaler.fit(combined_data)

    scaled_X = np.array([scaler.transform(x) for x in data])

    def concat_func(x):
        if concatenate:
            return np.concatenate(x, axis=0)
        else:
            return x

    train_data = concat_func(scaled_X[split_indices[0]])

    additional_data = []

    for sims in split_indices[1:]:
        additional_data.append(concat_func(scaled_X[sims]))

    return scaler, train_data, *additional_data


def train_val_test_split_tct():
    train_sims = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21]
    val_sims = [7, 13, 19]
    test_sims = [4, 10, 16]

    return train_sims, val_sims, test_sims


class SimulationDataset(Dataset):
    def __init__(self, *args, device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data = [torch.tensor(d, dtype=torch.float64).to(device) for d in args]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

    def append(self, *args) -> None:
        """
        Append new data to the existing dataset.
        """
        new_data_tensors = [torch.tensor(d, dtype=torch.float64).to(self.data[i].device) for i, d in enumerate(args)]
        self.data = [torch.cat((self.data[i], new_data_tensors[i])) for i in range(len(self.data))]


def stack_node_features(*arrays: np.ndarray, dim: int = 3) -> np.ndarray:
    """
    Interleave nodal arrays into node-major features.

    Each array has shape [..., n_nodes * dim] (DOFs ordered node by node). The result has shape
    [..., n_nodes * dim * len(arrays)] with the features of node i stored contiguously, e.g.
    [Fx, Fy, Fz, ux, uy, uz, ...] per node, which is the layout the GNN reshapes into node features.
    """
    reshaped = [a.reshape(*a.shape[:-1], -1, dim) for a in arrays]
    stacked = np.concatenate(reshaped, axis=-1)
    return stacked.reshape(*stacked.shape[:-2], -1)


def build_increment_dataset(forces: np.ndarray, disps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the incremental inverse dataset from forces f^n and displacements u^{n+1} ([N, T, n_dofs]).

    Inputs per node:  [f^n, u^n, u^n - u^{n-1}]
    Targets per node: u^{n+1} - u^n

    The interface starts at rest, so u^0 = 0 and the first previous increment is 0.
    """
    u_prev = np.zeros_like(disps)
    u_prev[:, 1:] = disps[:, :-1]
    du_prev = np.zeros_like(disps)
    du_prev[:, 1:] = u_prev[:, 1:] - u_prev[:, :-1]

    x = stack_node_features(forces, u_prev, du_prev)
    y = disps - u_prev
    return x, y


def build_perturbed_increment_dataset(
    forces: np.ndarray,
    disps: np.ndarray,
    errors: np.ndarray,
    force_deltas: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
    alpha_range: tuple[float, float] = (0.03, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrective samples for the incremental inverse model from one perturbed replay per simulation.

    The reduced domain was driven with u^{n+1} + E[n] instead of u^{n+1}, so at step n the interface
    displacement carries the error e_n = E[n-1] and the measured force is f^n + dF[n]. For small errors
    the force response is linear in the error, so each sample is rescaled by a random alpha
    (log-uniform magnitude in alpha_range, random sign):

        inputs:  [f^n + a dF[n], u^n + a e_n, (u^n - u^{n-1}) + a (e_n - e_{n-1})]
        target:  (u^{n+1} - u^n) - kappa a e_n

    i.e. the model learns to remove a fraction kappa of an interface error per step.

    Args are [N, T, n_dofs] arrays; returns (x, y) in the layout of build_increment_dataset.
    """
    x_clean, y_clean = build_increment_dataset(forces, disps)

    e = np.zeros_like(errors)
    e[:, 1:] = errors[:, :-1]
    de = np.zeros_like(e)
    de[:, 1:] = e[:, 1:] - e[:, :-1]

    lo, hi = np.log(alpha_range[0]), np.log(alpha_range[1])
    alpha = np.exp(rng.uniform(lo, hi, size=e.shape[:2] + (1,))) * rng.choice([-1.0, 1.0], size=e.shape[:2] + (1,))

    x = x_clean + alpha * stack_node_features(force_deltas, e, de)
    y = y_clean - kappa * alpha * e
    return x, y


def row_features(u_rows: np.ndarray, v_rows: np.ndarray, depths, velocity_at_interface: bool = True, include_velocity: bool = True) -> np.ndarray:
    """
    Node-major row features from fields sampled below the interface nodes.

    u_rows, v_rows: [..., n_depths, n_nodes, 3]. Returns [..., n_nodes * n_features] where each node holds
    its displacements at all depths followed by its velocities at all depths (optionally without the
    velocity at depth 0, or without any velocities with include_velocity=False).
    """
    depths = np.asarray(depths, dtype=float)
    u_nodes = np.swapaxes(u_rows, -3, -2)  # [..., n_nodes, n_depths, 3]
    parts = [u_nodes.reshape(*u_nodes.shape[:-2], -1)]
    if include_velocity:
        if not velocity_at_interface:
            v_rows = v_rows[..., depths > 0, :, :]
        v_nodes = np.swapaxes(v_rows, -3, -2)
        parts.append(v_nodes.reshape(*v_nodes.shape[:-2], -1))
    per_node = np.concatenate(parts, axis=-1)
    return per_node.reshape(*per_node.shape[:-2], -1)


def smooth_projection(coords: np.ndarray, degree: int, dim: int = 3) -> np.ndarray:
    """
    Orthogonal projector onto smooth interface displacement fields, as a matrix on flat node-major vectors.

    Each displacement component is restricted to polynomials x^i y^j with i + j <= degree in the in-plane
    interface coordinates (normalised by the interface radius).
    """
    radius = np.max(np.linalg.norm(coords[:, :2], axis=1))
    x, y = coords[:, 0] / radius, coords[:, 1] / radius
    basis = np.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)], axis=1)
    q, _ = np.linalg.qr(basis)
    return np.kron(q @ q.T, np.eye(dim))
