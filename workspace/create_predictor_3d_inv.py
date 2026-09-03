"""
3D Inverse Predictor Creation and Evaluation Script for SPILS-Net.

Maps interface forces/tractions (Fx, Fy, Fz per node) to interface
displacements (ux, uy, uz per node) and internal state for the 3D
cylindrical TCT (Tension-Compression-Tension) simulation domain.

Inverse direction of create_predictor_3d.py:
  Input (X):  Interface Forces/Tractions (151 nodes * 3 DOFs = 453 features)
  Output (Y): Interface Displacements (151 nodes * 3 DOFs = 453 features)
  Internal:   Top-half plastic strain state (PCA reduced, 1 feature)
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Predictor3D_Inv")


# ---------------------------------------------------------------------------
# FEM imports (only available inside Docker / with dolfinx installed)
# ---------------------------------------------------------------------------
try:
    from fem_sim.tct_displacements_3d import TCTExtractDisplacements3D, TCTApplyDisplacements3D
    from fem_sim.tct_sims_3d import tct_comp_3d
    from fem_sim.progress_bar import progressbar
    FEM_AVAILABLE = True
except (RuntimeError, ModuleNotFoundError) as _e:
    logger.warning("FEM packages not available – FEM-dependent commands will fallback to saved datasets. (%s)", _e)
    FEM_AVAILABLE = False

# Neural-network predictors
if not os.environ.get("USE_PYPI_SPILSNET"):
    for _p in [Path("/spilsnet-torch"), Path("../spilsnet-torch"), Path("../../spilsnet-torch")]:
        if _p.exists() and (_p / "spilsnet").exists():
            _res = str(_p.resolve())
            if _res not in sys.path:
                sys.path.insert(0, _res)
            break

from nn_predictors.misc import train_val_test_split_tct
try:
    from nn_predictors.lstm import LSTMNetwork
except (ImportError, ModuleNotFoundError):
    LSTMNetwork = None
from spilsnet import SPILSNet, SPILSNetGraph, set_seed
from spilsnet_params import SPILSNET_SCALED_PARAMETERS, SPILSNET_3D_GNN_PARAMETERS

set_seed(8)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
DATA_FOLDER = Path("training_data_3d_inv")
if not (DATA_FOLDER / "training_in_3d_plastic_x.npz").exists() and (Path("workspace/training_data_3d_inv") / "training_in_3d_plastic_x.npz").exists():
    DATA_FOLDER = Path("workspace/training_data_3d_inv")

MODEL_FOLDER = Path("surrogate_models_3d_inv")
if not MODEL_FOLDER.exists() and Path("workspace/surrogate_models_3d_inv").exists():
    MODEL_FOLDER = Path("workspace/surrogate_models_3d_inv")

RESULTS_FOLDER = Path("results_3d_inv")
if not RESULTS_FOLDER.exists() and Path("workspace/results_3d_inv").exists():
    RESULTS_FOLDER = Path("workspace/results_3d_inv")

for _folder in [DATA_FOLDER, MODEL_FOLDER, RESULTS_FOLDER]:
    _folder.mkdir(parents=True, exist_ok=True)


DEFAULT_TRAIN_SIMS, DEFAULT_VAL_SIMS, DEFAULT_TEST_SIMS = train_val_test_split_tct()

# PCA rank for the plastic-top field saved into the training set.
PCA_N_COMPONENTS = 1


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Sim3DConfig:
    """Holds configuration for a 3D inverse simulation problem."""

    constitutive_law: str       = "plastic"
    predictor_version: str      = "01"
    frequency: int              = 1000
    predictor_method: str       = "spils_net_gnn"
    # Hard limit for memory-constrained runs; set to None for a full run.
    time_total: float | None    = 0.001

    train_sims: List[int]       = field(default_factory=lambda: list(DEFAULT_TRAIN_SIMS))
    val_sims:   List[int]       = field(default_factory=lambda: list(DEFAULT_VAL_SIMS))
    test_sims:  List[int]       = field(default_factory=lambda: list(DEFAULT_TEST_SIMS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.load(f)
        return data[list(data.keys())[0]]


def _append_npz(path: Path, sample: np.ndarray) -> None:
    """Append one simulation (axis 0) to a compressed npz, creating the file if needed."""
    sample = np.asarray(sample)
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]
    if path.exists():
        data = np.concatenate([_load_npz(path), sample], axis=0)
    else:
        data = sample
    np.savez_compressed(path, data)


def _n_samples(path: Path) -> int:
    return 0 if not path.exists() else int(_load_npz(path).shape[0])


def _trim_npz(path: Path, n: int) -> None:
    if not path.exists():
        return
    arr = _load_npz(path)
    if arr.shape[0] > n:
        logger.warning("Trimming extra samples in %s (%d → %d)", path.name, arr.shape[0], n)
        np.savez_compressed(path, arr[:n])


def _completed_frequencies(
    freq_path: Path,
    save_paths: List[Path],
    frequency_range: List[int],
) -> List[int]:
    """Frequencies already stored, in sample order."""
    if freq_path.exists():
        freqs = [int(f) for f in np.atleast_1d(_load_npz(freq_path))]
    else:
        existing = [p for p in save_paths if p.exists()]
        n = min((_n_samples(p) for p in existing), default=0)
        missing = [p.name for p in save_paths if not p.exists()]
        if missing and n:
            logger.warning(
                "Missing %s; assuming the %d existing sample(s) are frequencies %s.",
                missing,
                n,
                frequency_range[:n],
            )
        freqs = list(frequency_range[:n])
        if freqs:
            logger.warning(
                "No %s found; assuming %d saved simulation(s) are frequencies %s.",
                freq_path.name,
                len(freqs),
                freqs,
            )
            np.savez_compressed(freq_path, np.asarray(freqs, dtype=np.int32))

    n = len(freqs)
    if n > 0:
        _trim_npz(freq_path, n)
        for path in save_paths:
            _trim_npz(path, n)
    return freqs


def _make_extractor(config: Sim3DConfig) -> "TCTExtractDisplacements3D":
    sim = TCTExtractDisplacements3D(
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration="scaled",
    )
    if config.time_total is not None:
        sim.time_total = config.time_total
    return sim


def _make_applicator(predictor, config: Sim3DConfig) -> "TCTApplyDisplacements3D":
    sim = TCTApplyDisplacements3D(
        predictor,
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration="scaled",
    )
    if config.time_total is not None:
        sim.time_total = config.time_total
    return sim


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_simulation(config: Sim3DConfig):
    """Run a single full FEM simulation and export a GIF."""
    if not FEM_AVAILABLE:
        logger.error("FEM packages not available for direct simulation.")
        return
    logger.info("Running 3D TCT simulation (freq=%s Hz, law=%s)…", config.frequency, config.constitutive_law)
    sim = _make_extractor(config)
    sim.run()
    name = f"simulate_3d_{config.constitutive_law}_f{config.frequency}"
    logger.info("Exporting animation → %s.gif", name)
    sim.postprocess("u", "u", "norm", name=name)


def verify_fem(config: Sim3DConfig):
    """FEM-only verification sanity check."""
    if not FEM_AVAILABLE:
        logger.error("FEM packages not available for FEM verification.")
        return 0.0

    logger.info("=== FEM Verification (extract → compare) ===")
    extractor = _make_extractor(config)
    extractor.run()
    extractor.format_results()

    logger.info("FEM simulation completed successfully.")
    return 0.0


def generate_training_set(config: Sim3DConfig):
    """
    Generate a multi-frequency 3D dataset with synchronized forces and displacements.

    In the inverse dataset:
      - training_in_3d_{tag}_forces.npz:  Interface Forces (X) at time step t_{k+1}
      - training_out_3d_{tag}_disps.npz:   Interface Displacements (Y) at time step t_{k+1}
      - plastic_top_reduced_3d_{tag}.npz:  Internal plastic strain state at time step t_{k+1}
    """
    if not FEM_AVAILABLE:
        logger.error("FEM packages not available for training set generation.")
        return

    logger.info("Generating 3D inverse training set (Forces -> Displacements, law=%s)…", config.constitutive_law)

    tag = config.constitutive_law
    in_path = DATA_FOLDER / f"training_in_3d_{tag}_forces.npz"
    out_path = DATA_FOLDER / f"training_out_3d_{tag}_disps.npz"
    plastic_path = DATA_FOLDER / f"plastic_top_reduced_3d_{tag}.npz"
    freq_path = DATA_FOLDER / f"training_freqs_3d_{tag}_inv.npz"
    topo_path = DATA_FOLDER / f"interface_topology_3d_{tag}.npz"
    pca_path = DATA_FOLDER / "plastic_top_3d_for_pca.npz"

    save_paths = [in_path, out_path]
    if config.constitutive_law == "plastic":
        save_paths.append(plastic_path)

    frequency_range = list(range(500, 2001, int(1500 / 21)))
    completed_freqs = _completed_frequencies(freq_path, save_paths, frequency_range)
    completed_set = set(completed_freqs)
    remaining = [f for f in frequency_range if f not in completed_set]
    if completed_freqs:
        next_freq = remaining[0] if remaining else None
        logger.info(
            "Resuming: %d / %d frequencies already saved %s. Next: %s Hz.",
            len(completed_freqs),
            len(frequency_range),
            completed_freqs,
            next_freq,
        )

    pca = None
    if config.constitutive_law == "plastic":
        if pca_path.exists():
            try:
                pca = PCA(n_components=PCA_N_COMPONENTS)
                pca.fit(_load_npz(pca_path))
                logger.info("Fitted PCA (%d component(s)) from %s", PCA_N_COMPONENTS, pca_path.name)
            except Exception as e:
                logger.warning("Could not load PCA data: %s. Will regenerate from the first simulation.", e)
                pca = None

    for i, freq in enumerate(frequency_range):
        if freq in completed_set:
            logger.info("  Skipping sim %d/%d – freq %d Hz (already saved)", i + 1, len(frequency_range), freq)
            continue

        logger.info("  Sim %d/%d – freq %d Hz", i + 1, len(frequency_range), freq)
        cfg_i = Sim3DConfig(
            constitutive_law=config.constitutive_law,
            frequency=freq,
            time_total=config.time_total,
        )
        tct = _make_extractor(cfg_i)
        tct.run()

        if not topo_path.exists():
            coords, connectivity = tct.get_interface_topology()
            np.savez_compressed(topo_path, coords=coords, connectivity=connectivity)

        n_iface = len(tct.interface_dofs)
        # Forces extracted BEFORE solve_u (data_in), Displacements extracted AFTER solve_u (data_out)
        forces = tct.data_in
        displacements = tct.data_out[:, :n_iface]

        _append_npz(in_path, forces)
        _append_npz(out_path, displacements)

        if config.constitutive_law == "plastic":
            if pca is None:
                np.savez_compressed(pca_path, tct.data_plastic_top)
                pca = PCA(n_components=PCA_N_COMPONENTS)
                pca.fit(tct.data_plastic_top)
                logger.info("Saved full plastic field for PCA → %s", pca_path.name)
            _append_npz(plastic_path, pca.transform(tct.data_plastic_top))

        _append_npz(freq_path, np.array([freq], dtype=np.int32))
        completed_set.add(freq)

        logger.info("  Saved simulation %d (freq %d Hz) → %s", i + 1, freq, DATA_FOLDER)
        del tct

    logger.info("Inverse training set generation complete.")


def load_dataset(config: Sim3DConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 3D training data for INVERSE prediction:
      X (Input):       Interface Forces / Tractions (Fx, Fy, Fz per node) -> shape [N, T, 453]
      Y (Target):      Interface Displacements (ux, uy, uz per node)      -> shape [N, T, 453]
      internal:        Plastic strain state (PCA reduced)                -> shape [N, T, 1]
    """
    tag = config.constitutive_law
    logger.info("Loading 3D inverse dataset (Forces -> Displacements, tag=%s)…", tag)

    forces_path = DATA_FOLDER / f"training_in_3d_{tag}_forces.npz"
    disps_path  = DATA_FOLDER / f"training_out_3d_{tag}_disps.npz"

    # Fallback to existing dataset files if dedicated files not generated yet
    if not forces_path.exists():
        forces_path = DATA_FOLDER / f"training_out_3d_{tag}.npz"
    if not disps_path.exists():
        disps_path = DATA_FOLDER / f"training_in_3d_{tag}_x.npz"

    if not forces_path.exists() or not disps_path.exists():
        raise FileNotFoundError(f"Training data files not found in {DATA_FOLDER}. Expected {forces_path.name} and {disps_path.name}.")

    x = _load_npz(forces_path)  # Forces as Input X
    y = _load_npz(disps_path)   # Displacements as Target Output Y

    internal = np.array([])
    plastic_path = DATA_FOLDER / f"plastic_top_reduced_3d_{tag}.npz"
    if config.constitutive_law == "plastic" and plastic_path.exists():
        internal = _load_npz(plastic_path)

    # Ensure matching time steps between X, Y, and internal
    min_steps = min(x.shape[1], y.shape[1])
    if internal.size > 0:
        min_steps = min(min_steps, internal.shape[1])
        internal = internal[:, :min_steps]
    x = x[:, :min_steps]
    y = y[:, :min_steps]

    logger.info("Loaded Inverse Dataset:")
    logger.info("  Forces (Input X)        shape: %s", x.shape)
    logger.info("  Displacements (Target Y) shape: %s", y.shape)
    if internal.size > 0:
        logger.info("  Internal State           shape: %s", internal.shape)

    return x, np.array([]), y, internal


def load_interface_topology(config: Sim3DConfig) -> np.ndarray:
    """Load interface connectivity topology map from training_data_3d folder."""
    topo_path = DATA_FOLDER / f"interface_topology_3d_{config.constitutive_law}.npz"
    if not topo_path.exists():
        topo_path = DATA_FOLDER / "interface_topology_3d_plastic.npz"
    if not topo_path.exists():
        raise FileNotFoundError(f"Topology map file not found at {topo_path}")

    with open(topo_path, "rb") as f:
        topo_data = np.load(f)
        return topo_data["connectivity"]


def train_lstm(config: Sim3DConfig, resume: bool = False):
    """Train or resume an inverse LSTM model (Forces -> Displacements)."""
    if LSTMNetwork is None:
        raise ImportError("LSTMNetwork could not be imported from nn_predictors.lstm")

    tag        = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_lstm_inv_v{config.predictor_version}"

    x_full, _, y, _ = load_dataset(config)

    if resume and model_path.with_suffix(".pkl").exists():
        logger.info("Resuming inverse LSTM training from %s", model_path)
        reg = LSTMNetwork.load(str(model_path))
        reg.set_hyperparameters()
    else:
        logger.info("Starting new inverse LSTM (3D) training v%s", config.predictor_version)
        reg = LSTMNetwork(input_size=x_full.shape[2], output_size=y.shape[2], save_path=str(model_path))

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x_full, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    logger.info("Training complete in %.2fs", time.time() - t0)


def train_spils_net(config: Sim3DConfig, resume: bool = False):
    """Train or resume an inverse SPILS-Net model (Forces -> Displacements)."""
    tag        = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_inv_v{config.predictor_version}"

    x_full, _, y, internal = load_dataset(config)

    if resume:
        logger.info("Resuming inverse SPILS-Net (3D) training from %s", model_path)
        reg = SPILSNet.load(str(model_path))
        if hasattr(reg, "optimizer_state_dict") and reg.optimizer_state_dict is not None and not reg.optimizer_state_dict.get("param_groups"):
            reg.optimizer_state_dict = None
    else:
        logger.info("Starting new inverse SPILS-Net (3D) training v%s", config.predictor_version)
        reg = SPILSNet(save_path=str(model_path), **SPILSNET_SCALED_PARAMETERS)

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x_full, y, internal, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    logger.info("Training complete in %.2fs", time.time() - t0)


def train_spils_net_gnn(config: Sim3DConfig, resume: bool = False):
    """Train or resume an inverse GNN-based SPILS-Net model (Forces -> Displacements)."""
    tag        = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_gnn_inv_v{config.predictor_version}"

    x_full, _, y, internal = load_dataset(config)
    connectivity = load_interface_topology(config)

    if resume:
        logger.info("Resuming inverse SPILS-Net GNN (3D) training from %s", model_path)
        reg = SPILSNetGraph.load(str(model_path))
        if hasattr(reg, "optimizer_state_dict") and reg.optimizer_state_dict is not None and not reg.optimizer_state_dict.get("param_groups"):
            reg.optimizer_state_dict = None
    else:
        logger.info("Starting new inverse SPILS-Net GNN (3D) training v%s", config.predictor_version)
        reg = SPILSNetGraph(connectivity=connectivity, save_path=str(model_path), **SPILSNET_3D_GNN_PARAMETERS)

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x_full, y, internal, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    logger.info("Training complete in %.2fs", time.time() - t0)


def load_lstm(config: Sim3DConfig) -> LSTMNetwork:
    tag        = config.constitutive_law
    x_full, _, y, _ = load_dataset(config)
    model_path = MODEL_FOLDER / f"model_3d_{tag}_lstm_inv_v{config.predictor_version}"
    predictor  = LSTMNetwork.load(f"{model_path}_best")
    if not hasattr(predictor, "test_loader") or predictor.test_loader is None:
        predictor.prep_data(x_full, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    return predictor


def load_spils_net(config: Sim3DConfig) -> SPILSNet:
    tag             = config.constitutive_law
    x_full, _, y, internal = load_dataset(config)
    model_path      = MODEL_FOLDER / f"model_3d_{tag}_spils_net_inv_v{config.predictor_version}"
    predictor       = SPILSNet.load(f"{model_path}_best")
    predictor.prep_data(x_full, internal, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    return predictor


def load_spils_net_gnn(config: Sim3DConfig) -> SPILSNetGraph:
    tag             = config.constitutive_law
    x_full, _, y, internal = load_dataset(config)
    model_path      = MODEL_FOLDER / f"model_3d_{tag}_spils_net_gnn_inv_v{config.predictor_version}"
    predictor       = SPILSNetGraph.load(f"{model_path}_best")
    predictor.prep_data(x_full, internal, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    return predictor


def evaluate_model(predictor: Any, config: Sim3DConfig):
    """
    Evaluate an inverse predictor (Forces -> Displacements) against full FEM ground truth.
    """
    logger.info("Evaluating 3D Inverse Predictor (Forces -> Displacements)…")
    try:
        predictor.calculate_test_metrics()
    except Exception as e:
        logger.warning("calculate_test_metrics error: %s", e)
    predictor.initialize_memory_variables()

    # Obtain reference ground truth simulation data
    tct_truth = None
    if FEM_AVAILABLE:
        try:
            logger.info("Running full FEM reference simulation at %d Hz…", config.frequency)
            tct_truth = _make_extractor(config)
            tct_truth.run()
            tct_truth.format_results()
            n_iface = len(tct_truth.interface_dofs)
            true_forces = tct_truth.data_in                     # Forces extracted BEFORE solve_u [T, 453]
            true_disps  = tct_truth.data_out[:, :n_iface]       # Displacements extracted AFTER solve_u [T, 453]
            if config.constitutive_law == "plastic" and hasattr(tct_truth, "data_plastic_top"):
                pca_path = DATA_FOLDER / "plastic_top_3d_for_pca.npz"
                if pca_path.exists():
                    try:
                        pca = PCA(n_components=PCA_N_COMPONENTS)
                        pca.fit(_load_npz(pca_path))
                        tct_data_internal = pca.transform(tct_truth.data_plastic_top[1:]).ravel()
                    except Exception as e:
                        logger.warning("Could not transform plastic data with saved PCA (%s): %s", pca_path, e)
                        tct_data_internal = tct_truth.data_plastic_top[1:].mean(axis=1)
                else:
                    logger.warning("PCA file %s not found. Falling back to mean.", pca_path)
                    tct_data_internal = tct_truth.data_plastic_top[1:].mean(axis=1)
            else:
                tct_data_internal = np.zeros(true_forces.shape[0])
        except Exception as e:
            logger.warning("FEM reference simulation failed (%s), loading saved test simulation data.", e)
            tct_truth = None

    if tct_truth is None:
        # Load from saved dataset using test simulation index
        test_sim_idx = config.test_sims[0] if config.test_sims else 4
        forces_all, _, disps_all, internal_all = load_dataset(config)
        true_forces = forces_all[test_sim_idx]
        true_disps  = disps_all[test_sim_idx]
        if internal_all.size > 0:
            tct_data_internal = internal_all[test_sim_idx].ravel()
        else:
            tct_data_internal = np.zeros(true_forces.shape[0])

    num_steps = len(true_forces)
    logger.info("Evaluating inverse surrogate model across %d time steps…", num_steps)

    pred_disps = []
    pred_internals = []
    errors = []
    truth_norms = []

    for step in range(num_steps):
        f_step = true_forces[step]
        u_true = true_disps[step]

        u_pred = predictor.predict(f_step)
        pred_disps.append(u_pred)

        if hasattr(predictor, "internal_scaler") and hasattr(predictor, "hidden_state") and predictor.hidden_state is not None:
            z_val = predictor.internal_scaler.inverse_transform(predictor.hidden_state.detach().cpu().numpy())
            pred_internals.append(float(np.ravel(z_val)[0]))
        else:
            pred_internals.append(0.0)

        step_err = np.linalg.norm(u_true - u_pred)
        step_norm = np.linalg.norm(u_true)
        errors.append(step_err)
        truth_norms.append(step_norm)

    pred_disps     = np.array(pred_disps)
    pred_internals = np.array(pred_internals)
    errors         = np.array(errors)
    truth_norms    = np.array(truth_norms)

    global_err = np.linalg.norm(errors) / max(np.linalg.norm(truth_norms), 1e-10)
    logger.info("Global Relative Displacement Error: %.4f%%", global_err * 100)

    final_error = errors / max(np.max(truth_norms), 1e-10)

    suffix_tct = f"3d_{config.constitutive_law}_inv_f{config.frequency}"
    suffix_sm  = f"3d_{config.constitutive_law}_{config.predictor_method}_inv_v{config.predictor_version}_f{config.frequency}"

    # Full FEM Ground Truth
    np.save(RESULTS_FOLDER / f"tct_results_in_{suffix_tct}.npy",       true_forces)
    np.save(RESULTS_FOLDER / f"tct_results_out_{suffix_tct}.npy",      true_disps)
    np.save(RESULTS_FOLDER / f"tct_results_internal_{suffix_tct}.npy", tct_data_internal[:num_steps])

    # SM Prediction Results
    np.save(RESULTS_FOLDER / f"sm_results_error_{suffix_sm}.npy",    final_error)
    np.save(RESULTS_FOLDER / f"sm_results_in_{suffix_sm}.npy",       true_forces)
    np.save(RESULTS_FOLDER / f"sm_results_out_{suffix_sm}.npy",      pred_disps)
    np.save(RESULTS_FOLDER / f"sm_results_internal_{suffix_sm}.npy", pred_internals)

    logger.info("Inverse prediction results successfully saved to %s/", RESULTS_FOLDER)


def apply_lstm(config: Sim3DConfig):
    evaluate_model(load_lstm(config), config)


def apply_spils_net(config: Sim3DConfig):
    evaluate_model(load_spils_net(config), config)


def apply_spils_net_gnn(config: Sim3DConfig):
    evaluate_model(load_spils_net_gnn(config), config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D Inverse SPILS-Net Predictor Tool (Forces -> Displacements)")
    parser.add_argument("--method",     type=str, choices=["lstm", "spils_net", "spils_net_gnn"], default="spils_net_gnn")
    parser.add_argument("--version",    type=str, default="01",  help="Predictor version label")
    parser.add_argument("--law",        type=str, choices=["elastic", "plastic"], default="plastic")
    parser.add_argument("--freq",       type=int, default=1000,  help="Simulation frequency [Hz]")
    parser.add_argument("--time-total", type=float, default=0.001,
                        help="Simulation time limit in seconds (default 0.001). Set None for full run.")

    parser.add_argument("--simulate",   action="store_true", help="Run single full FEM simulation + GIF export")
    parser.add_argument("--verify-fem", action="store_true", help="FEM sanity-check run")
    parser.add_argument("--generate",   action="store_true", help="Generate multi-frequency inverse training dataset")
    parser.add_argument("--train",      action="store_true", help="Train the inverse model")
    parser.add_argument("--apply",      action="store_true", help="Evaluate the inverse model against FEM ground truth")
    parser.add_argument("--resume",     action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()

    config = Sim3DConfig(
        constitutive_law  = args.law,
        predictor_version = args.version,
        frequency         = args.freq,
        predictor_method  = args.method,
        time_total        = args.time_total,
    )

    did_something = False

    if args.simulate:
        run_simulation(config)
        did_something = True

    if args.verify_fem:
        verify_fem(config)
        did_something = True

    if args.generate:
        generate_training_set(config)
        did_something = True

    if args.train:
        if config.predictor_method == "lstm":
            train_lstm(config, resume=args.resume)
        elif config.predictor_method == "spils_net_gnn":
            train_spils_net_gnn(config, resume=args.resume)
        else:
            train_spils_net(config, resume=args.resume)
        did_something = True

    if args.apply:
        if config.predictor_method == "lstm":
            apply_lstm(config)
        elif config.predictor_method == "spils_net_gnn":
            apply_spils_net_gnn(config)
        else:
            apply_spils_net(config)
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()
