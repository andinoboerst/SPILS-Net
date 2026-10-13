"""
3D Discontinuous Inverse Predictor Creation and Evaluation Script for SPILS-Net ("disc").

Maps interface forces/tractions (Fx, Fy, Fz per node) to interface displacements
(ux, uy, uz per node) and internal state for a 3D cylindrical column connected at the base
(z <= 35) and top (z >= 65) into a single piece, with a central opening / hole in the middle
(z in [35, 65]) creating two circular interface sections at the mid-plane (z = 50.0).

Discontinuous Inverse Formulation:
  Input (X):  Interface Forces/Tractions across both circular sections (N_nodes * 3 DOFs)
  Output (Y): Interface Displacements across both circular sections (N_nodes * 3 DOFs)
  Internal:   Top-half plastic strain state (PCA reduced, 1 feature)

Key CLI flags:
  --verify-fem    Runs the full extractor, then applies the extracted displacements
                  directly to the bottom-half mesh (bypassing ML), and computes
                  the relative displacement error. Use this as a sanity check.
  --simulate      Runs a single full FEM simulation and exports a deformation GIF.
  --generate      Generates multi-frequency discontinuous inverse dataset.
  --train         Trains the inverse model (SPILS-Net GNN, SPILS-Net, or LSTM).
  --apply         Evaluates trained model against FEM ground truth and exports error GIFs.
"""

import argparse
import copy
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
logger = logging.getLogger("Predictor3D_Disc")


# ---------------------------------------------------------------------------
# FEM imports (available inside Docker / with dolfinx and gmsh installed)
# ---------------------------------------------------------------------------
try:
    from fem_sim.tct_displacements_3d_disc import (
        TCTExtractDisplacements3DDisc,
        TCTApplyDisplacements3DDisc,
    )
    from fem_sim.tct_sims_3d_disc import tct_comp_3d_disc
    from fem_sim.progress_bar import progressbar

    FEM_AVAILABLE = True
except (RuntimeError, ModuleNotFoundError, ImportError) as _e:
    logger.warning(
        "FEM packages not available – FEM-dependent commands will fallback to saved/synthetic datasets. (%s)",
        _e,
    )
    FEM_AVAILABLE = False

# Neural-network predictors
if not os.environ.get("USE_PYPI_SPILSNET"):
    for _p in [
        Path("/spilsnet-torch"),
        Path("../spilsnet-torch"),
        Path("../../spilsnet-torch"),
    ]:
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
from spilsnet_params import (
    SPILSNET_SCALED_PARAMETERS,
    SPILSNET_3D_DISC_GNN_PARAMETERS,
)

set_seed(8)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
DATA_FOLDER = Path("training_data_3d_disc_inv")
if not (DATA_FOLDER / "training_in_3d_plastic_forces.npz").exists() and (
    Path("workspace/training_data_3d_disc_inv") / "training_in_3d_plastic_forces.npz"
).exists():
    DATA_FOLDER = Path("workspace/training_data_3d_disc_inv")

MODEL_FOLDER = Path("surrogate_models_3d_disc_inv")
if not MODEL_FOLDER.exists() and Path("workspace/surrogate_models_3d_disc_inv").exists():
    MODEL_FOLDER = Path("workspace/surrogate_models_3d_disc_inv")

RESULTS_FOLDER = Path("results_3d_disc_inv")
if not RESULTS_FOLDER.exists() and Path("workspace/results_3d_disc_inv").exists():
    RESULTS_FOLDER = Path("workspace/results_3d_disc_inv")

for _folder in [DATA_FOLDER, MODEL_FOLDER, RESULTS_FOLDER]:
    _folder.mkdir(parents=True, exist_ok=True)


DEFAULT_TRAIN_SIMS, DEFAULT_VAL_SIMS, DEFAULT_TEST_SIMS = train_val_test_split_tct()
PCA_N_COMPONENTS = 1


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Sim3DDiscConfig:
    """Holds configuration for a 3D discontinuous simulation problem."""

    constitutive_law: str = "plastic"
    predictor_version: str = "01"
    frequency: int = 1000
    predictor_method: str = "spils_net_gnn"
    # Hard limit for memory-constrained runs; set to None for a full run.
    time_total: float | None = 0.001

    train_sims: List[int] = field(default_factory=lambda: list(DEFAULT_TRAIN_SIMS))
    val_sims: List[int] = field(default_factory=lambda: list(DEFAULT_VAL_SIMS))
    test_sims: List[int] = field(default_factory=lambda: list(DEFAULT_TEST_SIMS))


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


def _make_extractor(config: Sim3DDiscConfig) -> "TCTExtractDisplacements3DDisc":
    sim = TCTExtractDisplacements3DDisc(
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration="scaled",
    )
    if config.time_total is not None:
        sim.time_total = config.time_total
    return sim


def _make_applicator(predictor, config: Sim3DDiscConfig) -> "TCTApplyDisplacements3DDisc":
    sim = TCTApplyDisplacements3DDisc(
        predictor,
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration="scaled",
    )
    if config.time_total is not None:
        sim.time_total = config.time_total
    return sim


# ---------------------------------------------------------------------------
# Offline / Synthetic Data Generator for Discontinuous Interface
# ---------------------------------------------------------------------------
def _generate_synthetic_discontinuous_dataset(
    config: Sim3DDiscConfig,
    n_nodes_per_section: int = 75,
    num_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates realistic offline benchmark data for two circular interface sections
    separated by a central hole when running outside full Docker/FEM environment.
    """
    logger.info("Generating synthetic discontinuous 3D interface benchmark data...")
    tag = config.constitutive_law

    # 1. Create two disjoint lateral interface sections on the cylinder cross-section (R=25)
    # cut by the horizontal through-hole of radius R_hole=10 (|x| <= 10 removed)
    # Left section: x in [-25, -10]
    # Right section: x in [+10, +25]
    rng = np.random.default_rng(42)
    coords_list = []
    connectivity_list = []

    for section_idx, sign in enumerate([-1.0, 1.0]):
        x_pts, y_pts = [], []
        while len(x_pts) < n_nodes_per_section:
            cand_x = sign * rng.uniform(10.0, 25.0)
            y_max = np.sqrt(max(0.0, 25.0**2 - cand_x**2))
            cand_y = rng.uniform(-y_max, y_max)
            x_pts.append(cand_x)
            y_pts.append(cand_y)

        x = np.array(x_pts)
        y = np.array(y_pts)
        z = np.full(n_nodes_per_section, 50.0)

        section_coords = np.column_stack([x, y, z])
        offset = section_idx * n_nodes_per_section

        # Delaunay-style triangulation for the section
        from scipy.spatial import Delaunay

        tri = Delaunay(section_coords[:, :2])
        for simplex in tri.simplices:
            connectivity_list.append(simplex + offset)

        coords_list.append(section_coords)

    all_coords = np.vstack(coords_list)
    all_conn = np.array(connectivity_list, dtype=np.int32)
    total_nodes = all_coords.shape[0]
    total_dofs = total_nodes * 3

    # 2. Time series responses
    freqs = list(range(500, 2001, int(1500 / 21)))
    n_sims = len(freqs)

    x_all = np.zeros((n_sims, num_steps, total_dofs), dtype=np.float32)
    y_all = np.zeros((n_sims, num_steps, total_dofs), dtype=np.float32)
    internal_all = np.zeros((n_sims, num_steps, 1), dtype=np.float32)

    t = np.linspace(0, 0.002, num_steps)
    for s_idx, f in enumerate(freqs):
        omega = 2 * np.pi * f
        base_u = 0.05 * np.sin(omega * t)[:, None]
        base_f = 250.0 * np.cos(omega * t)[:, None]

        # Left section vs right section phase and amplitude
        for n in range(total_nodes):
            x_pos = all_coords[n, 0]
            ampl_scale = 1.0 + 0.15 * (x_pos / 13.0)
            y_all[s_idx, :, n * 3 + 2] = (base_u * ampl_scale).squeeze()
            x_all[s_idx, :, n * 3 + 2] = (base_f * ampl_scale).squeeze()

        internal_all[s_idx, :, 0] = np.maximum(0.0, 0.02 * (1.0 - np.exp(-1000.0 * t)))

    # Save to disk
    topo_path = DATA_FOLDER / f"interface_topology_3d_{tag}.npz"
    np.savez_compressed(topo_path, coords=all_coords, connectivity=all_conn)

    in_path = DATA_FOLDER / f"training_in_3d_{tag}_forces.npz"
    out_path = DATA_FOLDER / f"training_out_3d_{tag}_disps.npz"
    plastic_path = DATA_FOLDER / f"plastic_top_reduced_3d_{tag}.npz"
    freq_path = DATA_FOLDER / f"training_freqs_3d_{tag}_inv.npz"

    np.savez_compressed(in_path, x_all)
    np.savez_compressed(out_path, y_all)
    np.savez_compressed(plastic_path, internal_all)
    np.savez_compressed(freq_path, np.asarray(freqs, dtype=np.int32))

    logger.info("Saved synthetic discontinuous interface dataset to %s", DATA_FOLDER)
    return x_all, y_all, internal_all, all_coords, all_conn


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_simulation(config: Sim3DDiscConfig):
    """Run a single full FEM simulation and export a GIF."""
    if not FEM_AVAILABLE:
        logger.error("FEM packages not available for direct simulation.")
        return
    logger.info(
        "Running 3D discontinuous TCT simulation (freq=%s Hz, law=%s)…",
        config.frequency,
        config.constitutive_law,
    )
    sim = _make_extractor(config)
    sim.run()
    name = f"simulate_3d_{config.constitutive_law}_disc_f{config.frequency}"
    logger.info("Exporting animation → %s.gif", name)
    sim.postprocess("u", "u", "norm", name=name)


def verify_fem(config: Sim3DDiscConfig):
    """
    FEM-only verification for discontinuous interface:
    1. Run the full extractor to obtain interface displacements across both circular sections.
    2. Feed those *true* displacements directly into the applicator (bottom-half reduced domain).
    3. Compare displacement fields and report error.
    """
    if not FEM_AVAILABLE:
        logger.error("FEM packages not available for FEM verification.")
        return 0.0

    logger.info("=== Discontinuous FEM Verification (extract → apply → compare) ===")

    class _TrueDisplacementPredictor:
        def __init__(self, displacement_sequence: np.ndarray) -> None:
            self._displacements = displacement_sequence
            self._step = 0

        def initialize_memory_variables(self) -> None:
            self._step = 0

        def predict(self, *args, **kwargs) -> np.ndarray:
            u = self._displacements[self._step]
            self._step += 1
            return u

    logger.info("Step 1 / 2 – Running full-domain discontinuous FEM extractor…")
    extractor = _make_extractor(config)
    extractor.run()

    n_iface = len(extractor.interface_dofs)
    true_displacements = extractor.data_out[:, :n_iface]

    logger.info("Step 2 / 2 – Applying extracted displacements to reduced domain…")
    predictor_replay = _TrueDisplacementPredictor(true_displacements)
    applicator = _make_applicator(predictor_replay, config)
    applicator.run()

    logger.info("Computing displacement error…")
    extractor.format_results()
    applicator.format_results()

    ext_u_bottom = extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :]
    app_u = applicator.formatted_plot_results["u"]

    diff_norm = np.linalg.norm(ext_u_bottom - app_u, axis=-1).mean()
    truth_norm = np.linalg.norm(ext_u_bottom, axis=-1).mean()
    rel_error = diff_norm / max(truth_norm, 1e-12)

    logger.info("Verification complete. Relative displacement error: %.4f%%", rel_error * 100)

    suffix = f"verify_3d_disc_{config.constitutive_law}_f{config.frequency}"
    extractor.postprocess("u", "u", "norm", name=f"{suffix}_full")

    error_u = ext_u_bottom - app_u
    error_norm = np.linalg.norm(error_u, axis=-1)
    applicator.postprocess(error_norm, "u", None, name=f"{suffix}_error")

    logger.info("Exported %s_full.gif and %s_error.gif", suffix, suffix)
    return rel_error


def generate_training_set(config: Sim3DDiscConfig):
    """
    Generate a multi-frequency 3D discontinuous dataset with synchronized forces and displacements.
    """
    if not FEM_AVAILABLE:
        logger.warning("FEM packages not available. Generating benchmark offline dataset instead...")
        _generate_synthetic_discontinuous_dataset(config)
        return

    logger.info(
        "Generating 3D discontinuous training set (Forces -> Displacements, law=%s)…",
        config.constitutive_law,
    )

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
                logger.warning(
                    "Could not load PCA data: %s. Will regenerate from the first simulation.", e
                )
                pca = None

    for i, freq in enumerate(frequency_range):
        if freq in completed_set:
            logger.info(" Skipping sim %d/%d – freq %d Hz (already saved)", i + 1, len(frequency_range), freq)
            continue

        logger.info(" Sim %d/%d – freq %d Hz", i + 1, len(frequency_range), freq)
        cfg_i = Sim3DDiscConfig(
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

        logger.info(" Saved simulation %d (freq %d Hz) → %s", i + 1, freq, DATA_FOLDER)
        del tct

    logger.info("Discontinuous training set generation complete.")


def load_dataset(config: Sim3DDiscConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 3D discontinuous training data for INVERSE prediction:
      X (Input):       Interface Forces across two circular sections -> shape [N, T, n_dofs]
      Y (Target):      Interface Displacements across two circular sections -> shape [N, T, n_dofs]
      internal:        Plastic strain state (PCA reduced) -> shape [N, T, 1]
    """
    tag = config.constitutive_law
    logger.info("Loading 3D discontinuous inverse dataset (tag=%s)…", tag)

    forces_path = DATA_FOLDER / f"training_in_3d_{tag}_forces.npz"
    disps_path = DATA_FOLDER / f"training_out_3d_{tag}_disps.npz"

    if not forces_path.exists() or not disps_path.exists():
        logger.warning(
            "Dataset files not found in %s. Generating synthetic discontinuous benchmark dataset...",
            DATA_FOLDER,
        )
        x_all, y_all, int_all, _, _ = _generate_synthetic_discontinuous_dataset(config)
        return x_all, np.array([]), y_all, int_all

    x = _load_npz(forces_path)
    y = _load_npz(disps_path)

    internal = np.array([])
    plastic_path = DATA_FOLDER / f"plastic_top_reduced_3d_{tag}.npz"
    if config.constitutive_law == "plastic" and plastic_path.exists():
        internal = _load_npz(plastic_path)

    min_steps = min(x.shape[1], y.shape[1])
    if internal.size > 0:
        min_steps = min(min_steps, internal.shape[1])
        internal = internal[:, :min_steps]
    x = x[:, :min_steps]
    y = y[:, :min_steps]

    logger.info("Loaded Discontinuous Inverse Dataset:")
    logger.info("  Forces (Input X)        shape: %s", x.shape)
    logger.info("  Displacements (Target Y) shape: %s", y.shape)
    if internal.size > 0:
        logger.info("  Internal State           shape: %s", internal.shape)

    return x, np.array([]), y, internal


def load_interface_topology(config: Sim3DDiscConfig) -> np.ndarray:
    """Load discontinuous interface connectivity topology map."""
    tag = config.constitutive_law
    topo_path = DATA_FOLDER / f"interface_topology_3d_{tag}.npz"
    if not topo_path.exists():
        topo_path = DATA_FOLDER / "interface_topology_3d_plastic.npz"
    if not topo_path.exists():
        _generate_synthetic_discontinuous_dataset(config)
        topo_path = DATA_FOLDER / f"interface_topology_3d_{tag}.npz"

    with open(topo_path, "rb") as f:
        topo_data = np.load(f)
        return topo_data["connectivity"]


def train_lstm(config: Sim3DDiscConfig, resume: bool = False):
    """Train or resume an inverse LSTM model for discontinuous interface."""
    if LSTMNetwork is None:
        raise ImportError("LSTMNetwork could not be imported from nn_predictors.lstm")

    tag = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_lstm_disc_v{config.predictor_version}"

    x_full, _, y, _ = load_dataset(config)

    if resume and model_path.with_suffix(".pkl").exists():
        logger.info("Resuming discontinuous inverse LSTM training from %s", model_path)
        reg = LSTMNetwork.load(str(model_path))
        reg.set_hyperparameters()
    else:
        logger.info("Starting new discontinuous inverse LSTM training v%s", config.predictor_version)
        reg = LSTMNetwork(input_size=x_full.shape[2], output_size=y.shape[2], save_path=str(model_path))

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(
        x_full,
        y,
        train_indices=config.train_sims,
        val_indices=config.val_sims,
        test_indices=config.test_sims,
    )
    logger.info("Training complete in %.2fs", time.time() - t0)


def train_spils_net(config: Sim3DDiscConfig, resume: bool = False):
    """Train or resume an inverse SPILS-Net model for discontinuous interface."""
    tag = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_disc_v{config.predictor_version}"

    x_full, _, y, internal = load_dataset(config)

    scaled_params = copy.deepcopy(SPILSNET_SCALED_PARAMETERS)
    scaled_params["model_config"]["input_size"] = x_full.shape[2]

    if resume:
        logger.info("Resuming discontinuous SPILS-Net training from %s", model_path)
        reg = SPILSNet.load(str(model_path))
        if (
            hasattr(reg, "optimizer_state_dict")
            and reg.optimizer_state_dict is not None
            and not reg.optimizer_state_dict.get("param_groups")
        ):
            reg.optimizer_state_dict = None
    else:
        logger.info("Starting new discontinuous SPILS-Net training v%s", config.predictor_version)
        reg = SPILSNet(save_path=str(model_path), **scaled_params)

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(
        x_full,
        y,
        internal,
        train_indices=config.train_sims,
        val_indices=config.val_sims,
        test_indices=config.test_sims,
    )
    logger.info("Training complete in %.2fs", time.time() - t0)


def train_spils_net_gnn(config: Sim3DDiscConfig, resume: bool = False):
    """Train or resume an inverse GNN SPILS-Net model for discontinuous interface."""
    tag = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_gnn_disc_v{config.predictor_version}"

    x_full, _, y, internal = load_dataset(config)
    connectivity = load_interface_topology(config)

    gnn_params = copy.deepcopy(SPILSNET_3D_DISC_GNN_PARAMETERS)
    gnn_params["model_config"]["input_size"] = x_full.shape[2]

    if resume:
        logger.info("Resuming discontinuous SPILS-Net GNN training from %s", model_path)
        reg = SPILSNetGraph.load(str(model_path))
        if (
            hasattr(reg, "optimizer_state_dict")
            and reg.optimizer_state_dict is not None
            and not reg.optimizer_state_dict.get("param_groups")
        ):
            reg.optimizer_state_dict = None
    else:
        logger.info("Starting new discontinuous SPILS-Net GNN training v%s", config.predictor_version)
        reg = SPILSNetGraph(connectivity=connectivity, save_path=str(model_path), **gnn_params)

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(
        x_full,
        y,
        internal,
        train_indices=config.train_sims,
        val_indices=config.val_sims,
        test_indices=config.test_sims,
    )
    logger.info("Training complete in %.2fs", time.time() - t0)


def load_lstm(config: Sim3DDiscConfig) -> LSTMNetwork:
    tag = config.constitutive_law
    x_full, _, y, _ = load_dataset(config)
    model_path = MODEL_FOLDER / f"model_3d_{tag}_lstm_disc_v{config.predictor_version}"
    predictor = LSTMNetwork.load(f"{model_path}_best")
    if not hasattr(predictor, "test_loader") or predictor.test_loader is None:
        predictor.prep_data(
            x_full,
            y,
            train_indices=config.train_sims,
            val_indices=config.val_sims,
            test_indices=config.test_sims,
        )
    return predictor


def load_spils_net(config: Sim3DDiscConfig) -> SPILSNet:
    tag = config.constitutive_law
    x_full, _, y, internal = load_dataset(config)
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_disc_v{config.predictor_version}"
    predictor = SPILSNet.load(f"{model_path}_best")
    predictor.prep_data(
        x_full,
        internal,
        y,
        train_indices=config.train_sims,
        val_indices=config.val_sims,
        test_indices=config.test_sims,
    )
    return predictor


def load_spils_net_gnn(config: Sim3DDiscConfig) -> SPILSNetGraph:
    tag = config.constitutive_law
    x_full, _, y, internal = load_dataset(config)
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_gnn_disc_v{config.predictor_version}"
    predictor = SPILSNetGraph.load(f"{model_path}_best")
    predictor.prep_data(
        x_full,
        internal,
        y,
        train_indices=config.train_sims,
        val_indices=config.val_sims,
        test_indices=config.test_sims,
    )
    return predictor


def evaluate_model(predictor: Any, config: Sim3DDiscConfig):
    """
    Evaluate discontinuous inverse predictor against FEM ground truth or dataset.
    """
    tag = config.constitutive_law
    suffix_tct = f"3d_{tag}_disc_f{config.frequency}"
    suffix_sm = f"3d_{tag}_{config.predictor_method}_disc_v{config.predictor_version}_f{config.frequency}"

    if not FEM_AVAILABLE:
        logger.warning("FEM packages not available – evaluating predictor against test dataset slice.")
        x_full, _, y_full, internal_full = load_dataset(config)
        test_idx = config.test_sims[0] if config.test_sims else 4
        if test_idx >= x_full.shape[0]:
            test_idx = 0

        tct_in = x_full[test_idx]
        tct_out = y_full[test_idx]
        tct_internal = internal_full[test_idx] if internal_full.size > 0 else np.zeros(tct_in.shape[0])

        sm_pred = []
        predictor.initialize_memory_variables()
        for step in range(tct_in.shape[0]):
            pred = predictor.predict(tct_in[step])
            sm_pred.append(pred)
        sm_out = np.array(sm_pred)

        diff = tct_out - sm_out
        error_norm = np.linalg.norm(diff, axis=-1)
        truth_norm = np.linalg.norm(tct_out, axis=-1)
        rel_error = error_norm / np.maximum(np.max(truth_norm), 1e-10)

        np.save(RESULTS_FOLDER / f"tct_results_in_{suffix_tct}.npy", tct_in)
        np.save(RESULTS_FOLDER / f"tct_results_out_{suffix_tct}.npy", tct_out)
        np.save(RESULTS_FOLDER / f"tct_results_internal_{suffix_tct}.npy", tct_internal)

        np.save(RESULTS_FOLDER / f"sm_results_error_{suffix_sm}.npy", rel_error)
        np.save(RESULTS_FOLDER / f"sm_results_in_{suffix_sm}.npy", tct_in)
        np.save(RESULTS_FOLDER / f"sm_results_out_{suffix_sm}.npy", sm_out)
        np.save(RESULTS_FOLDER / f"sm_results_internal_{suffix_sm}.npy", tct_internal)

        logger.info("Evaluation complete against dataset. Mean relative error: %.4f%%", np.mean(rel_error) * 100)
        return

    logger.info("Evaluating 3D Discontinuous Inverse Predictor (Forces -> Displacements)…")
    try:
        predictor.calculate_test_metrics()
    except Exception as e:
        logger.warning("calculate_test_metrics error: %s", e)
    predictor.initialize_memory_variables()

    tct_predict = _make_applicator(predictor, config)
    tct_error = _make_extractor(config)

    tct_predict.setup()
    tct_error.setup()

    tct_predict.export_results()
    tct_error.export_results()

    predict_dofs = tct_predict.get_dofs(tct_predict.bottom_half_nodes)
    error_dofs = tct_error.get_dofs(tct_error.bottom_half_nodes)

    predict_iface_dofs = tct_predict.interface_dofs
    error_iface_dofs = tct_error.interface_dofs

    errors, truth_norms = [], []
    iface_errors, iface_truth_norms = [], []

    logger.info("Starting discontinuous evaluation loop…")
    for step in progressbar(range(tct_predict.num_steps)):
        tct_predict.step = step
        tct_error.step = step

        tct_predict.advance_time()
        tct_error.advance_time()

        tct_predict.solve_time_step()
        tct_error.solve_time_step()

        u_true = tct_error.u_next.x.array[error_dofs]
        u_pred = tct_predict.u_next.x.array[predict_dofs]

        errors.append(np.linalg.norm(u_true - u_pred))
        truth_norms.append(np.linalg.norm(u_true))

        u_true_iface = tct_error.u_next.x.array[error_iface_dofs]
        u_pred_iface = tct_predict.u_next.x.array[predict_iface_dofs]

        iface_errors.append(np.linalg.norm(u_true_iface - u_pred_iface))
        iface_truth_norms.append(np.linalg.norm(u_true_iface))

    errors = np.array(errors)
    truth_norms = np.array(truth_norms)
    global_err = np.linalg.norm(errors) / max(np.linalg.norm(truth_norms), 1e-10)
    logger.info("Global Relative Error: %.4f%%", global_err * 100)

    iface_errors = np.array(iface_errors)
    iface_truth_norms = np.array(iface_truth_norms)
    iface_err = np.linalg.norm(iface_errors) / max(np.linalg.norm(iface_truth_norms), 1e-10)
    logger.info("Interface Relative Error: %.4f%%", iface_err * 100)

    final_error = errors / max(np.max(truth_norms), 1e-10)

    tct_predict.format_results()
    tct_error.format_results()

    if config.constitutive_law == "plastic" and hasattr(tct_error, "data_plastic_top"):
        pca_path = DATA_FOLDER / "plastic_top_3d_for_pca.npz"
        if pca_path.exists():
            try:
                pca = PCA(n_components=PCA_N_COMPONENTS)
                pca.fit(_load_npz(pca_path))
                tct_data_internal = pca.transform(tct_error.data_plastic_top[1:]).ravel()
            except Exception as e:
                logger.warning("Could not transform plastic data with saved PCA: %s", e)
                tct_data_internal = tct_error.data_plastic_top[1:].mean(axis=1)
        else:
            tct_data_internal = tct_error.data_plastic_top[1:].mean(axis=1)
    else:
        tct_data_internal = np.zeros(tct_error.data_in.shape[0])

    # Full FEM Ground Truth
    np.save(RESULTS_FOLDER / f"tct_results_in_{suffix_tct}.npy", tct_error.data_in)
    np.save(
        RESULTS_FOLDER / f"tct_results_out_{suffix_tct}.npy",
        tct_error.data_out[:, : len(tct_error.interface_dofs)],
    )
    np.save(RESULTS_FOLDER / f"tct_results_internal_{suffix_tct}.npy", tct_data_internal)

    # SM Prediction Results
    np.save(RESULTS_FOLDER / f"sm_results_error_{suffix_sm}.npy", final_error)
    np.save(RESULTS_FOLDER / f"sm_results_in_{suffix_sm}.npy", tct_predict.data_in)
    np.save(
        RESULTS_FOLDER / f"sm_results_out_{suffix_sm}.npy",
        tct_predict.data_out[:, : len(tct_predict.interface_dofs)],
    )
    np.save(RESULTS_FOLDER / f"sm_results_internal_{suffix_sm}.npy", tct_predict.data_internal)

    logger.info("Discontinuous prediction results successfully saved to %s/", RESULTS_FOLDER)

    gif_suffix = (
        f"apply_3d_{config.constitutive_law}_{config.predictor_method}_disc_v{config.predictor_version}_f{config.frequency}"
    )
    logger.info("Exporting deformation animations (%s_full.gif and %s_error.gif)…", gif_suffix, gif_suffix)

    try:
        tct_error.postprocess("u", "u", "norm", name=f"{gif_suffix}_full")
    except Exception as e:
        logger.warning("Could not export full-domain GIF: %s", e)

    try:
        ext_u_bottom = tct_error.formatted_plot_results["u"][:, tct_error.bottom_half_nodes, :]
        app_u = tct_predict.formatted_plot_results["u"]
        error_u = ext_u_bottom - app_u
        error_norm = np.linalg.norm(error_u, axis=-1)

        tct_predict.postprocess(error_norm, "u", None, name=f"{gif_suffix}_error")
        logger.info("Deformation GIFs exported successfully to current directory.")
    except Exception as e:
        logger.warning("Could not export error-field GIF: %s", e)


def apply_lstm(config: Sim3DDiscConfig):
    evaluate_model(load_lstm(config), config)


def apply_spils_net(config: Sim3DDiscConfig):
    evaluate_model(load_spils_net(config), config)


def apply_spils_net_gnn(config: Sim3DDiscConfig):
    evaluate_model(load_spils_net_gnn(config), config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3D Discontinuous Interface SPILS-Net Predictor Tool (Forces -> Displacements)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["lstm", "spils_net", "spils_net_gnn"],
        default="spils_net_gnn",
    )
    parser.add_argument("--version", type=str, default="01", help="Predictor version label")
    parser.add_argument("--law", type=str, choices=["elastic", "plastic"], default="plastic")
    parser.add_argument("--freq", type=int, default=1000, help="Simulation frequency [Hz]")
    parser.add_argument(
        "--time-total",
        type=float,
        default=0.001,
        help="Simulation time limit in seconds (default 0.001). Set None for full run.",
    )

    parser.add_argument("--simulate", action="store_true", help="Run single full FEM simulation + GIF export")
    parser.add_argument(
        "--verify-fem",
        action="store_true",
        help="Extract displacements, re-apply them (no ML), compute error",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate multi-frequency discontinuous inverse training dataset",
    )
    parser.add_argument("--train", action="store_true", help="Train the discontinuous inverse model")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Evaluate the discontinuous inverse model against FEM ground truth",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()

    config = Sim3DDiscConfig(
        constitutive_law=args.law,
        predictor_version=args.version,
        frequency=args.freq,
        predictor_method=args.method,
        time_total=args.time_total,
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
