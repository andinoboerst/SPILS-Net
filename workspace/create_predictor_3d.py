"""
3D Predictor Creation and Evaluation Script for SPILS-Net.

Mirrors the functionality of create_predictor.py but for the 3D cylindrical
TCT (Tension-Compression-Tension) simulation domain.

Key additions over create_predictor.py:
  --verify-fem    Runs the full extractor, then applies the extracted tractions
                  back in a reduced-domain simulation, and computes the
                  displacement error — useful for sanity-checking before
                  training any ML model.
"""

import argparse
import logging
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
logger = logging.getLogger("Predictor3D")


# ---------------------------------------------------------------------------
# FEM imports  (only available inside Docker / with dolfinx installed)
# ---------------------------------------------------------------------------
try:
    from fem_sim.tct_tractions_3d import TCTExtractTractions3D, TCTApplyTractions3D
    from fem_sim.tct_sims_3d import tct_comp_3d
    from fem_sim.progress_bar import progressbar
    FEM_AVAILABLE = True
except (RuntimeError, ModuleNotFoundError) as _e:
    logger.warning("FEM packages not available – FEM-dependent commands will fail. (%s)", _e)
    FEM_AVAILABLE = False

# Neural-network predictors
from nn_predictors.misc import train_val_test_split_tct
from nn_predictors.lstm import LSTMNetwork
from spilsnet import SPILSNet, set_seed
from spilsnet_params import SPILSNET_SCALED_PARAMETERS

set_seed(8)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
DATA_FOLDER    = Path("training_data_3d")
MODEL_FOLDER   = Path("surrogate_models_3d")
RESULTS_FOLDER = Path("results_3d")

for _folder in [DATA_FOLDER, MODEL_FOLDER, RESULTS_FOLDER]:
    _folder.mkdir(parents=True, exist_ok=True)


DEFAULT_TRAIN_SIMS, DEFAULT_VAL_SIMS, DEFAULT_TEST_SIMS = train_val_test_split_tct()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Sim3DConfig:
    """Holds configuration for a 3D simulation problem."""

    constitutive_law: str       = "elastic"
    predictor_version: str      = "01"
    frequency: int              = 1000
    predictor_method: str       = "spils_net"
    # Hard limit for memory-constrained runs; set to None for a full run.
    time_total: float | None    = 0.002

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


def _make_extractor(config: Sim3DConfig) -> "TCTExtractTractions3D":
    sim = TCTExtractTractions3D(
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration="scaled",
    )
    if config.time_total is not None:
        sim.time_total = config.time_total
    return sim


def _make_applicator(predictor, config: Sim3DConfig) -> "TCTApplyTractions3D":
    sim = TCTApplyTractions3D(
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
    logger.info("Running 3D TCT simulation  (freq=%s Hz, law=%s)…", config.frequency, config.constitutive_law)
    sim = _make_extractor(config)
    sim.run()
    name = f"simulate_3d_{config.constitutive_law}_f{config.frequency}"
    logger.info("Exporting animation → %s.gif", name)
    sim.postprocess("u", "u", "norm", name=name)


def verify_fem(config: Sim3DConfig):
    """
    FEM-only verification:
    1. Run the full extractor to obtain interface tractions.
    2. Feed those *true* tractions directly into the applicator (bottom-half
       reduced domain) — bypassing any ML predictor.
    3. Compare displacement fields and report error.

    This checks whether the FEM traction-coupling machinery works before
    we involve a surrogate model.
    """
    logger.info("=== FEM Verification (extract → apply → compare) ===")

    class _TrueTractionPredictor:
        """Trivial 'predictor' that replays ground-truth tractions."""
        def __init__(self, traction_sequence: np.ndarray) -> None:
            self._tractions = traction_sequence  # shape (num_steps, n_dofs)
            self._step = 0

        def initialize_memory_variables(self) -> None:
            self._step = 0

        def predict(self, *args, **kwargs) -> np.ndarray:
            t = self._tractions[self._step]
            self._step += 1
            return t

    # --- Step 1: extract tractions ---
    logger.info("Step 1 / 2  –  Running full-domain FEM extractor…")
    extractor = _make_extractor(config)
    extractor.run()

    true_tractions = extractor.data_out  # (num_steps, n_interface_dofs)

    # --- Step 2: apply true tractions in the reduced domain ---
    logger.info("Step 2 / 2  –  Applying extracted tractions to reduced domain…")

    predictor_replay = _TrueTractionPredictor(true_tractions)
    applicator = _make_applicator(predictor_replay, config)
    applicator.run()

    # --- Step 3: compute displacement error in the bottom-half nodes ---
    logger.info("Computing displacement error…")

    extractor.format_results()
    applicator.format_results()

    # Bottom-half DOF indices in the extractor mesh (full domain)
    ext_u_bottom = extractor.formatted_plot_results["u"][:, extractor.bottom_half_nodes, :]

    # Corresponding DOF indices in the applicator mesh (bottom half only)
    app_u = applicator.formatted_plot_results["u"]  # all nodes = bottom half

    # Relative L2 error (time-averaged)
    diff_norm   = np.linalg.norm(ext_u_bottom - app_u, axis=-1).mean()
    truth_norm  = np.linalg.norm(ext_u_bottom, axis=-1).mean()
    rel_error   = diff_norm / max(truth_norm, 1e-12)

    logger.info("Verification complete.  Relative displacement error: %.4f%%", rel_error * 100)

    # Export GIFs for visual inspection
    suffix = f"verify_3d_{config.constitutive_law}_f{config.frequency}"

    # Full-domain: z-displacement
    extractor.postprocess("u", "u", "norm", name=f"{suffix}_full")

    # Reduced-domain: per-node displacement error magnitude  |u_ext - u_app|
    # Shape: (num_steps, n_bottom_nodes, 3)
    error_u = ext_u_bottom - app_u  # (time, nodes, 3)
    error_norm = np.linalg.norm(error_u, axis=-1)  # (time, nodes) – scalar per node

    # postprocess accepts a raw ndarray as `scalars`; vectors=None to disable warping
    applicator.postprocess(error_norm, "u", None, name=f"{suffix}_error")

    logger.info("Exported %s_full.gif and %s_error.gif", suffix, suffix)
    return rel_error


def generate_training_set(config: Sim3DConfig):
    """Generate a training dataset using multiple FEM simulations."""
    logger.info("Generating 3D training set  (law=%s)…", config.constitutive_law)

    frequency_range = range(500, 2001, int(1500 / 21))
    training_in, training_out, plastic_top = [], [], []

    pca_path = DATA_FOLDER / f"plastic_top_{config.constitutive_law}_for_pca.npz"
    pca = PCA(n_components=1)
    generate_pca_data = not (config.constitutive_law == "plastic" and pca_path.exists())

    for i, freq in enumerate(frequency_range):
        logger.info("  Sim %d/22 – freq %d Hz", i + 1, freq)
        cfg_i         = Sim3DConfig(constitutive_law=config.constitutive_law, frequency=freq, time_total=config.time_total)
        tct           = _make_extractor(cfg_i)
        tct.run()

        if i == 0 and generate_pca_data and config.constitutive_law == "plastic":
            with open(pca_path, "wb") as f:
                np.savez_compressed(f, tct.data_plastic_top)
            pca.fit(tct.data_plastic_top)

        training_in.append(tct.data_in)
        training_out.append(tct.data_out)
        if config.constitutive_law == "plastic":
            plastic_top.append(tct.data_plastic_top)
        del tct

    training_in = np.array(training_in)
    n_iface = training_in.shape[2] // 3  # input = [u | v | a], each of length n_iface

    tag = config.constitutive_law
    np.savez_compressed(DATA_FOLDER / f"training_in_{tag}_x.npz", training_in[:, :, :n_iface])
    np.savez_compressed(DATA_FOLDER / f"training_in_{tag}_v.npz", training_in[:, :, n_iface:2*n_iface])
    np.savez_compressed(DATA_FOLDER / f"training_in_{tag}_a.npz", training_in[:, :, 2*n_iface:])
    np.savez_compressed(DATA_FOLDER / f"training_out_{tag}.npz",  np.array(training_out))

    if config.constitutive_law == "plastic":
        np.savez_compressed(DATA_FOLDER / f"plastic_top_reduced_{tag}.npz", np.array(plastic_top))

    logger.info("Training set generation complete.")


def load_dataset(config: Sim3DConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data (X = [u|v|a], Y = tractions, internal = plastic state)."""
    tag = config.constitutive_law
    logger.info("Loading 3D dataset  (tag=%s)…", tag)
    x = _load_npz(DATA_FOLDER / f"training_in_{tag}_x.npz")
    v = _load_npz(DATA_FOLDER / f"training_in_{tag}_v.npz")
    a = _load_npz(DATA_FOLDER / f"training_in_{tag}_a.npz")
    y = _load_npz(DATA_FOLDER / f"training_out_{tag}.npz")

    internal = np.array([])
    if config.constitutive_law == "plastic":
        internal = _load_npz(DATA_FOLDER / f"plastic_top_reduced_{tag}.npz")

    # SPILS-Net expects input = [u | v | a] concatenated
    x_full = np.concatenate([x, v, a], axis=2)
    return x_full, v, y, internal


def train_lstm(config: Sim3DConfig, resume: bool = False):
    """Train or resume an LSTM model."""
    tag        = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_lstm_v{config.predictor_version}"

    x_full, _, y, _ = load_dataset(config)

    if resume and model_path.with_suffix(".pkl").exists():
        logger.info("Resuming LSTM training from %s", model_path)
        reg = LSTMNetwork.load(str(model_path))
        reg.set_hyperparameters()
    else:
        logger.info("Starting new LSTM (3D) training  v%s", config.predictor_version)
        reg = LSTMNetwork(input_size=x_full.shape[2], output_size=y.shape[2], save_path=str(model_path))

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x_full, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    logger.info("Training complete in %.2fs", time.time() - t0)


def train_spils_net(config: Sim3DConfig, resume: bool = False):
    """Train or resume a SPILS-Net model."""
    tag        = config.constitutive_law
    model_path = MODEL_FOLDER / f"model_3d_{tag}_spils_net_v{config.predictor_version}"

    x_full, _, y, internal = load_dataset(config)

    if resume:
        logger.info("Resuming SPILS-Net (3D) training from %s", model_path)
        reg = SPILSNet.load(str(model_path))
    else:
        logger.info("Starting new SPILS-Net (3D) training  v%s", config.predictor_version)
        reg = SPILSNet(save_path=str(model_path), **SPILSNET_SCALED_PARAMETERS)

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x_full, y, internal, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    logger.info("Training complete in %.2fs", time.time() - t0)


def load_lstm(config: Sim3DConfig) -> LSTMNetwork:
    tag        = config.constitutive_law
    x_full, _, y, _ = load_dataset(config)
    model_path = MODEL_FOLDER / f"model_3d_{tag}_lstm_v{config.predictor_version}"
    predictor  = LSTMNetwork.load(f"{model_path}_best")
    if not hasattr(predictor, "test_loader") or predictor.test_loader is None:
        predictor.prep_data(x_full, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    return predictor


def load_spils_net(config: Sim3DConfig) -> SPILSNet:
    tag             = config.constitutive_law
    x_full, _, y, internal = load_dataset(config)
    model_path      = MODEL_FOLDER / f"model_3d_{tag}_spils_net_v{config.predictor_version}"
    predictor       = SPILSNet.load(f"{model_path}_best")
    predictor.prep_data(x_full, internal, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    return predictor


def evaluate_model(predictor: Any, config: Sim3DConfig):
    """Evaluate a predictor against the full-domain FEM solution."""
    logger.info("Evaluating 3D predictor…")
    predictor.calculate_test_metrics()
    predictor.initialize_memory_variables()

    tct_predict = _make_applicator(predictor, config)
    tct_error   = _make_extractor(config)

    tct_predict.setup()
    tct_error.setup()

    tct_predict.export_results()
    tct_error.export_results()

    predict_dofs = tct_predict.get_dofs(tct_predict.bottom_half_nodes)
    error_dofs   = tct_error.get_dofs(tct_error.bottom_half_nodes)

    errors, truth_norms = [], []

    logger.info("Starting evaluation loop…")
    for step in progressbar(range(tct_predict.num_steps)):
        tct_predict.step = step
        tct_error.step   = step

        tct_predict.advance_time()
        tct_error.advance_time()

        tct_predict.solve_time_step()
        tct_error.solve_time_step()

        u_true = tct_error.u_next.x.array[error_dofs]
        u_pred = tct_predict.u_next.x.array[predict_dofs]

        errors.append(np.linalg.norm(u_true - u_pred))
        truth_norms.append(np.linalg.norm(u_true))

    errors       = np.array(errors)
    truth_norms  = np.array(truth_norms)
    global_err   = np.linalg.norm(errors) / max(np.linalg.norm(truth_norms), 1e-10)
    logger.info("Global Relative Error: %.4f%%", global_err * 100)

    final_error  = errors / max(np.max(truth_norms), 1e-10)
    suffix = f"3d_{config.constitutive_law}_{config.predictor_method}_v{config.predictor_version}_f{config.frequency}"

    tct_predict.format_results()
    tct_error.format_results()

    np.save(RESULTS_FOLDER / f"results_error_{suffix}.npy",    final_error)
    np.save(RESULTS_FOLDER / f"results_in_{suffix}.npy",       tct_predict.data_in)
    np.save(RESULTS_FOLDER / f"results_out_{suffix}.npy",      tct_predict.data_out)
    np.save(RESULTS_FOLDER / f"results_internal_{suffix}.npy", tct_predict.data_internal)

    logger.info("Results saved to %s/", RESULTS_FOLDER)


def apply_lstm(config: Sim3DConfig):
    evaluate_model(load_lstm(config), config)


def apply_spils_net(config: Sim3DConfig):
    evaluate_model(load_spils_net(config), config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D SPILS-Net Predictor Tool")
    parser.add_argument("--method",   type=str, choices=["lstm", "spils_net"], default="spils_net")
    parser.add_argument("--version",  type=str, default="01",  help="Predictor version label")
    parser.add_argument("--law",      type=str, choices=["elastic", "plastic"], default="plastic")
    parser.add_argument("--freq",     type=int, default=1000,  help="Simulation frequency [Hz]")
    parser.add_argument("--time-total", type=float, default=0.001,
                        help="Simulation time limit in seconds (default 0.002). "
                             "Set a large value or comment out in code for a full run.")

    parser.add_argument("--simulate",     action="store_true", help="Run a single full FEM simulation + GIF export")
    parser.add_argument("--verify-fem",   action="store_true", help="Extract tractions, re-apply them (no ML), compute error")
    parser.add_argument("--generate",     action="store_true", help="Generate multi-frequency training dataset")
    parser.add_argument("--train",        action="store_true", help="Train the model")
    parser.add_argument("--apply",        action="store_true", help="Evaluate the model against FEM")
    parser.add_argument("--resume",       action="store_true", help="Resume training from checkpoint")

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
        else:
            train_spils_net(config, resume=args.resume)
        did_something = True

    if args.apply:
        if config.predictor_method == "lstm":
            apply_lstm(config)
        else:
            apply_spils_net(config)
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()
