"""
Predictor Creation and Evaluation Script for SPILS-Net.

This script handles the generation of training data, training of LSTM and SPILS-Net models,
and evaluation of the predictors against FEM simulations.
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
from sklearn.decomposition import PCA


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PredictorTool")


# FEM Simulation imports
try:
    from fem_sim.tct_tractions import TCTExtractTractions as Extractor, TCTApplyTractions as Applicator
    from fem_sim.progress_bar import progressbar
except (RuntimeError, ModuleNotFoundError):
    logger.warning("FEM packages not installed. Cannot run FEM simulations in this script.")

# Neural Network predictors
from nn_predictors.misc import train_val_test_split_tct
from nn_predictors.lstm import LSTMNetwork
from spilsnet import SPILSNet, set_seed
from spilsnet_params import SPILSNET_BENCHMARK_PARAMETERS

set_seed(8)

# Project Constants
DATA_FOLDER = Path("training_data")
MODEL_FOLDER = Path("surrogate_models")
RESULTS_FOLDER = Path("results")

# Ensure directories exist
for folder in [DATA_FOLDER, MODEL_FOLDER, RESULTS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Default split indices from original repository
DEFAULT_TRAIN_SIMS, DEFAULT_VAL_SIMS, DEFAULT_TEST_SIMS = train_val_test_split_tct()


@dataclass
class SimulationConfig:
    """Holds configuration for a simulation problem."""
    constitutive_law: str = "plastic"
    problem_configuration: str = "benchmark"
    predictor_version: str = "02"
    frequency: int = 740
    predictor_method: str = "spils_net"
    data_version: str = "benchmark"  # Often matches problem_configuration

    # Training split indices
    train_sims: List[int] = field(default_factory=lambda: list(DEFAULT_TRAIN_SIMS))
    val_sims: List[int] = field(default_factory=lambda: list(DEFAULT_VAL_SIMS))
    test_sims: List[int] = field(default_factory=lambda: list(DEFAULT_TEST_SIMS))


def load_npz_file(file_path: Path) -> np.ndarray:
    """Loads the first array from a .npz file."""
    with open(file_path, "rb") as f:
        data = np.load(f)
        return data[list(data.keys())[0]]


def generate_training_set(config: SimulationConfig):
    """Generates a training dataset using FEM simulations."""
    logger.info(f"Generating training set for version: {config.data_version}")

    frequency_range = range(500, 2001, int(1500 / 21))
    training_in = []
    training_out = []
    plastic_top = []

    pca_data_path = DATA_FOLDER / f"plastic_top_{config.data_version}_for_pca.npz"
    pca = PCA(n_components=1)

    try:
        if config.constitutive_law == "plastic" and pca_data_path.exists():
            data_plastic_top = load_npz_file(pca_data_path)
            pca.fit(data_plastic_top)
            generate_pca_data = False
        else:
            generate_pca_data = True
    except Exception as e:
        logger.warning(f"Could not load PCA data: {e}. Will regenerate.")
        generate_pca_data = True

    logger.debug(f"Generate PCA data flag: {generate_pca_data}")

    for i, freq in enumerate(frequency_range):
        logger.info(f"Step {i+1}: Running Simulation for frequency: {freq}")
        tct = Extractor(
            frequency=freq,
            constitutive_model=config.constitutive_law,
            configuration=config.problem_configuration
        )
        tct.run()

        if i == 0 and generate_pca_data:
            if config.constitutive_law == "plastic":
                with open(pca_data_path, "wb") as f:
                    np.savez_compressed(f, tct.data_plastic_top)
                pca.fit(tct.data_plastic_top)

        training_in.append(tct.data_in)
        training_out.append(tct.data_out)

        if config.constitutive_law == "plastic":
            plastic_top.append(tct.data_plastic_top)

        del tct

    training_in = np.array(training_in)
    interface_nodes = training_in.shape[2] // 2

    # Save processed data
    np.savez_compressed(DATA_FOLDER / f"training_in_{config.data_version}_x.npz", training_in[:, :, :interface_nodes])
    np.savez_compressed(DATA_FOLDER / f"training_in_{config.data_version}_v.npz", training_in[:, :, interface_nodes:])
    np.savez_compressed(DATA_FOLDER / f"training_out_{config.data_version}.npz", np.array(training_out))

    if config.constitutive_law == "plastic":
        np.savez_compressed(DATA_FOLDER / f"plastic_top_reduced_{config.data_version}.npz", np.array(plastic_top))

    logger.info("Training set generation complete.")


def load_dataset(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads training data (X, V, Y, internal_states)."""
    version = config.data_version
    logger.info(f"Loading dataset: {version}")

    try:
        x = load_npz_file(DATA_FOLDER / f"training_in_{version}_x.npz")
        v = load_npz_file(DATA_FOLDER / f"training_in_{version}_v.npz")
        y = load_npz_file(DATA_FOLDER / f"training_out_{version}.npz")

        internal = np.array([])
        if config.constitutive_law == "plastic":
            internal = load_npz_file(DATA_FOLDER / f"plastic_top_reduced_{version}.npz")

        return x, v, y, internal
    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}")
        raise


def train_lstm(config: SimulationConfig, resume: bool = False):
    """Trains or resumes training for an LSTM model."""
    model_path = MODEL_FOLDER / f"model_data_{config.data_version}_lstm_v{config.predictor_version}"

    if resume and (model_path.with_suffix(".pkl")).exists():
        logger.info(f"Resuming LSTM training from {model_path}")
        reg = LSTMNetwork.load(str(model_path))
        reg.set_hyperparameters()
    else:
        logger.info(f"Starting new LSTM training for v{config.predictor_version}")
        x, v, y, _ = load_dataset(config)
        training_in = np.concatenate((x, v), axis=2)

        reg = LSTMNetwork(
            input_size=training_in.shape[2],
            output_size=y.shape[2],
            save_path=str(model_path)
        )

    logger.info(f"Trainable parameters: {reg.get_trainable_params():,}")

    # Data is needed for fit if it wasn't prepped
    if not resume:
        x, v, y, _ = load_dataset(config)
        training_in = np.concatenate((x, v), axis=2)
        reg.fit(training_in, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    else:
        # For resume, we still need the data if train_loader is None
        x, v, y, _ = load_dataset(config)
        training_in = np.concatenate((x, v), axis=2)
        reg.fit(training_in, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)


def train_spils_net(config: SimulationConfig, resume: bool = False):
    """Trains or resumes training for a SPILS-Net model."""
    model_name = f"model_data_{config.data_version}_spils_net_v{config.predictor_version}"
    model_path = MODEL_FOLDER / model_name

    x, _, y, internal = load_dataset(config)

    if resume:
        logger.info(f"Resuming SPILS-Net training for {model_name}")
        reg = SPILSNet.load(str(model_path))
    else:
        logger.info(f"Starting new SPILS-Net training for v{config.predictor_version}")
        reg = SPILSNet(save_path=str(model_path), **SPILSNET_BENCHMARK_PARAMETERS)

    logger.info(f"Trainable parameters: {reg.get_trainable_params():,}")

    start_time = time.time()
    reg.fit(
        x, y, internal,
        train_indices=config.train_sims,
        val_indices=config.val_sims,
        test_indices=config.test_sims
    )
    duration = time.time() - start_time
    logger.info(f"Training complete in {duration:.2f}s")


def evaluate_model(predictor: Any, config: SimulationConfig):
    """Evaluates a predictor by running a FEM simulation comparison."""
    logger.info("Evaluating predictor accuracy...")
    predictor.calculate_test_metrics()
    predictor.initialize_memory_variables()

    # Solve predicted vs true
    tct_predict = Applicator(
        predictor,
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration=config.problem_configuration
    )
    tct_error = Extractor(
        frequency=config.frequency,
        constitutive_model=config.constitutive_law,
        configuration=config.problem_configuration
    )

    tct_predict.setup()
    tct_error.setup()

    tct_predict.export_results()
    tct_error.export_results()

    predict_dofs = tct_predict.get_dofs(tct_predict.bottom_half_nodes)
    error_dofs = tct_error.get_dofs(tct_error.bottom_half_nodes)

    errors = []
    truth_norms = []

    logger.info("Starting simulation loop...")
    for step in progressbar(range(tct_predict.num_steps)):
        tct_predict.step = step
        tct_error.step = step

        tct_predict.advance_time()
        tct_error.advance_time()

        tct_predict.solve_time_step()
        tct_error.solve_time_step()

        u_true = tct_error.u_next.x.array[error_dofs]
        u_pred = tct_predict.u_next.x.array[predict_dofs]

        norm_diff = np.linalg.norm(u_true - u_pred)
        norm_true = np.linalg.norm(u_true)

        truth_norms.append(norm_true)
        errors.append(norm_diff)

    errors = np.array(errors)
    truth_norms = np.array(truth_norms)

    global_relative_error = np.linalg.norm(errors) / max(np.linalg.norm(truth_norms), 1e-10)
    logger.info(f"Evaluation finished. Global Relative Error: {global_relative_error:.4%}")

    # Save results
    final_error = errors / max(np.max(truth_norms), 1e-10)
    suffix = f"{config.data_version}_{config.predictor_method}_v{config.predictor_version}_freq{config.frequency}.npy"

    np.save(RESULTS_FOLDER / f"sim_results_error_{suffix}", final_error)
    np.save(RESULTS_FOLDER / f"sim_results_in_{suffix}", tct_predict.data_in)
    np.save(RESULTS_FOLDER / f"sim_results_out_{suffix}", tct_predict.data_out)
    np.save(RESULTS_FOLDER / f"sim_results_internal_{suffix}", tct_predict.data_internal)

    tct_predict.format_results()
    tct_error.format_results()


def apply_lstm(config: SimulationConfig):
    """Loads and applies an LSTM predictor."""
    x, v, y, _ = load_dataset(config)
    training_in = np.concatenate((x, v), axis=2)

    model_path = MODEL_FOLDER / f"model_data_{config.data_version}_lstm_v{config.predictor_version}"

    predictor = LSTMNetwork.load(f"{model_path}_best")
    if not hasattr(predictor, "test_loader") or predictor.test_loader is None:
        predictor.prep_data(training_in, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)

    evaluate_model(predictor, config)


def apply_spils_net(config: SimulationConfig):
    """Loads and applies a SPILS-Net predictor."""
    x, _, y, internal = load_dataset(config)

    model_path = MODEL_FOLDER / f"model_data_{config.data_version}_spils_net_v{config.predictor_version}"

    # Load the best model using the new unified load method
    predictor = SPILSNet.load(f"{model_path}_best")

    # Prepare data (test loader) for evaluation
    predictor.prep_data(x, internal, y, train_indices=config.train_sims, val_indices=config.val_sims, test_indices=config.test_sims)

    evaluate_model(predictor, config)


def main():
    parser = argparse.ArgumentParser(description="SPILS-Net Predictor Tool")
    parser.add_argument("--method", type=str, choices=["lstm", "spils_net"], default="spils_net", help="Prediction method")
    parser.add_argument("--version", type=str, default="04", help="Predictor version label")
    parser.add_argument("--data-version", type=str, default="benchmark", help="Data version to use")
    parser.add_argument("--law", type=str, choices=["elastic", "plastic"], default="plastic", help="Constitutive law")
    parser.add_argument("--freq", type=int, default=740, help="Simulation frequency")

    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--apply", action="store_true", help="Run application/evaluation")
    parser.add_argument("--generate", action="store_true", help="Generate training set if missing")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()

    config = SimulationConfig(
        constitutive_law=args.law,
        problem_configuration=args.data_version,
        predictor_version=args.version,
        frequency=args.freq,
        predictor_method=args.method,
        data_version=args.data_version
    )

    if args.generate:
        generate_training_set(config)

    if args.train:
        if config.predictor_method == "lstm":
            train_lstm(config, resume=args.resume)
        else:
            train_spils_net(config, resume=args.resume)

    if args.apply:
        if config.predictor_method == "lstm":
            apply_lstm(config)
        else:
            apply_spils_net(config)

    # Default behavior if no action specified
    if not (args.train or args.apply or args.generate):
        parser.print_help()


if __name__ == "__main__":
    main()
