import time
import numpy as np
import logging

from sklearn.decomposition import PCA

from fem_sim.tct_tractions import TCTExtractTractions as extractor, TCTApplyTractions as applicator
from fem_sim.progress_bar import progressbar

from nn_predictors.lstm import LSTMNetwork
from nn_predictors.spils_net import SPILSNet


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_FOLDER = "workspace/training_data"
MODEL_FOLDER = "workspace/surrogate_models"
RESULTS_FOLDER = "workspace/results"

constitutive_law = "plastic"
problem_configuration = "benchmark"
predictor_version = "v129"
frequency = 740
predictor_method = "spils_net"
training_set_exists = True
resume_training = True
predictor_model_exists = True


def generate_training_set(version: str):
    frequency_range = range(500, 2001, int(1500 / 21))

    training_in = []
    training_out = []

    plastic_top = []

    try:
        if constitutive_law == "plastic":
            with open(f"{DATA_FOLDER}/plastic_top_{version}_for_pca.npz", "rb") as f:
                data = np.load(f)
                data_plastic_top = data[list(data.keys())[0]]

            pca = PCA(n_components=1)
            pca.fit(data_plastic_top)

        generate_pca_data = False
    except Exception:
        generate_pca_data = True

    logger.info(f"Generate PCA data: {generate_pca_data}")

    for i, frequency in enumerate(frequency_range):
        logger.info(f"Running Simulation for frequency: {frequency}")
        tct = extractor(frequency=frequency, constitutive_model=constitutive_law, configuration=problem_configuration)  # type: ignore
        tct.run()

        if i == 0 and generate_pca_data:
            if constitutive_law == "plastic":
                with open(f"{DATA_FOLDER}/plastic_top_{version}_for_pca.npz", "wb") as f:
                    np.savez_compressed(f, tct.data_plastic_top)

            pca = PCA(n_components=1)
            pca.fit(tct.data_plastic_top)

        # with open(f"{DATA_FOLDER}/training_in_{version}_{frequency}.npz", "rb") as f:
        #     data = np.load(f)
        #     data_in = data[list(data.keys())[0]]

        # with open(f"{DATA_FOLDER}/training_out_{version}_{frequency}.npz", "rb") as f:
        #     data = np.load(f)
        #     data_out = data[list(data.keys())[0]]

        # if constitutive_law == "plastic":
        #     with open(f"{DATA_FOLDER}/plastic_top_reduced_{version}_{frequency}.npz", "rb") as f:
        #         data = np.load(f)
        #         data_plastic_top = data[list(data.keys())[0]]

        training_in.append(tct.data_in)
        training_out.append(tct.data_out)

        if constitutive_law == "plastic":
            # plastic_top.append(pca.transform(tct.data_plastic_top))  # type: ignore
            plastic_top.append(tct.data_plastic_top)

        # with open(f"{DATA_FOLDER}/training_in_{version}_{frequency}.npz", "wb") as f:
        #     np.savez_compressed(f, tct.data_in)

        # with open(f"{DATA_FOLDER}/training_out_{version}_{frequency}.npz", "wb") as f:
        #     np.savez_compressed(f, tct.data_out)

        # if constitutive_law == "plastic":
        #     with open(f"{DATA_FOLDER}/plastic_top_reduced_{version}_{frequency}.npz", "wb") as f:
        #         np.savez_compressed(f, pca.transform(tct.data_plastic_top))

        del tct

    training_in = np.array(training_in)
    interface_nodes = training_in.shape[2] // 2
    with open(f"{DATA_FOLDER}/training_in_{version}_x.npz", "wb") as f:
        np.savez_compressed(f, training_in[:, :, :interface_nodes])

    with open(f"{DATA_FOLDER}/training_in_{version}_v.npz", "wb") as f:
        np.savez_compressed(f, training_in[:, :, interface_nodes:])

    with open(f"{DATA_FOLDER}/training_out_{version}.npz", "wb") as f:
        np.savez_compressed(f, np.array(training_out))

    if constitutive_law == "plastic":
        with open(f"{DATA_FOLDER}/plastic_top_reduced_{version}.npz", "wb") as f:
            np.savez_compressed(f, plastic_top)


def load_npz_file(file_path: str):
    with open(file_path, "rb") as f:
        data = np.load(f)
        return data[list(data.keys())[0]]


def load_training_data(version: str):
    training_in_x = load_npz_file(f"{DATA_FOLDER}/training_in_{version}_x.npz")
    training_in_v = load_npz_file(f"{DATA_FOLDER}/training_in_{version}_v.npz")
    training_in = np.concatenate((training_in_x, training_in_v), axis=2)

    training_out = load_npz_file(f"{DATA_FOLDER}/training_out_{version}.npz")

    if constitutive_law == "plastic":
        plastic_top = load_npz_file(f"{DATA_FOLDER}/plastic_top_reduced_{version}.npz")

        return training_in, training_out, plastic_top

    return training_in, training_out, np.array([])


def train_lstm(data_version: str, predictor_version: str, resume_training: bool = False) -> None:

    if resume_training:
        reg = LSTMNetwork.load(f"{MODEL_FOLDER}/model_data_{data_version}_lstm_{predictor_version}.pkl")
        reg.set_hyperparameters()
    else:
        training_in, training_out, _ = load_training_data(data_version)

        reg = LSTMNetwork(training_in, training_out, save_path=f"{MODEL_FOLDER}/model_data_{data_version}_lstm_{predictor_version}.pkl")

    logging.info(f"Number of trainable parameters: {reg.get_trainable_params()}")
    start = time.time()
    reg.fit()
    end = time.time()
    logger.info("Finished training model")
    logger.info(f"Training time: {end - start} seconds")


def train_spils_net(data_version: str, predictor_version: str, resume_training: bool = False) -> None:

    if resume_training:
        reg = SPILSNet.load(f"{MODEL_FOLDER}/model_data_{data_version}_spils_net_{predictor_version}")
        reg.set_hyperparameters()
    else:
        training_in, training_out, internal_states = load_training_data(data_version)
        reg = SPILSNet(X=training_in, Y=training_out, internal_states=internal_states, save_path=f"{MODEL_FOLDER}/model_data_{data_version}_spils_net_{predictor_version}")

    logging.info(f"Number of trainable parameters: {reg.get_trainable_params()}")
    start = time.time()
    reg.fit()
    end = time.time()
    logger.info("Finished training model")
    logger.info(f"Training time: {end - start} seconds")


def apply_lstm(data_version: str, predictor_version: str, frequency: int = 1000) -> None:

    training_in, training_out, _ = load_training_data(data_version)

    predictor = LSTMNetwork(training_in, training_out)
    predictor.load_weights(f"{MODEL_FOLDER}/model_data_{data_version}_lstm_{predictor_version}.pkl_best_weights")

    evaluate_predictor_model(predictor, data_version)


def apply_spils_net(data_version: str, predictor_version: str, frequency: int = 1000) -> None:

    training_in, training_out, internal_states = load_training_data(data_version)

    predictor = SPILSNet(training_in, training_out, internal_states, save_path=f"{MODEL_FOLDER}/model_data_{data_version}_spils_net_{predictor_version}.pkl")
    predictor.load_weights(f"{MODEL_FOLDER}/model_data_{data_version}_spils_net_{predictor_version}_best_weights")

    evaluate_predictor_model(predictor, data_version)


def evaluate_predictor_model(predictor, data_version: str) -> None:
    predictor.calc_test_loss()
    predictor.initialize_memory_variables()

    tct_predict = applicator(predictor, frequency=frequency, constitutive_model=constitutive_law, configuration=problem_configuration)
    tct_error = extractor(frequency=frequency, constitutive_model=constitutive_law, configuration=problem_configuration)  # type: ignore

    # time_total = 4e-3
    # tct_predict.time_total = time_total
    # tct_error.time_total = time_total

    tct_predict.setup()
    tct_error.setup()

    tct_predict.export_results()
    tct_error.export_results()

    predict_dofs = tct_predict.get_dofs(tct_predict.bottom_half_nodes)
    error_dofs = tct_error.get_dofs(tct_error.bottom_half_nodes)

    errors = []
    truth_norms = []

    for tct_predict.step in progressbar(range(tct_predict.num_steps)):
        tct_error.step = tct_predict.step

        tct_predict.advance_time()
        tct_error.advance_time()

        tct_predict.solve_time_step()
        tct_error.solve_time_step()

        u_true = tct_error.u_next.x.array[error_dofs]
        u_pred = tct_predict.u_next.x.array[predict_dofs]

        norm_diff = np.linalg.norm(u_true - u_pred)
        norm_true = np.linalg.norm(u_true)

        step_error = norm_diff  # / (norm_true + 1e-10)

        truth_norms.append(norm_true)

        errors.append(step_error)

        # errors.append((np.square(tct_error.u_next.x.array[error_dofs] - tct_predict.u_next.x.array[predict_dofs])).mean(axis=None))  # type: ignore

    errors = np.array(errors)
    truth_norms = np.array(truth_norms)

    # hybrid_threshold = max(np.max(truth_norms) * 0.01, 1e-10)
    # final_error = errors / np.maximum(truth_norms, hybrid_threshold)

    final_error = errors / max(np.max(truth_norms), 1e-10)

    tct_predict.format_results()
    tct_error.format_results()

    logging.info(f"Global Error: {np.linalg.norm(errors) / np.linalg.norm(truth_norms):.4f}")

    with open(f"{RESULTS_FOLDER}/sim_results_error_{data_version}_{predictor_method}_{predictor_version}.npy", "wb") as f:
        np.save(f, final_error)

    with open(f"{RESULTS_FOLDER}/sim_results_in_{data_version}_{predictor_method}_{predictor_version}.npy", "wb") as f:
        np.save(f, tct_predict.data_in)

    with open(f"{RESULTS_FOLDER}/sim_results_out_{data_version}_{predictor_method}_{predictor_version}.npy", "wb") as f:
        np.save(f, tct_predict.data_out)

    with open(f"{RESULTS_FOLDER}/sim_results_internal_{data_version}_{predictor_method}_{predictor_version}.npy", "wb") as f:
        np.save(f, tct_predict.data_internal)


PREDICTOR_FUNCTIONS = {
    "lstm": (train_lstm, apply_lstm),
    "spils_net": (train_spils_net, apply_spils_net),
}


def run(data_version: str, predictor_version: str, frequency: int = 1000, predictor_method: str = "lstm", training_set_exists: bool = False, resume_training: bool = False, predictor_model_exists: bool = False) -> None:
    if not predictor_model_exists:
        if not training_set_exists:
            generate_training_set(data_version)
        PREDICTOR_FUNCTIONS[predictor_method][0](data_version, predictor_version, resume_training)

    PREDICTOR_FUNCTIONS[predictor_method][1](data_version, predictor_version, frequency)


if __name__ == "__main__":
    run(problem_configuration, predictor_version, frequency, predictor_method, training_set_exists, resume_training, predictor_model_exists)
