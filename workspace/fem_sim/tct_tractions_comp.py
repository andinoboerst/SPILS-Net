import numpy as np
import logging

from workspace.fem_sim.tct_tractions import TCTExtractTractions as extractor
from workspace.fem_sim.tct_sims import TCTSimulation
from workspace.fem_sim.progress_bar import progressbar


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RESULTS_FOLDER = "workspace/results"


class TCTApplyFixedTractions(TCTSimulation):

    height = 25.0

    def __init__(self, tractions, *args, **kwargs) -> None:
        self.tractions = tractions
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs) * 3))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_internal = np.zeros(self.num_steps)

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_nodes_local, self.neumann_interface_marker, facets=self.interface_facets)

    def _solve_time_step(self) -> None:
        self.data_in[self.step, :] = np.concatenate([self.u_next.x.array[self.interface_dofs], self.v_next.x.array[self.interface_dofs], self.a_next.x.array[self.interface_dofs]])

        self.data_out[self.step, :] = self.tractions[self.step]

        self.update_neumann_bc(self.tractions[self.step], self.neumann_interface_marker)

        self.solve_u()


def evaluate_fem_model() -> None:
    frequency = 750
    problem_configuration = "scaled"

    with open(f"{RESULTS_FOLDER}/tct_results_out_{problem_configuration}_freq_{frequency}.npy", "rb") as f:
        tractions = np.load(f)

    tct_predict = TCTApplyFixedTractions(frequency=750, constitutive_model="plastic", configuration=problem_configuration, tractions=tractions)
    tct_error = extractor(frequency=frequency, constitutive_model="plastic", configuration=problem_configuration)  # type: ignore

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

        step_error = norm_diff / (norm_true + 1e-10)

        errors.append(step_error)

    tct_predict.format_results()
    tct_error.format_results()

    logging.info(f"Error: {np.mean(errors)}")

    with open(f"{RESULTS_FOLDER}/sim_results_error_{problem_configuration}_fixed_fem.npy", "wb") as f:
        np.save(f, np.array(errors))

    with open(f"{RESULTS_FOLDER}/sim_results_in_{problem_configuration}_fixed_fem.npy", "wb") as f:
        np.save(f, tct_predict.data_in)

    with open(f"{RESULTS_FOLDER}/sim_results_out_{problem_configuration}_fixed_fem.npy", "wb") as f:
        np.save(f, tct_predict.data_out)

    with open(f"{RESULTS_FOLDER}/sim_results_internal_{problem_configuration}_fixed_fem.npy", "wb") as f:
        np.save(f, tct_predict.data_internal)


def compare_force_application() -> None:
    tct = TCTExtractTractions(frequency=1000, constitutive_model=DEFORMATION)
    # tct.time_total = 5e-4
    tct.run()

    # tct.postprocess("u", "u", "y", name="tractions_full")

    tct_apply = TCTApplyFixedTractions(tct.data_out, frequency=1000, constitutive_model=DEFORMATION)
    # tct_apply.time_total = 5e-4
    tct_apply.run()

    tct_apply.postprocess("u", "u", "y", name="tractions_applied")

    u_k_app_error = np.zeros(tct.formatted_plot_results["u"].shape)
    u_k_app_error[:, tct.bottom_half_nodes, :] = tct.formatted_plot_results["u"][:, tct.bottom_half_nodes, :] - tct_apply.formatted_plot_results["u"][:, tct_apply.bottom_half_nodes, :]
    tct.postprocess(u_k_app_error, "u", "norm", name="tractions_applied_error")

    v_k_app_error = np.zeros(tct.formatted_plot_results["u"].shape)
    v_k_app_error[:, tct.bottom_half_nodes, :] = tct.formatted_plot_results["v"][:, tct.bottom_half_nodes, :] - tct_apply.formatted_plot_results["v"][:, tct_apply.bottom_half_nodes, :]
    tct.postprocess(v_k_app_error, "u", "norm", name="tractions_applied_error_velocity")


if __name__ == "__main__":
    evaluate_fem_model()
