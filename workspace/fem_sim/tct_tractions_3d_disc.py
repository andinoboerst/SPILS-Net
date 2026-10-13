import numpy as np

from fem_sim.tct_sims_3d_disc import TCTSimulation3DDisc
from spilsnet import SPILSNet


class TCTExtractTractions3DDisc(TCTSimulation3DDisc):
    """
    Extracts data for discontinuous 3D forward prediction: Displacements -> Tractions.
    """

    def _preprocess(self) -> None:
        super()._preprocess()

        # Input: Interface displacements, velocities, accelerations [u | v | a]
        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs) * 3))
        # Output: Interface forces / tractions
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))

        if self.constitutive_model == "plastic":
            self.data_plastic_top = np.zeros((self.num_steps + 1, len(self.top_half_cells)))
            self.data_plastic_top[0, :] = self.eq_epsilon_p_next.x.array[self.top_half_cells]

        self.eps_data = []
        self.sig_data = []

    def _solve_time_step(self):
        self.data_in[self.step, :] = np.concatenate([
            self.u_next.x.array[self.interface_dofs],
            self.v_next.x.array[self.interface_dofs],
            self.a_next.x.array[self.interface_dofs],
        ])

        self.solve_u()

        if self.constitutive_model == "plastic":
            self.data_plastic_top[self.step + 1, :] = self.eq_epsilon_p_next.x.array[self.top_half_cells]

        self.data_out[self.step, :] = self.calculate_interface_tractions()

        self.eps_data.append(self.eps.x.array.copy())
        self.sig_data.append(self.sig.x.array.copy())


class TCTApplyTractions3DDisc(TCTSimulation3DDisc):
    """
    Applies predicted tractions as a Neumann BC to the bottom-half reduced domain
    for the discontinuous 3D interface based on incoming displacements.
    """

    height = 50.0
    use_top_half = False

    def __init__(self, predictor, *args, **kwargs) -> None:
        self.predictor = predictor
        if isinstance(self.predictor, SPILSNet) or hasattr(self.predictor, "internal_scaler"):
            self.save_internal_state = True
        else:
            self.save_internal_state = False
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs) * 3))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_internal = np.zeros(self.num_steps)

        self.neumann_interface_marker = 88

        self.add_neumann_bc(self.interface_nodes_local, self.neumann_interface_marker, facets=self.interface_facets)

    def _solve_time_step(self) -> None:
        self.data_in[self.step, :] = np.concatenate([
            self.u_next.x.array[self.interface_dofs],
            self.v_next.x.array[self.interface_dofs],
            self.a_next.x.array[self.interface_dofs],
        ])

        if (
            self.save_internal_state
            and hasattr(self.predictor, "internal_scaler")
            and hasattr(self.predictor, "hidden_state")
            and self.predictor.hidden_state is not None
        ):
            self.data_internal[self.step] = self.predictor.internal_scaler.inverse_transform(
                self.predictor.hidden_state.detach().cpu().numpy()
            )

        if isinstance(self.predictor, SPILSNet):
            prediction = self.predictor.predict(self.u_next.x.array[self.interface_dofs])
        else:
            prediction = self.predictor.predict(
                self.u_next.x.array[self.interface_dofs], self.v_next.x.array[self.interface_dofs]
            )

        self.data_out[self.step, :] = prediction

        self.update_neumann_bc(prediction, self.neumann_interface_marker)

        self.solve_u()
