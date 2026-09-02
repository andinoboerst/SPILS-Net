import numpy as np

from fem_sim.tct_sims_3d import TCTSimulation3D
from spilsnet import SPILSNet


class TCTExtractDisplacements3D(TCTSimulation3D):
    """
    Extracts data for inverse prediction: Forces -> Displacements.
    
    Time synchronization:
      - Forces are calculated and saved BEFORE solving for the time step (data_in).
      - Time step is solved (solve_u).
      - Resulting displacements are saved AFTER solving the time step (data_out).
    """

    def _preprocess(self) -> None:
        super()._preprocess()

        # Input: Interface forces (Fx, Fy, Fz per node)
        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        # Output: Interface displacements, velocities, accelerations
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs) * 3))

        if self.constitutive_model == "plastic":
            self.data_plastic_top = np.zeros((self.num_steps + 1, len(self.top_half_cells)))
            self.data_plastic_top[0, :] = self.eq_epsilon_p_next.x.array[self.top_half_cells]

        self.eps_data = []
        self.sig_data = []

    def _solve_time_step(self):
        # 1. Save force BEFORE solving the time step
        self.data_in[self.step, :] = self.calculate_interface_tractions()

        # 2. Solve for the time step
        self.solve_u()

        # 3. Save internal state and displacements AFTER solving
        if self.constitutive_model == "plastic":
            self.data_plastic_top[self.step + 1, :] = self.eq_epsilon_p_next.x.array[self.top_half_cells]

        self.data_out[self.step, :] = np.concatenate([
            self.u_next.x.array[self.interface_dofs],
            self.v_next.x.array[self.interface_dofs],
            self.a_next.x.array[self.interface_dofs]
        ])

        self.eps_data.append(self.eps.x.array.copy())
        self.sig_data.append(self.sig.x.array.copy())


class TCTApplyDisplacements3D(TCTSimulation3D):
    """
    Applies predicted displacements to the bottom-half reduced domain
    based on incoming interface forces.
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

        self.data_in = np.zeros((self.num_steps, len(self.interface_dofs)))
        self.data_out = np.zeros((self.num_steps, len(self.interface_dofs) * 3))
        self.data_internal = np.zeros(self.num_steps)

        self.dirichlet_interface_marker = 88
        self.add_dirichlet_bc(self.interface_nodes_local, self.dirichlet_interface_marker)

    def _solve_time_step(self) -> None:
        # 1. Obtain interface force before solving
        self.data_in[self.step, :] = self.calculate_interface_tractions()

        if self.save_internal_state and hasattr(self.predictor, "internal_scaler") and hasattr(self.predictor, "hidden_state") and self.predictor.hidden_state is not None:
            self.data_internal[self.step] = self.predictor.internal_scaler.inverse_transform(self.predictor.hidden_state.detach().cpu().numpy())

        # 2. Predict displacement based on force
        prediction = self.predictor.predict(self.data_in[self.step, :])
        self.data_out[self.step, :len(self.interface_dofs)] = prediction

        # 3. Update boundary conditions with predicted displacement
        self.update_dirichlet_bc(prediction, self.dirichlet_interface_marker)

        # 4. Solve time step
        self.solve_u()
