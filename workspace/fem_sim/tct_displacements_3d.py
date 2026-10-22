import numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, Function
from dolfinx.mesh import meshtags
from ufl import Measure, TestFunction, dot

from fem_sim.interface_rows import InterfaceRowSampler
from fem_sim.progress_bar import progressbar
from fem_sim.time_grid import GridResampler
from fem_sim.tct_sims_3d import TCTSimulation3D
from nn_predictors.misc import stack_node_features
from spilsnet import SPILSNet


class TCTExtractDisplacements3D(TCTSimulation3D):
    """
    Extracts data for inverse prediction: Forces -> Displacements.
    
    Time synchronization:
      - Forces are calculated and saved BEFORE solving for the time step (data_in).
      - Time step is solved (solve_u).
      - Resulting displacements are saved AFTER solving the time step (data_out).
    """

    # Keep a copy of the strain and stress fields of every step (eps_data, sig_data); about 1.2 MB per step
    # on the 3D mesh, so switch it off for long (explicit) runs that do not need them
    store_field_history = True

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

        if self.store_field_history:
            self.eps_data.append(self.eps.x.array.copy())
            self.sig_data.append(self.sig.x.array.copy())


class TCTReplayDisplacements3D(TCTSimulation3D):
    """
    Bottom-half reduced domain with the interface displacement prescribed from outside.

    Set `next_iface` before each solve_time_step(). The full solver state can be saved and
    restored to probe the response to different interface displacements from the same state.
    """

    height = 50.0
    use_top_half = False
    # Interface coupling: None prescribes next_iface exactly (Dirichlet). A stiffness k [N/mm^3] instead ties
    # the interface to next_iface through distributed springs (Robin): the traction on the reduced domain is
    # k * (next_iface - u). The interface then follows the reduced domain's own stiffness as well, and
    # next_iface = u + t / k reproduces a traction t exactly.
    robin_stiffness = None

    def _preprocess(self) -> None:
        super()._preprocess()
        self.iface_marker = 88
        if self.robin_stiffness is None:
            self.add_dirichlet_bc(self.interface_nodes, self.iface_marker)
        else:
            self.robin_target = Function(self.V)
            self.robin_k = Constant(self.mesh, default_scalar_type(self.robin_stiffness))
            fdim = self.mesh.topology.dim - 1
            facets = np.sort(self.interface_facets)
            facet_tag = meshtags(self.mesh, fdim, facets, np.full_like(facets, self.iface_marker))
            self.ds_iface = Measure("ds", domain=self.mesh, subdomain_data=facet_tag)(self.iface_marker)
        self.next_iface = np.zeros(len(self.interface_dofs))

    def get_problem_equations(self, u_next, u_k, v_k, a_k, f, sigma_func, sigma_kwargs) -> tuple:
        u_next, residual, bcs = super().get_problem_equations(u_next, u_k, v_k, a_k, f, sigma_func, sigma_kwargs)
        # Only the main problem gets the springs, not the traction-recovery problem on the same mesh
        if self.robin_stiffness is not None and u_next is self.u_next:
            residual += self.robin_k * dot(u_next - self.robin_target, TestFunction(u_next.function_space)) * self.ds_iface
        return u_next, residual, bcs

    def _solve_time_step(self) -> None:
        if self.robin_stiffness is None:
            self.update_dirichlet_bc(self.next_iface, self.iface_marker)
        else:
            self.robin_target.x.array[self.interface_dofs] = self.next_iface
        self.solve_u()

    def _state_functions(self) -> list:
        return [
            self.u_next, self.v_next, self.a_next, self.u_k, self.v_k, self.a_k,
            self.eq_epsilon_p_next, self.eq_epsilon_p_k, self.sigma_next, self.sigma_k, self.eps, self.sig,
        ]

    def snapshot(self) -> list:
        return [f.x.array.copy() for f in self._state_functions()]

    def restore(self, state: list) -> None:
        for f, arr in zip(self._state_functions(), state):
            f.x.array[:] = arr

    def force_response(self, state: list, u_iface: np.ndarray) -> np.ndarray:
        """Force f^{n+1} after solving the current step from `state` with interface displacement u_iface."""
        self.restore(state)
        self.next_iface = u_iface
        self.solve_time_step()
        f = self.calculate_interface_tractions()
        self.restore(state)
        return f


class TCTApplyRowDisplacements3D(TCTReplayDisplacements3D):
    """
    Reduced domain driven by a surrogate that predicts the interface displacement from the displacement
    and velocity fields sampled at fixed depths below each interface node:

        [u, v at depths d_1..d_k below node i]^n  ->  u_iface^{n+1}               (absolute_target)
                                                   ->  u_iface^{n+1} - u_iface^n   (increment)

    No force is computed or used, and nothing filters or relaxes the prediction.

    With sm_dt set, the surrogate runs on its own time grid T_k = k * sm_dt (its training step) while the
    simulation takes steps of any, possibly varying, size dt: at each T_k the rows are extrapolated from the
    last state with the row velocities, the surrogate predicts the interface at T_{k+1}, and the interface is
    interpolated linearly in time between the predictions for T_k and T_{k+1}.
    """

    def __init__(
        self, predictor, *args, depths=(2.5, 5.0, 7.5, 10.0), velocity_at_interface: bool = True, include_velocity: bool = True, absolute_target: bool = True, output_projection=None, input_builder=None, robin_stiffness=None, sm_dt=None, **kwargs
    ) -> None:
        self.predictor = predictor
        self.sm_dt = sm_dt
        # Spring (Robin) coupling instead of prescribing the prediction; see TCTReplayDisplacements3D
        self.robin_stiffness = robin_stiffness
        self.depths = depths
        self.absolute_target = absolute_target
        # Optional projector (matrix on the flat interface vector) applied to every prediction
        self.output_projection = output_projection
        # Optional callable (u_rows, v_rows) -> (features, anchor) replacing the default row features; the
        # prediction is then added to the anchor (e.g. the displacement of the first row)
        self.input_builder = input_builder
        self._anchor = 0.0
        self.velocity_at_interface = velocity_at_interface
        self.include_velocity = include_velocity
        self.save_internal_state = hasattr(self.predictor, "internal_scaler")
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()
        self.sampler = InterfaceRowSampler(self.V, self.interface_nodes, self.depths)
        self.current_iface = np.zeros(len(self.interface_dofs))

        n = len(self.interface_dofs)
        if self.input_builder is not None:
            n_in = self.input_builder(np.zeros((len(self.depths), self.sampler.n_nodes, 3)), np.zeros((len(self.depths), self.sampler.n_nodes, 3)))[0].size
        else:
            n_in = self.sampler.n_nodes * self.sampler.n_features(self.velocity_at_interface, self.include_velocity)
        self.data_in = np.zeros((self.num_steps, n_in))
        self.data_out = np.zeros((self.num_steps, n))
        self.data_internal = np.zeros(self.num_steps)

    def _store_internal(self, value: np.ndarray) -> None:
        value = np.ravel(value)
        if self.data_internal.ndim == 1 and value.size > 1:
            self.data_internal = np.zeros((len(self.data_internal), value.size))
        self.data_internal[self.step] = value if value.size > 1 else value[0]

    def row_features(self, u=None) -> np.ndarray:
        u = self.u_next.x.array if u is None else u
        if self.input_builder is not None:
            x, self._anchor = self.input_builder(self.sampler.sample(u), self.sampler.sample(self.v_next.x.array))
            return x
        return self.sampler.features(u, self.v_next.x.array, self.velocity_at_interface, self.include_velocity)

    def _surrogate_step(self, x) -> np.ndarray:
        """One surrogate call: rows -> interface at the next surrogate time."""
        prediction = self.predictor.predict(x)
        if self.output_projection is not None:
            prediction = self.output_projection @ prediction
        prediction = self._anchor + prediction
        return prediction if self.absolute_target else self._iface_to + prediction

    def _solve_subcycled(self) -> None:
        if not hasattr(self, "_t_from"):
            self._t_from = self._t_to = 0.0
            self._iface_from = self._iface_to = self.u_next.x.array[self.interface_dofs].copy()
        t_old = self.time - self.dt  # the state in u_next, v_next belongs to t_old
        while self._t_to < self.time - 1e-9 * self.sm_dt:
            # New surrogate step at T = _t_to, with the rows extrapolated from t_old to T
            T = self._t_to
            x = self.row_features(self.u_next.x.array + (T - t_old) * self.v_next.x.array)
            self._last_x = x
            if self.save_internal_state and getattr(self.predictor, "hidden_state", None) is not None:
                self._last_internal = self.predictor.internal_scaler.inverse_transform(
                    self.predictor.hidden_state.detach().cpu().numpy()
                ).ravel()
            prediction = self._surrogate_step(x)
            self._t_from, self._iface_from = T, self._iface_to
            self._t_to, self._iface_to = T + self.sm_dt, prediction
        # Every step records the inputs and internal state of the latest surrogate call (held between calls)
        self.data_in[self.step] = self._last_x
        if hasattr(self, "_last_internal"):
            self._store_internal(self._last_internal)
        w = (self.time - self._t_from) / self.sm_dt
        self.current_iface = (1.0 - w) * self._iface_from + w * self._iface_to
        self.data_out[self.step] = self.current_iface
        self.next_iface = self.current_iface
        TCTReplayDisplacements3D._solve_time_step(self)

    def _solve_time_step(self) -> None:
        if self.sm_dt is not None:
            self._solve_subcycled()
            return
        x = self.row_features()
        self.data_in[self.step] = x

        if self.save_internal_state and getattr(self.predictor, "hidden_state", None) is not None:
            self._store_internal(self.predictor.internal_scaler.inverse_transform(
                self.predictor.hidden_state.detach().cpu().numpy()
            ))

        prediction = self.predictor.predict(x)
        if self.output_projection is not None:
            prediction = self.output_projection @ prediction
        prediction = self._anchor + prediction
        self.current_iface = prediction if self.absolute_target else self.current_iface + prediction
        self.data_out[self.step] = self.current_iface
        self.next_iface = self.current_iface
        super()._solve_time_step()


class TCTApplyDisplacements3D(TCTSimulation3D):
    """
    Applies predicted displacements to the bottom-half reduced domain
    based on incoming interface forces.
    """

    height = 50.0
    use_top_half = False

    def __init__(
        self,
        predictor,
        *args,
        warmup_steps: int = 60,
        stabilization_steps: int = 0,
        ramp_steps: int = 40,
        force_filter_beta: float = 0.08,
        disp_relaxation_alpha: float = 0.3,
        clip_forces: bool = True,
        clip_displacements: bool = True,
        force_clip_mode: str = "hard",
        force_soft_margin: float = 0.25,
        notch_filter: bool = True,
        increment_mode: bool = False,
        **kwargs,
    ) -> None:
        self.predictor = predictor
        self.warmup_steps = warmup_steps
        self.stabilization_steps = stabilization_steps
        self.ramp_steps = ramp_steps
        # force_filter_beta = 1.0 disables the EMA low-pass (no added phase lag)
        self.force_filter_beta = force_filter_beta
        self.disp_relaxation_alpha = disp_relaxation_alpha
        self.clip_forces = clip_forces
        self.clip_displacements = clip_displacements
        # "hard": np.clip to training range; "soft": identity inside the range and a
        # tanh saturation of width force_soft_margin * half-range outside it
        if force_clip_mode not in ("hard", "soft"):
            raise ValueError(f"Unknown force_clip_mode '{force_clip_mode}'")
        self.force_clip_mode = force_clip_mode
        self.force_soft_margin = force_soft_margin
        self.notch_filter = notch_filter
        # Incremental surrogate: input [f^n, u^n, u^n - u^{n-1}] per node, output u^{n+1} - u^n
        self.increment_mode = increment_mode

        self.true_interface_disp = None
        self.current_applied_disp = None
        self.prev_applied_disp = None
        self.prev_force = None
        self.filtered_force = None
        self.f_min = None
        self.f_max = None
        self.u_min = None
        self.u_max = None

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
        self.current_applied_disp = np.zeros(len(self.interface_dofs))
        self.prev_applied_disp = np.zeros(len(self.interface_dofs))
        self.filtered_force = np.zeros(len(self.interface_dofs))
        self.prev_force = None

        # Determine force scaler limits for clamping if available
        if hasattr(self.predictor, "input_scaler") and hasattr(self.predictor.input_scaler, "data_min_"):
            self.f_min = self.predictor.input_scaler.data_min_
            self.f_max = self.predictor.input_scaler.data_max_
            if self.increment_mode:
                # Keep only the force features of the node-major stacked input
                n = len(self.interface_dofs)
                is_force = stack_node_features(np.ones(n), np.zeros(n), np.zeros(n)).astype(bool)
                self.f_min, self.f_max = self.f_min[is_force], self.f_max[is_force]
        else:
            self.f_min, self.f_max = None, None

        # Determine displacement scaler limits with margin for safety clamping
        # (in increment mode these bound the increment u^{n+1} - u^n)
        if hasattr(self.predictor, "output_scaler") and hasattr(self.predictor.output_scaler, "data_min_"):
            margin = 0.25
            u_range = self.predictor.output_scaler.data_max_ - self.predictor.output_scaler.data_min_
            self.u_min = self.predictor.output_scaler.data_min_ - margin * u_range
            self.u_max = self.predictor.output_scaler.data_max_ + margin * u_range
        else:
            self.u_min, self.u_max = None, None

        self.dirichlet_interface_marker = 88
        self.add_dirichlet_bc(self.interface_nodes, self.dirichlet_interface_marker)

    def _soft_saturate(self, f: np.ndarray, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
        """Identity inside [f_min, f_max]; smooth tanh saturation (slope 1 at the bounds)
        that asymptotes to the bounds +/- force_soft_margin * half-range outside."""
        m = np.maximum(self.force_soft_margin * 0.5 * (f_max - f_min), 1e-12)
        out = f.copy()
        hi = f > f_max
        lo = f < f_min
        out[hi] = f_max[hi] + m[hi] * np.tanh((f[hi] - f_max[hi]) / m[hi])
        out[lo] = f_min[lo] - m[lo] * np.tanh((f_min[lo] - f[lo]) / m[lo])
        return out

    def _solve_time_step(self) -> None:
        # 1. Obtain interface force before solving
        f_raw = self.calculate_interface_tractions()
        self.data_in[self.step, :] = f_raw

        # 2. Force pre-clamping to training distribution limits
        if self.clip_forces and self.f_min is not None and self.f_max is not None:
            if self.force_clip_mode == "soft":
                f_clamped = self._soft_saturate(f_raw, self.f_min, self.f_max)
            else:
                f_clamped = np.clip(f_raw, self.f_min, self.f_max)
        else:
            f_clamped = f_raw

        # 3. Nyquist notch filter (2-step moving average) to cancel out (-1)^k flip-flop modes
        if self.prev_force is None or not self.notch_filter:
            f_notched = f_clamped.copy()
        else:
            f_notched = 0.5 * (f_clamped + self.prev_force)
        self.prev_force = f_clamped.copy()

        # 4. EMA low-pass filtering on top of notched force to suppress residual high frequencies
        if self.step == 0:
            self.filtered_force = f_notched.copy()
        else:
            self.filtered_force = (1.0 - self.force_filter_beta) * self.filtered_force + self.force_filter_beta * f_notched

        # 5. Always advance predictor state on incoming force to keep hidden state synchronized
        if (
            self.save_internal_state
            and hasattr(self.predictor, "internal_scaler")
            and hasattr(self.predictor, "hidden_state")
            and self.predictor.hidden_state is not None
        ):
            self.data_internal[self.step] = self.predictor.internal_scaler.inverse_transform(
                self.predictor.hidden_state.detach().cpu().numpy()
            )

        if self.increment_mode:
            u_n = self.current_applied_disp
            model_in = stack_node_features(self.filtered_force, u_n, u_n - self.prev_applied_disp)
            delta = self.predictor.predict(model_in)
            if self.clip_displacements and self.u_min is not None and self.u_max is not None:
                delta = np.clip(delta, self.u_min, self.u_max)
            raw_pred = u_n + delta
        else:
            raw_pred = self.predictor.predict(self.filtered_force)

        # 6. Applied displacement determination:
        # Case A: Ground-truth warm-up (allows the structure to move realistically instead of being clamped)
        if self.warmup_steps > 0 and self.step < self.warmup_steps and self.true_interface_disp is not None:
            actual_disp = self.true_interface_disp.copy()
        # Case B: Smooth handoff from ground truth to surrogate prediction
        elif (
            self.ramp_steps > 0
            and self.warmup_steps > 0
            and self.step < self.warmup_steps + self.ramp_steps
            and self.true_interface_disp is not None
        ):
            w = (self.step - self.warmup_steps + 1) / float(self.ramp_steps)
            target_disp = (1.0 - w) * self.true_interface_disp + w * raw_pred
            if self.clip_displacements and not self.increment_mode and self.u_min is not None and self.u_max is not None:
                target_disp = np.clip(target_disp, self.u_min, self.u_max)
            actual_disp = (1.0 - self.disp_relaxation_alpha) * self.current_applied_disp + self.disp_relaxation_alpha * target_disp
        # Case C: Zero hold stabilization fallback (if no ground truth is provided)
        elif self.step < self.stabilization_steps:
            actual_disp = np.zeros(len(self.interface_dofs))
        # Case D: Autonomous surrogate model driving
        else:
            target_disp = raw_pred
            if self.clip_displacements and not self.increment_mode and self.u_min is not None and self.u_max is not None:
                target_disp = np.clip(target_disp, self.u_min, self.u_max)
            actual_disp = (1.0 - self.disp_relaxation_alpha) * self.current_applied_disp + self.disp_relaxation_alpha * target_disp

        self.prev_applied_disp = self.current_applied_disp
        self.current_applied_disp = actual_disp.copy()
        self.data_out[self.step, : len(self.interface_dofs)] = actual_disp

        # 7. Update boundary conditions with applied displacement
        self.update_dirichlet_bc(actual_disp, self.dirichlet_interface_marker)

        # 8. Solve time step
        self.solve_u()


class TCTExtractRowsExplicit3D(TCTSimulation3D):
    """
    Full domain, explicit integration with a given (possibly varying) step sequence. Streams onto the fixed
    surrogate grid T_k = k * dt_grid (GridResampler):

      u_rows, v_rows  at the row depths below the interface, extrapolated from the last state as in the
                      coupled run (inputs; v_rows is the velocity of that state)
      iface           interface displacement, interpolated (targets)
      plastic_top     equivalent plastic strain of the top half, interpolated (internal state)
      u_top           displacement field of the top half, interpolated (internal state)

    The rows are sampled in the full domain directly; below the interface they equal the rows of a replay
    of the retained bottom half.
    """

    time_integration = "explicit"

    def __init__(self, *args, depths=(0.0, 2.5, 5.0, 7.5, 10.0), dt_grid: float = 5e-7, **kwargs) -> None:
        self.depths = depths
        self.dt_grid = dt_grid
        super().__init__(*args, **kwargs)

    def _preprocess(self) -> None:
        super()._preprocess()
        self.sampler = InterfaceRowSampler(self.V, self.interface_nodes, self.depths)
        self.top_half_dofs = self.get_dofs(self.top_half_nodes)

    def _plastic_top(self) -> np.ndarray:
        return self.eq_epsilon_p_next.x.array[self.top_half_cells].copy()

    def _state(self) -> dict:
        return {
            "u_rows": self.sampler.sample(self.u_next.x.array),
            "v_rows": self.sampler.sample(self.v_next.x.array),
            "iface": self.u_next.x.array[self.interface_dofs].copy(),
            "plastic_top": self._plastic_top(),
            "u_top": self.u_next.x.array[self.top_half_dofs].copy(),
        }

    def generate(self, dt_sequence: np.ndarray) -> dict:
        """Run with the given steps (they should sum to at least time_total) and return the grid data."""
        self.dt = float(np.min(dt_sequence))
        self.setup()
        self.num_steps = len(dt_sequence)
        n_grid = int(round(self.time_total / self.dt_grid)) + 1
        resampler = GridResampler(self.dt_grid, n_grid)
        old = self._state()
        for self.step in progressbar(range(self.num_steps)):
            self.dt = float(dt_sequence[self.step])
            t_old = self.time
            self.advance_time()
            self.solve_time_step()
            new = self._state()
            resampler.step(
                t_old, self.time,
                interp={"iface": (old["iface"], new["iface"]), "plastic_top": (old["plastic_top"], new["plastic_top"]),
                        "u_top": (old["u_top"], new["u_top"])},
                extrap={"u_rows": (old["u_rows"], old["v_rows"]), "v_rows": (old["v_rows"], 0.0)},
            )
            old = new
            if resampler.complete:
                break
        resampler.finish(self.time, old)
        if not resampler.complete:
            raise RuntimeError(f"Steps end at t = {self.time:.4g} before the grid end {self.time_total:.4g}")
        return resampler.data
