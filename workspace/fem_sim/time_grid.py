import numpy as np


class GridResampler:
    """
    Streams the states of a simulation with arbitrary, possibly varying, time steps onto the fixed grid
    T_k = k * dt_grid, k = 0..n_points-1, without storing the simulation steps themselves.

    Call step() once per simulation step with the times of the old and new state. Every grid time T in
    [t_old, t_new) is filled from that step, with the same rules the coupled simulation uses
    (TCTApplyRowDisplacements3D with sm_dt):

      interp:  linear interpolation between the old and new state (targets, internal state)
      extrap:  value + (T - t_old) * rate from the old state only (surrogate inputs, which in the coupled
               run must be formed before the step to T is taken)
    """

    def __init__(self, dt_grid: float, n_points: int) -> None:
        self.dt_grid = dt_grid
        self.n_points = n_points
        self.k = 0
        self.data = {}

    @property
    def next_time(self) -> float:
        return self.k * self.dt_grid

    @property
    def complete(self) -> bool:
        return self.k >= self.n_points

    def _store(self, name: str, value: np.ndarray) -> None:
        if name not in self.data:
            self.data[name] = np.zeros((self.n_points, *np.shape(value)))
        self.data[name][self.k] = value

    def step(self, t_old: float, t_new: float, interp: dict | None = None, extrap: dict | None = None) -> None:
        """
        interp: {name: (value_old, value_new)}; extrap: {name: (value_old, rate_old)}. Values are only
        evaluated if a grid time falls into the step, so callables returning the pair are accepted too.
        """
        tol = 1e-9 * self.dt_grid
        while not self.complete and self.next_time < t_new - tol:
            T = self.next_time
            w = (T - t_old) / (t_new - t_old)
            for name, pair in (interp or {}).items():
                old, new = pair() if callable(pair) else pair
                self._store(name, (1.0 - w) * old + w * new)
            for name, pair in (extrap or {}).items():
                value, rate = pair() if callable(pair) else pair
                self._store(name, value + (T - t_old) * rate)
            self.k += 1

    def finish(self, t_last: float, values: dict) -> None:
        """Fill grid times up to t_last (inclusive, e.g. T = time_total) from the final state."""
        tol = 1e-9 * self.dt_grid
        while not self.complete and self.next_time <= t_last + tol:
            for name, value in values.items():
                self._store(name, value)
            self.k += 1
