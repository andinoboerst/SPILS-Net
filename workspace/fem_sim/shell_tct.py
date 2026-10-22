"""
Square-tube shell counterpart of the 3D TCT problem: explicit RM shell (fem_sim.shell_explicit), bottom ring
driven axially with amplitude * sin(omega t) (rotations fixed), top ring clamped, interface ring at
mid-height. The reduced model is the bottom half with the interface ring's displacements prescribed from
outside; its rotations stay free (displacements only, as in the crash setup).
"""

import numpy as np

from fem_sim.progress_bar import progressbar
from fem_sim.shell_explicit import ExplicitShell, square_tube_mesh
from fem_sim.time_grid import GridResampler


class ShellTubeTCT:
    side = 50.0
    height = 100.0
    h = 2.5
    amplitude = 0.5  # mm at the bottom ring

    def __init__(self, frequency: float = 1000, reduced: bool = False, plastic: bool = True, **shell_parameters) -> None:
        self.omega = 2 * np.pi * frequency
        self.frequency = frequency
        self.reduced = reduced
        z_top = self.height / 2 if reduced else self.height
        self.points, cells, self.ring_s = square_tube_mesh(self.side, z_top, self.h)
        self.shell = ExplicitShell(normal_hint=lambda c: np.stack([c[:, 0], c[:, 1], 0 * c[:, 0]], 1))
        self.shell.plastic = plastic
        for key, value in shell_parameters.items():
            setattr(self.shell, key, value)
        self.shell.setup(self.points, cells)
        sh = self.shell

        z = self.points[:, 2]
        self.bottom_nodes = self._ring(0.0)
        self.interface_nodes = self._ring(self.height / 2)
        sh.prescribe_u(sh.node_dofs(self.bottom_nodes))
        sh.prescribe_theta(sh.node_dofs(self.bottom_nodes))
        if reduced:
            sh.prescribe_u(sh.node_dofs(self.interface_nodes))
        else:
            top = self._ring(self.height)
            sh.prescribe_u(sh.node_dofs(top))
            sh.prescribe_theta(sh.node_dofs(top))
        self.interface_dofs = sh.node_dofs(self.interface_nodes)
        self.top_half_cells = np.where(sh.centroids[:, 2] > self.height / 2)[0]
        self._bottom_z_dofs = sh.node_dofs(self.bottom_nodes)[2::3]
        self.next_iface = np.zeros(len(self.interface_dofs))

    def _ring(self, z_value: float) -> np.ndarray:
        """Nodes of the ring at height z_value, ordered along the perimeter."""
        nodes = np.where(np.abs(self.points[:, 2] - z_value) < 1e-9)[0]
        return nodes[np.argsort(self.ring_s[nodes])]

    def ring_nodes(self, depths) -> list:
        """Node rings at the given depths below the interface (multiples of h), perimeter-ordered."""
        return [self._ring(self.height / 2 - d) for d in depths]

    def interface_topology(self) -> tuple:
        """Interface ring coordinates and its edges (closed ring) as segments [n, 2]."""
        n = len(self.interface_nodes)
        return self.points[self.interface_nodes], np.stack([np.arange(n), (np.arange(n) + 1) % n], 1)

    def step(self, dt: float) -> None:
        sh = self.shell
        u_p = np.zeros(3 * sh.n_nodes)
        u_p[self._bottom_z_dofs] = self.amplitude * np.sin(self.omega * (sh.time + dt))
        if self.reduced:
            u_p[self.interface_dofs] = self.next_iface
        sh.step(dt, u_prescribed=u_p)

    @property
    def time(self) -> float:
        return self.shell.time

    def u(self) -> np.ndarray:
        return self.shell.u.x.array

    def v(self) -> np.ndarray:
        return self.shell.v_u

    def plastic_top(self) -> np.ndarray:
        """Equivalent plastic strain of the top-half elements (all thickness points), flat."""
        return self.shell.eq_eps_p[self.top_half_cells].ravel().copy()

    def generate(self, dt_sequence: np.ndarray, dt_grid: float, depths, time_total: float, iface_on_grid=None) -> dict:
        """
        Run with the given steps, streaming onto the grid T_k = k dt_grid (GridResampler): row displacements
        and velocities (extrapolated from the last state, as in the coupled run), interface displacement and,
        for the full model, top-half plastic strain (interpolated). iface_on_grid [n_grid, n_iface]: interface
        values on the grid for the reduced model, interpolated linearly in time.
        """
        rings = self.ring_nodes(depths)
        sh = self.shell

        def state():
            u, v = sh.u.x.array, sh.v_u
            out = {
                "u_rows": np.stack([sh.nodal(u, r) for r in rings]),
                "v_rows": np.stack([sh.nodal(v, r) for r in rings]),
                "iface": u[self.interface_dofs].copy(),
            }
            if not self.reduced:
                out["plastic_top"] = self.plastic_top()
            return out

        n_grid = int(round(time_total / dt_grid)) + 1
        grid_t = dt_grid * np.arange(n_grid)
        resampler = GridResampler(dt_grid, n_grid)
        old = state()
        for dt in progressbar(list(dt_sequence)):
            t_old = sh.time
            if self.reduced:
                t_new = t_old + dt
                k = min(int(np.searchsorted(grid_t, t_new)), n_grid - 1)
                w = 1.0 if k == 0 else np.clip((t_new - grid_t[k - 1]) / dt_grid, 0.0, 1.0)
                self.next_iface = iface_on_grid[0] if k == 0 else (1 - w) * iface_on_grid[k - 1] + w * iface_on_grid[k]
            self.step(dt)
            new = state()
            interp = {"iface": (old["iface"], new["iface"])}
            if "plastic_top" in new:
                interp["plastic_top"] = (old["plastic_top"], new["plastic_top"])
            resampler.step(t_old, sh.time, interp=interp,
                           extrap={"u_rows": (old["u_rows"], old["v_rows"]), "v_rows": (old["v_rows"], 0.0)})
            old = new
            if resampler.complete:
                break
        resampler.finish(sh.time, old)
        if not resampler.complete:
            raise RuntimeError(f"Steps end at t = {sh.time:.4g} before the grid end {time_total:.4g}")
        return resampler.data
