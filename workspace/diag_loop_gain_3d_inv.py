"""
Loop-gain diagnostic for the 3D inverse (Forces -> Displacements) coupling.

The reduced bottom-half domain computes the interface force f^n from the displacement it was given,
and the surrogate turns that force into the next interface displacement. A perturbation therefore
travels around the loop

    absolute model:    du^{n+1} = J_f S du^n
    increment model:   du^{n+1} = (I + J_u + J_d + J_f S) du^n - J_d du^{n-1}

where S = d f^{n+1} / d u_iface^{n+1} is the dynamic interface stiffness of the reduced FEM domain and
J_* are the surrogate Jacobians w.r.t. its force / displacement / previous-increment inputs. The
coupling is stable only if the spectral radius of this loop operator stays below 1.

The reduced domain is driven with the true interface displacements of one training simulation. At
each requested step, S is applied matrix-free by finite differences (one reduced FEM solve per
product) and the largest eigenvalues of the loop operator are computed with ARPACK.

Usage (inside Docker, from workspace/):
    python3 diag_loop_gain_3d_inv.py --method spils_net_gnn --steps 100 300 400
"""

import argparse
import logging
import time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigs

import create_predictor_3d_inv as cp
from fem_sim.tct_displacements_3d import TCTReplayDisplacements3D
from nn_predictors.misc import stack_node_features
from spilsnet import SPILSNetGraph

logger = logging.getLogger("LoopGain3D_Inv")

def surrogate_jacobians(model: SPILSNetGraph, x: np.ndarray, internal: np.ndarray, increment: bool, n_dofs: int) -> dict:
    h = torch.tensor(model.internal_scaler.transform(internal.reshape(1, -1)), dtype=torch.float64, device=model.device)
    J = model.input_jacobian(x, hidden_state=h)
    if not increment:
        return {"f": J}
    ones, zeros = np.ones(n_dofs), np.zeros(n_dofs)
    cols = {
        "f": stack_node_features(ones, zeros, zeros).astype(bool),
        "u": stack_node_features(zeros, ones, zeros).astype(bool),
        "d": stack_node_features(zeros, zeros, ones).astype(bool),
    }
    return {k: J[:, mask] for k, mask in cols.items()}


def analyse_step(sim, model, n, X, Y, internal, increment, eps_rel, n_eigs) -> dict:
    n_dofs = Y.shape[1]
    f_n = sim.calculate_interface_tractions()
    force_consistency = np.linalg.norm(f_n - X[n]) / (np.linalg.norm(X[n]) + 1e-12)

    state = sim.snapshot()
    u_base = Y[n]
    eps = eps_rel * max(np.linalg.norm(u_base), 1e-8)
    f_base = sim.force_response(state, u_base)

    n_solves = [0]

    def S(d: np.ndarray) -> np.ndarray:
        n_solves[0] += 1
        norm = np.linalg.norm(d)
        if norm == 0.0:
            return np.zeros_like(d)
        return (sim.force_response(state, u_base + eps * d / norm) - f_base) * norm / eps

    # Linearity check of the finite difference on a random direction
    rng = np.random.default_rng(0)
    d = rng.standard_normal(n_dofs)
    d /= np.linalg.norm(d)
    s1 = S(d)
    s2 = (sim.force_response(state, u_base + 2 * eps * d) - f_base) / (2 * eps)
    fd_nonlinearity = np.linalg.norm(s1 - s2) / (np.linalg.norm(s1) + 1e-12)

    # Mean z-translation of the interface: aggregate stiffness, comparable to the trajectory compliance
    mode_z = np.zeros(n_dofs)
    mode_z[2::3] = 1.0
    S_mode_z = S(mode_z / np.linalg.norm(mode_z))
    stiffness_z = S_mode_z[2::3].sum() / (1.0 / np.linalg.norm(mode_z))

    if increment:
        x_model = stack_node_features(X[n + 1], Y[n], Y[n] - Y[n - 1])
    else:
        x_model = X[n + 1]
    J = surrogate_jacobians(model, x_model, internal[n + 1], increment, n_dofs)

    if increment:
        def loop(z: np.ndarray) -> np.ndarray:
            a, b = z[:n_dofs], z[n_dofs:]
            top = a + J["u"] @ a + J["d"] @ (a - b) + J["f"] @ S(a)
            return np.concatenate([top, a])
        size = 2 * n_dofs
    else:
        def loop(z: np.ndarray) -> np.ndarray:
            return J["f"] @ S(z)
        size = n_dofs

    op = LinearOperator((size, size), matvec=loop, dtype=np.float64)
    t0 = time.time()
    vals = eigs(op, k=n_eigs, which="LM", tol=1e-3, return_eigenvectors=False)
    vals = vals[np.argsort(-np.abs(vals))]

    return {
        "step": n,
        "force_consistency": force_consistency,
        "fd_nonlinearity": fd_nonlinearity,
        "stiffness_z": stiffness_z,
        # Mean uz response to a total Fz of 1 N spread uniformly over the interface nodes
        "compliance_z": J["f"][2::3, 2::3].sum() / (n_dofs // 3) ** 2,
        "spectral_radius": np.abs(vals[0]),
        "eigenvalues": vals,
        "fem_solves": n_solves[0],
        "eig_time": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=str, choices=list(cp.GNN_METHODS), default="spils_net_gnn")
    parser.add_argument("--version", type=str, default="01")
    parser.add_argument("--law", type=str, choices=["elastic", "plastic"], default="plastic")
    parser.add_argument("--sim-index", type=int, default=7, help="Training simulation to replay (7 = 997 Hz)")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 300, 400])
    parser.add_argument("--eps-rel", type=float, default=1e-3, help="FD step relative to |u_iface|")
    parser.add_argument("--n-eigs", type=int, default=4)
    args = parser.parse_args()

    config = cp.Sim3DConfig(constitutive_law=args.law, predictor_method=args.method, predictor_version=args.version)
    increment = cp._is_increment(config)

    X_all, _, Y_all, internal_all = cp.load_dataset(config)
    X, Y, internal = X_all[args.sim_index], Y_all[args.sim_index], internal_all[args.sim_index]
    freq = cp.FREQUENCY_RANGE[args.sim_index]

    model_path = cp.MODEL_FOLDER / f"model_3d_{args.law}_{args.method}_inv_v{args.version}_best"
    model = SPILSNetGraph.load(str(model_path))

    sim = TCTReplayDisplacements3D(frequency=freq, constitutive_model=args.law, configuration="scaled")
    sim.time_total = cp.Sim3DConfig.time_total
    sim.setup()

    logger.info("Replaying training sim %d (%d Hz) on the reduced domain; model %s", args.sim_index, freq, model_path.name)
    results = []
    for step in range(max(args.steps) + 1):
        sim.step = step
        sim.advance_time()
        if step in args.steps:
            res = analyse_step(sim, model, step, X, Y, internal, increment, args.eps_rel, args.n_eigs)
            results.append(res)
            logger.info(
                "step %d | rho = %.3f | eigs %s | S_z = %.3e N/mm | J_z = %.3e mm/N | f-consistency %.1e | FD nonlin %.1e | %d solves, %.0fs",
                res["step"], res["spectral_radius"], np.round(res["eigenvalues"], 3), res["stiffness_z"],
                res["compliance_z"], res["force_consistency"], res["fd_nonlinearity"], res["fem_solves"], res["eig_time"],
            )
        sim.next_iface = Y[step]
        sim.solve_time_step()

    print("\nstep | spectral radius | S_z [N/mm] | J_z [mm/N] | S_z*J_z | leading eigenvalues")
    for r in results:
        print(f"{r['step']:4d} | {r['spectral_radius']:15.3f} | {r['stiffness_z']:.3e} | {r['compliance_z']:.3e} | "
              f"{r['stiffness_z'] * r['compliance_z']:7.3f} | {np.round(r['eigenvalues'], 3)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
