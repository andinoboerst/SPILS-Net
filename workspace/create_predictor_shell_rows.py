"""
Row-input surrogate for the square shell tube (fem_sim.shell_tct): the bottom half is kept as an explicit RM
shell model, the top half is replaced by a GNN that predicts the interface-ring displacements from the
displacements of the node rings 2.5..10 mm below the interface. Rotations at the interface stay free.

Workflow (all explicit, with possibly varying steps; data streamed onto the surrogate grid of 5e-7 s):

    --generate            full tube runs per frequency -> per-sim grid data (rows, interface, top plasticity)
    --assemble            stack, PCA of the top-half plastic strain, ring topology
    --generate-perturbed  reduced-tube replays with the interface driven by truth + a known error (denoising)
    --train               SPILS-Net GNN on the ring graph
    --apply               coupled run: reduced tube + surrogate (subcycled) vs. the full tube in lockstep

Data in training_data_shell[_<tag>]/, models in surrogate_models_shell/, results in results_shell/.
"""

import argparse
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import create_predictor_3d_inv as cp
import create_predictor_3d_rows as cr
from nn_predictors.misc import row_features
from spilsnet import SPILSNetGraph, set_seed

from fem_sim.progress_bar import progressbar
from fem_sim.shell_tct import ShellTubeTCT

logger = logging.getLogger("PredictorShell_Rows")

SURROGATE_DT = cr.SURROGATE_DT
DEPTHS = (0.0, 2.5, 5.0, 7.5, 10.0)  # sampled; rows at 2.5..10 mm are the inputs
INPUT_DEPTHS = (2.5, 5.0, 7.5, 10.0)
METHOD = "spils_net_gnn_shell_rows"
DATA_FOLDER = Path("training_data_shell")
MODEL_FOLDER = Path("surrogate_models_shell")
RESULTS_FOLDER = Path("results_shell")


def _sim_path(i: int) -> Path:
    return DATA_FOLDER / "sims" / f"shell_grid_sim{i:02d}.npz"


def _pert_path(i: int, tag: str) -> Path:
    return DATA_FOLDER / "sims" / f"shell_pert{tag}_sim{i:02d}.npz"


def _model_path(version: str) -> Path:
    return MODEL_FOLDER / f"model_shell_{METHOD}_v{version}"


def _features(u_rows: np.ndarray) -> np.ndarray:
    """Node-major displacement features of the input rings; u_rows [..., len(DEPTHS), n, 3]."""
    idx = [DEPTHS.index(d) for d in INPUT_DEPTHS]
    return row_features(u_rows[..., idx, :, :], u_rows[..., idx, :, :], INPUT_DEPTHS, True, False)


def generate(args) -> None:
    (DATA_FOLDER / "sims").mkdir(parents=True, exist_ok=True)
    sims = args.sims if args.sims is not None else list(range(len(cp.FREQUENCY_RANGE)))
    for i in sims:
        if _sim_path(i).exists():
            logger.info("Skipping sim %d: already saved", i)
            continue
        steps = cr._dt_sequence(args.dt_max, args.dt_jitter, args.time_total, seed=100 + i)
        tube = ShellTubeTCT(frequency=cp.FREQUENCY_RANGE[i])
        logger.info("Shell sim %d (%d Hz): %d steps, dt in [%.3g, %.3g]", i, cp.FREQUENCY_RANGE[i], len(steps), steps.min(), steps.max())
        g = tube.generate(steps, SURROGATE_DT, DEPTHS, args.time_total)
        n = len(g["iface"]) - 1
        np.savez_compressed(_sim_path(i), u_rows=g["u_rows"][:n], v_rows=g["v_rows"][:n], iface=g["iface"],
                            plastic_top=g["plastic_top"], frequency=cp.FREQUENCY_RANGE[i])
        logger.info("Saved %s (max top plastic strain %.2e)", _sim_path(i), g["plastic_top"].max())


def assemble(args) -> None:
    """
    Stack the per-sim grid data. The internal state is the top-half equivalent plastic strain (all elements
    and thickness points) reduced by PCA. Unlike the solid tube, one component fitted on a single run does
    not describe the shell field at all frequencies (where it yields shifts between the interface region and
    the clamp), so the basis is fitted on all training sims (every 10th grid step) with --pca-components.
    """
    from sklearn.decomposition import PCA

    n_sims = len(cp.FREQUENCY_RANGE)
    missing = [i for i in range(n_sims) if not _sim_path(i).exists()]
    if missing:
        raise FileNotFoundError(f"Shell data missing for sims {missing}")
    sample = []
    for i in cp.DEFAULT_TRAIN_SIMS:
        with np.load(_sim_path(i)) as d:
            sample.append(d["plastic_top"][::10])
    pca = PCA(n_components=args.pca_components).fit(np.concatenate(sample))
    del sample
    logger.info("PCA of the top-half plastic strain: %d components, explained variance %s",
                args.pca_components, np.round(pca.explained_variance_ratio_, 4))
    disps, internal = [], []
    for i in range(n_sims):
        with np.load(_sim_path(i)) as d:
            disps.append(d["iface"][1:])
            internal.append((d["plastic_top"] - pca.mean_) @ pca.components_.T)
    np.savez_compressed(DATA_FOLDER / "dataset.npz", disps=np.array(disps), internal=np.array(internal),
                        pca_mean=pca.mean_, pca_components=pca.components_)
    tube = ShellTubeTCT(frequency=cp.FREQUENCY_RANGE[0], reduced=True)
    coords, edges = tube.interface_topology()
    np.savez_compressed(DATA_FOLDER / "topology.npz", coords=coords, edges=edges)
    logger.info("Assembled %d sims, %d grid steps, %d interface nodes", n_sims, len(disps[0]), len(coords))


def generate_perturbed(args) -> None:
    coords = np.load(DATA_FOLDER / "topology.npz")["coords"]
    sims = args.sims if args.sims is not None else list(range(len(cp.FREQUENCY_RANGE)))
    for i in sims:
        if _pert_path(i, args.pert_tag).exists():
            logger.info("Skipping sim %d: perturbed replay already saved", i)
            continue
        with np.load(_sim_path(i)) as d:
            truth, clean_u, clean_v = d["iface"], d["u_rows"], d["v_rows"]
        n = len(truth) - 1
        errors = args.perturb_scale * cp.interface_perturbation(
            coords, n, np.random.default_rng(1000 + i), fast_rms=args.fast_rms, persistent_rms=args.persistent_rms)
        grid_iface = truth.copy()
        grid_iface[1:] += errors
        steps = cr._dt_sequence(args.dt_max, args.dt_jitter, args.time_total, seed=2000 + i)
        tube = ShellTubeTCT(frequency=cp.FREQUENCY_RANGE[i], reduced=True)
        logger.info("Perturbed replay of sim %d: |E| rms %.2e mm, %d steps", i, np.sqrt(np.mean(errors**2)), len(steps))
        g = tube.generate(steps, SURROGATE_DT, DEPTHS, args.time_total, iface_on_grid=grid_iface)
        np.savez_compressed(_pert_path(i, args.pert_tag), du_rows=g["u_rows"][:n] - clean_u, dv_rows=g["v_rows"][:n] - clean_v,
                            errors=errors)


def load_dataset(args):
    with np.load(DATA_FOLDER / "dataset.npz") as d:
        disps, internal = d["disps"], d["internal"]
    n = disps.shape[1]
    internal = internal[:, :n]
    x = []
    for i in range(len(disps)):
        with np.load(_sim_path(i)) as d:
            x.append(_features(d["u_rows"]))
    x = np.array(x)
    y = disps.copy()
    train_sims = list(cp.DEFAULT_TRAIN_SIMS)
    if args.denoise:
        # As the solid row model: perturbed rows (deviation scaled by a random alpha) with the true target
        rng = np.random.default_rng(8)
        x_p, y_p, available = [], [], []
        for i in cp.DEFAULT_TRAIN_SIMS:
            if not _pert_path(i, args.pert_tag).exists():
                continue
            with np.load(_sim_path(i)) as clean, np.load(_pert_path(i, args.pert_tag)) as pert:
                alpha = np.exp(rng.uniform(np.log(0.03), 0.0, size=(n, 1, 1, 1))) * rng.choice([-1.0, 1.0], size=(n, 1, 1, 1))
                x_p.append(_features(clean["u_rows"] + alpha * pert["du_rows"]))
                y_p.append(disps[i])
                available.append(i)
        if not available:
            raise FileNotFoundError("No perturbed shell replays found; run --generate-perturbed first.")
        new = list(range(len(x), len(x) + len(available)))
        x = np.concatenate([x, np.array(x_p)])
        y = np.concatenate([y, np.array(y_p)])
        internal = np.concatenate([internal, internal[available]])
        train_sims += new
        logger.info("Denoising: perturbed copies of sims %s as training sims %s", available, new)
    logger.info("Shell dataset: X %s, Y %s, internal %s", x.shape, y.shape, internal.shape)
    return x, y, internal, train_sims


def _ring_graph():
    topo = np.load(DATA_FOLDER / "topology.npz")
    coords, edges = topo["coords"], topo["edges"]
    edge_index = np.concatenate([edges, edges[:, ::-1]]).T  # both directions
    radius = np.max(np.linalg.norm(coords[:, :2], axis=1))
    return coords, edge_index, coords[:, :2] / radius


def train(args) -> None:
    if args.seed is not None:
        set_seed(args.seed)
    x, y, internal, train_sims = load_dataset(args)
    coords, edge_index, node_xy = _ring_graph()
    n_nodes, n_features = len(coords), x.shape[2] // len(coords)
    gnn_args = SimpleNamespace(tuned=True, internal_noise=args.internal_noise, smoothness_gamma=0.0,
                               mean_weight=args.mean_weight, mean_patches=args.mean_patches)
    params = cr._gnn_parameters(n_nodes, n_features, local_skip=True, args=gnn_args)
    params["node_coordinates"] = node_xy
    params["model_config"]["internal_state_size"] = internal.shape[2]
    if args.epochs:
        params["hyperparameters"]["num_epochs"] = args.epochs
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    reg = SPILSNetGraph(edge_index=edge_index, save_path=str(_model_path(args.version)), **params)
    logger.info("Shell GNN v%s: %d ring nodes, %d features per node, %d parameters", args.version, n_nodes, n_features, reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x, y, internal, train_indices=train_sims, val_indices=cp.DEFAULT_VAL_SIMS, test_indices=cp.DEFAULT_TEST_SIMS)
    logger.info("Training complete in %.0f s", time.time() - t0)


def apply(args) -> None:
    model = SPILSNetGraph.load(f"{_model_path(args.version)}_best")
    model._model.eval()
    model.initialize_memory_variables()
    with np.load(DATA_FOLDER / "dataset.npz") as d:
        pca_mean, pca_components = d["pca_mean"], d["pca_components"]

    steps = cr._dt_sequence(args.dt_max, args.dt_jitter, args.time_total, seed=0)
    red = ShellTubeTCT(frequency=args.freq, reduced=True)
    ref = ShellTubeTCT(frequency=args.freq)
    rings = red.ring_nodes(INPUT_DEPTHS)
    n_red = len(red.points)  # the reduced points are the first n_red points of the full tube
    n_if = len(red.interface_dofs)
    logger.info("Coupled shell run v%s, %d Hz: %d steps, dt in [%.3g, %.3g], surrogate step %.3g",
                args.version, args.freq, len(steps), steps.min(), steps.max(), SURROGATE_DT)

    out = {k: [] for k in ("times", "ref_iface", "sm_iface", "sm_internal", "ref_internal", "sm_in", "error")}
    t_from = t_to = 0.0
    iface_from = iface_to = np.zeros(n_if)
    x = None
    internal = np.zeros(len(pca_components))
    for dt in progressbar(list(steps)):
        t_old, t_new = red.time, red.time + dt
        ref_internal = (ref.plastic_top() - pca_mean) @ pca_components.T
        while t_to < t_new - 1e-9 * SURROGATE_DT:
            T = t_to
            u_ext = red.u()[: 3 * red.shell.n_nodes] + (T - t_old) * red.v()
            u_rows = np.stack([red.shell.nodal(u_ext, r) for r in rings])
            x = row_features(u_rows, u_rows, INPUT_DEPTHS, True, False)
            if args.true_internal:
                # Diagnostic: replace the surrogate's internal state by the reference's (state before this step)
                import torch
                model.hidden_state = torch.tensor(model.internal_scaler.transform(ref_internal[None]), dtype=torch.float64, device=model.device)
            if getattr(model, "hidden_state", None) is not None:
                internal = model.internal_scaler.inverse_transform(model.hidden_state.detach().cpu().numpy()).ravel()
            prediction = model.predict(x)
            t_from, iface_from = T, iface_to
            t_to, iface_to = T + SURROGATE_DT, prediction
        w = (t_new - t_from) / SURROGATE_DT
        red.next_iface = (1 - w) * iface_from + w * iface_to
        red.step(dt)
        ref.step(dt)
        u_red = red.shell.nodal(red.u(), np.arange(n_red))
        u_ref = ref.shell.nodal(ref.u(), np.arange(n_red))
        out["times"].append(red.time)
        out["ref_iface"].append(ref.u()[ref.interface_dofs].copy())
        out["sm_iface"].append(red.u()[red.interface_dofs].copy())
        out["sm_internal"].append(internal)
        out["ref_internal"].append(ref_internal)
        out["sm_in"].append(x)
        out["error"].append(np.linalg.norm(u_red - u_ref) / max(np.linalg.norm(u_ref), 1e-12))
    out = {k: np.array(v) for k, v in out.items()}
    rel_if = np.linalg.norm(out["sm_iface"] - out["ref_iface"]) / np.linalg.norm(out["ref_iface"])
    z_shift = (out["sm_iface"] - out["ref_iface"]).reshape(len(steps), -1, 3)[:, :, 2].mean(1) * 1e3
    logger.info("Interface relative error %.2f%%, max |rigid z-shift| %.2f um, rms domain error %.2f%%",
                100 * rel_if, np.abs(z_shift).max(), 100 * np.sqrt(np.mean(out["error"] ** 2)))
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = RESULTS_FOLDER / f"shell_v{args.version}_f{args.freq}_{args.run_tag}.npz"
    np.savez_compressed(path, **out)
    logger.info("Saved %s", path)


def _shell_state(tube: ShellTubeTCT) -> list:
    sh = tube.shell
    return [sh.u.x.array, sh.theta.x.array, sh.v_u, sh.v_t, sh.a_u, sh.a_t, sh.eps_p, sh.eq_eps_p]


def loop_gain(args) -> None:
    """
    Closed-loop error growth of the coupled shell model (as --loop-gain for the solid rows model): two coupled
    reduced tubes A and B in lockstep, each with its own surrogate, B starting from A plus a small interface
    difference. Every --renorm surrogate steps the difference of the complete state (shell fields, plastic
    state and the surrogate's internal state) is rescaled to --eps, so the growth stays linear (Benettin).
    """
    import torch

    models = []
    for _ in range(2):
        m = SPILSNetGraph.load(f"{_model_path(args.version)}_best")
        m._model.eval()
        m.initialize_memory_variables()
        models.append(m)
    steps = cr._dt_sequence(args.dt_max, args.dt_jitter, args.time_total, seed=0)
    tubes = [ShellTubeTCT(frequency=args.freq, reduced=True) for _ in range(2)]
    rings = tubes[0].ring_nodes(INPUT_DEPTHS)
    coords = tubes[0].interface_topology()[0]
    n_if = len(tubes[0].interface_dofs)
    rng = np.random.default_rng(0)
    state = [dict(t_from=0.0, t_to=0.0, i_from=np.zeros(n_if), i_to=np.zeros(n_if)) for _ in range(2)]

    def predict(k, T, t_old):
        tube, m = tubes[k], models[k]
        u_ext = tube.u()[: 3 * tube.shell.n_nodes] + (T - t_old) * tube.v()
        u_rows = np.stack([tube.shell.nodal(u_ext, r) for r in rings])
        return m.predict(row_features(u_rows, u_rows, INPUT_DEPTHS, True, False))

    def difference():
        return np.linalg.norm(state[1]["i_to"] - state[0]["i_to"])

    def rescale(factor):
        for a, b in zip(_shell_state(tubes[0]), _shell_state(tubes[1])):
            b[...] = a + factor * (b - a)
        for key in ("i_from", "i_to"):
            state[1][key] = state[0][key] + factor * (state[1][key] - state[0][key])
        h0, h1 = models[0].hidden_state, models[1].hidden_state
        if h0 is not None and h1 is not None:
            models[1].hidden_state = h0 + factor * (h1 - h0)

    logs, n_calls, perturbed, window_start = [], 0, False, None
    for dt in progressbar(list(steps)):
        t_old, t_new = tubes[0].time, tubes[0].time + dt
        while state[0]["t_to"] < t_new - 1e-9 * SURROGATE_DT:
            T = state[0]["t_to"]
            for k in range(2):
                p = predict(k, T, t_old)
                state[k].update(t_from=T, i_from=state[k]["i_to"], t_to=T + SURROGATE_DT, i_to=p)
            n_calls += 1
            if not perturbed and n_calls >= args.start_step:
                # Random smooth interface difference of size eps on B's latest prediction
                d = cp.interface_perturbation(coords, 60, rng, jitter_rms=0.0)[-1]
                state[1]["i_to"] = state[0]["i_to"] + args.eps * d / np.linalg.norm(d)
                perturbed, window_start = True, n_calls
            elif perturbed and n_calls - window_start >= args.renorm:
                g = difference() / args.eps
                delta = (state[1]["i_to"] - state[0]["i_to"]).reshape(-1, 3)
                rigid = np.linalg.norm(delta.mean(0)) * np.sqrt(len(delta)) / max(np.linalg.norm(delta), 1e-30)
                logs.append((n_calls, g ** (1.0 / args.renorm), rigid, delta.copy()))
                rescale(1.0 / g)
                window_start = n_calls
        for k in range(2):
            w = (t_new - state[k]["t_from"]) / SURROGATE_DT
            tubes[k].next_iface = (1 - w) * state[k]["i_from"] + w * state[k]["i_to"]
            tubes[k].step(dt)
        if n_calls >= args.end_step:
            break
    gains = np.array([l[1] for l in logs[3:]])
    print(f"per-window growth per surrogate step: {' '.join(f'{l[1]:.4f}' for l in logs)}")
    print(f"geometric mean (after 3 windows): {np.exp(np.mean(np.log(gains))):.4f}; "
          f"share of the difference in the rigid interface translation (last windows): {np.mean([l[2] for l in logs[-3:]]):.3f}")
    last = logs[-1][3]
    print(f"dominant difference shape: rms per component x/y/z {np.sqrt(np.mean(last**2, 0)) / np.sqrt(np.mean(last**2))}")


def main():
    global DATA_FOLDER
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--generate-perturbed", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--loop-gain", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-5, help="--loop-gain: size of the state difference [mm]")
    parser.add_argument("--renorm", type=int, default=20, help="--loop-gain: surrogate steps per renormalisation window")
    parser.add_argument("--start-step", type=int, default=100)
    parser.add_argument("--end-step", type=int, default=400)
    parser.add_argument("--data-tag", type=str, default="")
    parser.add_argument("--sims", type=int, nargs="+", default=None)
    parser.add_argument("--time-total", type=float, default=1e-3)
    parser.add_argument("--dt-max", type=float, default=2.5e-7, help="largest explicit step (critical step ~3.0e-7)")
    parser.add_argument("--dt-jitter", type=float, default=0.5)
    parser.add_argument("--perturb-scale", type=float, default=0.4, help="perturbations relative to the solid model's (smaller interface motion)")
    parser.add_argument("--fast-rms", type=float, default=2e-3)
    parser.add_argument("--persistent-rms", type=float, default=5e-3)
    parser.add_argument("--pert-tag", type=str, default="_persist")
    parser.add_argument("--pca-components", type=int, default=2, help="--assemble: PCA components of the internal state")
    parser.add_argument("--version", type=str, default="s01")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--internal-noise", type=float, default=0.15)
    parser.add_argument("--mean-weight", type=float, default=0.0)
    parser.add_argument("--mean-patches", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None, help="--train: override the number of epochs (testing)")
    parser.add_argument("--true-internal", action="store_true", help="--apply: feed the reference internal state (diagnostic)")
    parser.add_argument("--freq", type=int, default=1000)
    parser.add_argument("--run-tag", type=str, default="full2000")
    args = parser.parse_args()

    if args.data_tag:
        DATA_FOLDER = Path(f"{DATA_FOLDER}_{args.data_tag}")
    if args.generate:
        generate(args)
    if args.assemble:
        assemble(args)
    if args.generate_perturbed:
        generate_perturbed(args)
    if args.train:
        train(args)
    if args.apply:
        apply(args)
    if args.loop_gain:
        loop_gain(args)
    if not (args.generate or args.assemble or args.generate_perturbed or args.train or args.apply or args.loop_gain):
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
