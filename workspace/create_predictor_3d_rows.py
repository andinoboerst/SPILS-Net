"""
3D "row input" predictor for SPILS-Net: interface displacement from the fields just inside the interface.

The removed top half is replaced by a surrogate that predicts the interface displacement increment from
the displacement and velocity of the retained (bottom) domain sampled at fixed depths below every
interface node (default 0, 2.5, 5, 7.5 and 10 mm):

    Input (X):  [u, v at depths d_1..d_k below interface node i] per node   (node-major)
    Output (Y): u_iface^{n+1} (--target absolute, default) or u_iface^{n+1} - u_iface^n (increment)

By default the interface itself (depth 0) is not an input: its displacement is the surrogate's own output
and its velocity is that output differentiated by the time integrator, so both would feed surrogate
errors straight back. With an absolute target there is also no step-to-step accumulation of outputs.
    Internal:   Top-half plastic strain state (PCA reduced, 1 feature), as in the inverse model

No interface force is used. Sampling at fixed depths below each interface node keeps the input
one-to-one with the interface nodes, so the existing interface graph is reused.

Commands (inside Docker, from workspace/):
  --generate     Replay training sims on the reduced domain with the true interface displacements and
                 save the sampled rows (one process per sim is safest: pass one --sims value per call)
  --train        Train the row-input GNN
  --loop-gain    Growth rate of interface errors in the coupled loop around the true trajectory
  --apply        Closed-loop evaluation against the full FEM reference
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

import create_predictor_3d_inv as cp
from nn_predictors.misc import row_features, smooth_projection
from spilsnet import SPILSNetGraph, set_seed
from spilsnet_params import SPILSNET_3D_GNN_CONFIG, SPILSNET_3D_GNN_HYPERPARAMETERS, MinMaxScaler

if cp.FEM_AVAILABLE:
    from fem_sim.interface_rows import InterfaceRowSampler
    from fem_sim.tct_displacements_3d import TCTApplyRowDisplacements3D, TCTExtractRowsExplicit3D, TCTReplayDisplacements3D
    from fem_sim.time_grid import GridResampler
    from fem_sim.progress_bar import progressbar

logger = logging.getLogger("Predictor3D_Rows")

# Depths sampled by --generate; training and evaluation select a subset with --depths
DEPTHS = (0.0, 2.5, 5.0, 7.5, 10.0)
DEFAULT_INPUT_DEPTHS = (2.5, 5.0, 7.5, 10.0)
ROWS_FOLDER = cp.DATA_FOLDER / "rows"
METHOD = "spils_net_gnn_rows"
# Time step of the surrogate: the step of the (implicit) training data, and the grid explicit data is resampled to
SURROGATE_DT = 5e-7


def _rows_path(law: str, sim_index: int) -> Path:
    return ROWS_FOLDER / f"rows_3d_{law}_sim{sim_index:02d}.npz"


def _rows_pert_path(law: str, sim_index: int, tag: str = "") -> Path:
    return ROWS_FOLDER / f"rows_pert{tag}_3d_{law}_sim{sim_index:02d}.npz"


def _model_path(args) -> Path:
    return cp.MODEL_FOLDER / f"model_3d_{args.law}_{METHOD}_v{args.version}"


def _gnn_parameters(n_nodes: int, n_features: int, local_skip: bool, args=None) -> dict:
    """GNN settings; the --tuned flag switches to the configuration revised for absolute row targets."""
    model_config = {
        **SPILSNET_3D_GNN_CONFIG,
        "input_size": n_nodes * n_features,
        "input_node_features": n_features,
        "use_local_skip": local_skip,
    }
    hyperparameters = dict(SPILSNET_3D_GNN_HYPERPARAMETERS)
    if args is not None and args.tuned:
        # Unbounded output (no tanh saturation at the scaled extremes), local linear path on the final
        # output, no dropout, and a lower learning rate with more patience for the noisy validation loss
        model_config.update({"output_activation": "none", "local_skip_on_output": True, "dropout_rate": 0.0})
        hyperparameters.update({"learning_rate": 1e-3, "early_stop_patience": 100})
    if args is not None and getattr(args, "internal_weight", None) is not None:
        # Weight of the internal-state term in the loss (the interface term keeps its weight)
        hyperparameters["loss_beta"] = args.internal_weight
    if args is not None and getattr(args, "patience", None):
        hyperparameters["early_stop_patience"] = args.patience
    if args is not None and args.internal_noise > 0:
        hyperparameters["internal_noise_std"] = args.internal_noise
    if args is not None and args.smoothness_gamma > 0:
        # Dirichlet energy of the physical, per-component normalised error with cotangent edge weights
        if not args.tuned:
            raise ValueError("--smoothness-gamma needs --tuned (cotangent weights use the node positions).")
        hyperparameters.update({"loss_gamma": args.smoothness_gamma, "smoothness_mode": "dirichlet",
                                "smoothness_weights": "cotangent", "smoothness_physical": True})
    if args is not None and args.mean_weight > 0:
        # Patch-mean error term against the slow drift of the interface along its weakly restored modes
        if args.mean_patches > 1 and not args.tuned:
            raise ValueError("--mean-patches > 1 needs --tuned (patches are clustered on the node positions).")
        hyperparameters.update({"loss_mean_weight": args.mean_weight, "loss_mean_patches": args.mean_patches})
    return {
        "model_config": model_config,
        "hyperparameters": hyperparameters,
        "input_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
        "internal_in_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
        "internal_out_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
        "output_scaler_class": MinMaxScaler(feature_range=(-1, 1)),
    }


def _replay_sim(law: str, sim_index: int):
    sim = TCTReplayDisplacements3D(frequency=cp.FREQUENCY_RANGE[sim_index], constitutive_model=law, configuration="scaled")
    sim.time_total = cp.Sim3DConfig.time_total
    sim.setup()
    return sim


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_rows(args) -> None:
    """
    Replay sims with the true interface displacements and save u, v sampled below the interface.

    With --perturb-scale > 0 the interface is driven with the truth plus a known error E (the same
    perturbation as the inverse corrective replays) and the deviations of the sampled rows from the
    clean replay are saved instead, for denoising training.
    """
    ROWS_FOLDER.mkdir(parents=True, exist_ok=True)
    config = cp.Sim3DConfig(constitutive_law=args.law)
    _, _, disps, _ = cp.load_dataset(config)
    sims = args.sims if args.sims is not None else list(range(len(disps)))
    perturbed = args.perturb_scale > 0
    if perturbed:
        coords = np.load(cp.DATA_FOLDER / f"interface_topology_3d_{args.law}.npz")["coords"]

    for i in sims:
        path = _rows_pert_path(args.law, i, args.pert_tag) if perturbed else _rows_path(args.law, i)
        if path.exists():
            logger.info("Skipping sim %d: %s already saved", i, path.name)
            continue
        if perturbed and not _rows_path(args.law, i).exists():
            raise FileNotFoundError(f"Clean rows of sim {i} are needed first: {_rows_path(args.law, i)}")
        errors = np.zeros_like(disps[i])
        if perturbed:
            errors = args.perturb_scale * cp.interface_perturbation(
                coords, disps.shape[1], np.random.default_rng(1000 + i), fast_rms=args.fast_rms, persistent_rms=args.persistent_rms
            )
            logger.info("Perturbation of sim %d: |E| rms %.2e mm", i, np.sqrt(np.mean(errors**2)))

        sim = _replay_sim(args.law, i)
        sampler = InterfaceRowSampler(sim.V, sim.interface_nodes, DEPTHS)
        n_steps = disps.shape[1]
        u_rows = np.zeros((n_steps, len(DEPTHS), sampler.n_nodes, 3))
        v_rows = np.zeros_like(u_rows)

        logger.info("Row replay of sim %d (%d Hz), %d steps", i, cp.FREQUENCY_RANGE[i], n_steps)
        max_iface_mismatch = 0.0
        for step in progressbar(range(n_steps)):
            sim.step = step
            sim.advance_time()
            u_rows[step] = sampler.sample(sim.u_next.x.array)
            v_rows[step] = sampler.sample(sim.v_next.x.array)
            # Depth 0 must reproduce the interface displacement applied in the previous step
            u_prev = disps[i, step - 1] + errors[step - 1] if step > 0 else np.zeros(disps.shape[2])
            max_iface_mismatch = max(max_iface_mismatch, np.abs(u_rows[step, 0].reshape(-1) - u_prev).max())
            sim.next_iface = disps[i, step] + errors[step]
            sim.solve_time_step()

        logger.info("Max |depth-0 sample - applied interface displacement| = %.2e mm", max_iface_mismatch)
        if perturbed:
            with np.load(_rows_path(args.law, i)) as clean:
                du_rows, dv_rows = u_rows - clean["u_rows"], v_rows - clean["v_rows"]
            np.savez_compressed(path, du_rows=du_rows, dv_rows=dv_rows, errors=errors, depths=np.array(DEPTHS),
                                frequency=cp.FREQUENCY_RANGE[i])
        else:
            np.savez_compressed(path, u_rows=u_rows, v_rows=v_rows, depths=np.array(DEPTHS), frequency=cp.FREQUENCY_RANGE[i])
        logger.info("Saved %s", path)


def _model_inputs(u_rows: np.ndarray, v_rows: np.ndarray, args):
    """
    Model inputs and output anchor from rows sampled at args.depths ([..., n_depths, n_nodes, 3]).

    Default: the row features (displacements, optionally velocities) and a zero anchor. With --anchor:
    the differences between neighbouring rows (strain-like, free of rigid-body translation) as inputs and
    the displacement of the first row as anchor, so the model predicts the jump u_iface - u(d_1).
    """
    if args.anchor:
        diffs = u_rows[..., :-1, :, :] - u_rows[..., 1:, :, :]
        x = row_features(diffs, diffs, list(range(diffs.shape[-3])), True, include_velocity=False)
        anchor = u_rows[..., 0, :, :].reshape(*u_rows.shape[:-3], -1)
        return x, anchor
    x = row_features(u_rows, v_rows, args.depths, not args.no_iface_velocity, not args.no_velocity)
    return x, np.zeros(x.shape[:-1] + (u_rows.shape[-2] * 3,))


def load_rows_dataset(args):
    """
    X: row features per node; Y: interface displacement (absolute or increment); internal: plastic PCA
    state; and the training sim indices. With --denoise, perturbed copies of the training sims are
    appended: rows from a replay with a perturbed interface (deviations rescaled by a random alpha,
    log-uniform magnitude in [0.03, 1] with random sign) paired with the unchanged true target, so the
    model learns not to pass interface errors that come back through the rows into its output.
    """
    config = cp.Sim3DConfig(constitutive_law=args.law)
    _, _, disps, internal = cp.load_dataset(config)

    def depth_index(data):
        saved = list(np.round(data["depths"], 6))
        return [saved.index(round(d, 6)) for d in args.depths]

    if args.anchor and args.target != "absolute":
        raise ValueError("--anchor needs --target absolute.")

    x, anchors = [], []
    for i in range(len(disps)):
        with np.load(_rows_path(args.law, i)) as data:
            idx = depth_index(data)
            xi, ai = _model_inputs(data["u_rows"][:, idx], data["v_rows"][:, idx], args)
            x.append(xi)
            anchors.append(ai)
    x, anchors = np.array(x), np.array(anchors)

    if args.target == "absolute":
        y = disps - anchors
    else:
        u_prev = np.zeros_like(disps)
        u_prev[:, 1:] = disps[:, :-1]
        y = disps - u_prev
    logger.info("Row dataset (depths %s): X %s, Y %s (%s), internal %s", list(args.depths), x.shape, y.shape, args.target, internal.shape)

    train_sims = list(config.train_sims)
    if args.denoise:
        if args.target != "absolute":
            raise ValueError("--denoise needs --target absolute (the clean target is the true interface displacement).")
        available = [i for i in config.train_sims if _rows_pert_path(args.law, i, args.pert_tag).exists()]
        if not available:
            raise FileNotFoundError("No perturbed rows found; run --generate --perturb-scale first.")
        missing = sorted(set(config.train_sims) - set(available))
        if missing:
            logger.warning("No perturbed rows for training sims %s", missing)
        rng = np.random.default_rng(8)
        x_p, y_p = [], []
        for i in available:
            with np.load(_rows_path(args.law, i)) as clean, np.load(_rows_pert_path(args.law, i, args.pert_tag)) as pert:
                idx = depth_index(clean)
                n_steps = clean["u_rows"].shape[0]
                alpha = np.exp(rng.uniform(np.log(0.03), 0.0, size=(n_steps, 1, 1, 1))) * rng.choice([-1.0, 1.0], size=(n_steps, 1, 1, 1))
                u = clean["u_rows"][:, idx] + alpha * pert["du_rows"][:, idx]
                v = clean["v_rows"][:, idx] + alpha * pert["dv_rows"][:, idx]
                xi, ai = _model_inputs(u, v, args)
                x_p.append(xi)
                y_p.append(disps[i] - ai)
        new_indices = list(range(len(x), len(x) + len(available)))
        x = np.concatenate([x, np.array(x_p)])
        y = np.concatenate([y, np.array(y_p)])
        internal = np.concatenate([internal, internal[available]])
        train_sims += new_indices
        logger.info("Denoising: added perturbed copies of sims %s as training sims %s", available, new_indices)

    return x, y, internal, train_sims


def _n_features(args) -> int:
    if args.anchor:
        return 3 * (len(args.depths) - 1)
    n_v = 0 if args.no_velocity else len(args.depths) - (1 if (0.0 in args.depths and args.no_iface_velocity) else 0)
    return 3 * (len(args.depths) + n_v)


# ---------------------------------------------------------------------------
# Training / loading
# ---------------------------------------------------------------------------

def train(args) -> None:
    if args.seed is not None:
        # Re-seed the network initialisation and batch order; the default keeps the import-time seed (8)
        set_seed(args.seed)
    x, y, internal, train_sims = load_rows_dataset(args)
    config = cp.Sim3DConfig(constitutive_law=args.law)
    connectivity = cp.load_interface_topology(config)
    n_nodes = y.shape[2] // 3
    params = _gnn_parameters(n_nodes, x.shape[2] // n_nodes, local_skip=not args.no_local_skip, args=args)
    params["model_config"]["internal_state_size"] = internal.shape[2]
    if args.tuned:
        # Normalised in-plane node positions as extra encoder inputs (the interface lies in one z-plane)
        coords = np.load(cp.DATA_FOLDER / f"interface_topology_3d_{args.law}.npz")["coords"]
        radius = np.max(np.linalg.norm(coords[:, :2], axis=1))
        params["node_coordinates"] = coords[:, :2] / radius

    model_path = _model_path(args)
    if args.resume:
        reg = SPILSNetGraph.load(str(model_path))
        # Saved optimizer state may come back without param_groups; start a fresh optimizer then
        if reg.optimizer_state_dict is not None and not reg.optimizer_state_dict.get("param_groups"):
            reg.optimizer_state_dict = None
        if reg.scheduler_state_dict is None and args.resume_lr is not None:
            # Checkpoint from before the library saved scheduler/patience state: rebuild them from the
            # learning rate reached so far (read off the training log) and the epochs since the best one
            import torch
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            reg.learning_rate = args.resume_lr
            reg.optimizer_state_dict = None
            bad_epochs = reg.curr_epoch - reg.best_epoch
            dummy = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=args.resume_lr)
            sched = ReduceLROnPlateau(dummy, **reg.scheduler_kwargs)
            sched.best, sched.num_bad_epochs = reg.best_val_loss, bad_epochs
            reg.scheduler_state_dict = sched.state_dict()
            reg.patience_counter = bad_epochs
            reg.next_epoch = reg.curr_epoch + 1
            logger.info("Legacy resume: lr %.3g, %d epochs since best, continuing at epoch %d",
                        args.resume_lr, bad_epochs, reg.next_epoch + 1)
    else:
        logger.info("Starting row-input GNN v%s (local skip: %s, tuned: %s)", args.version, not args.no_local_skip, args.tuned)
        logger.info("Model config: %s | hyperparameters: %s", params["model_config"], params["hyperparameters"])
        reg = SPILSNetGraph(connectivity=connectivity, save_path=str(model_path), **params)

    logger.info("Trainable parameters: %d", reg.get_trainable_params())
    t0 = time.time()
    reg.fit(x, y, internal, train_indices=train_sims, val_indices=config.val_sims, test_indices=config.test_sims)
    logger.info("Training complete in %.2fs", time.time() - t0)


def load_model(args) -> SPILSNetGraph:
    model = SPILSNetGraph.load(f"{_model_path(args)}_best")
    model._model.eval()  # loaded modules start in training mode, which would leave dropout active
    if model._model.in_features != _n_features(args):
        raise ValueError(f"Model has {model._model.in_features} input features per node but --depths/--no-velocity/--no-iface-velocity "
                         f"give {_n_features(args)}; pass the settings the model was trained with.")
    return model


def predict_with_internal(model: SPILSNetGraph, x: np.ndarray, internal: np.ndarray) -> np.ndarray:
    """Model output for raw input x at a given raw internal state, without touching model.hidden_state."""
    model._model.eval()
    xs = torch.tensor(model.input_scaler.transform(x.reshape(1, -1)), dtype=torch.float64, device=model.device)
    h = torch.tensor(model.internal_scaler.transform(np.reshape(internal, (1, -1))), dtype=torch.float64, device=model.device)
    with torch.no_grad():
        out, _ = model._model(xs, h, edge_index=model.edge_index)
    return model.output_scaler.inverse_transform(out.cpu().numpy())[0]


# ---------------------------------------------------------------------------
# Loop gain: growth of interface errors in the coupled loop around the true trajectory
# ---------------------------------------------------------------------------

def loop_gain(args) -> None:
    """
    Two reduced domains in lockstep: A is driven by the true interface displacement, B by the truth plus
    an error delta that the surrogate propagates: delta <- model(x_B) - model(x_A) for an absolute target,
    delta <- delta + model(x_B) - model(x_A) for an increment target (the model's own clean error cancels). Every `renorm` steps delta and B's state deviation from A are rescaled to
    the initial size; the geometric-mean growth per step is the dominant closed-loop growth rate
    (< 1: interface errors decay, > 1: they grow). Both use the true internal state.
    """
    config = cp.Sim3DConfig(constitutive_law=args.law)
    _, _, disps, internal = cp.load_dataset(config)
    Y, internal_sim = disps[args.sim_index], internal[args.sim_index]
    model = load_model(args)
    velocity = not args.no_iface_velocity

    sim_a, sim_b = _replay_sim(args.law, args.sim_index), _replay_sim(args.law, args.sim_index)
    sampler = InterfaceRowSampler(sim_a.V, sim_a.interface_nodes, args.depths)
    projection = _output_projection(args)
    carry = 1.0 if args.target == "increment" else 0.0

    # --freeze u|v: B sees A's values for that input channel, isolating the feedback through the other one
    n_d = len(args.depths)
    ones, zeros = np.ones((n_d, sampler.n_nodes, 3)), np.zeros((n_d, sampler.n_nodes, 3))
    frozen = None
    if args.anchor and args.freeze != "none":
        frozen = np.ones(_n_features(args) * sampler.n_nodes, dtype=bool)  # anchored inputs are displacement-only
    elif args.freeze == "u":
        frozen = row_features(ones, zeros, args.depths, velocity, not args.no_velocity).astype(bool)
    elif args.freeze == "v":
        frozen = row_features(zeros, ones, args.depths, velocity, not args.no_velocity).astype(bool)
    elif args.freeze == "all":
        frozen = row_features(ones, ones, args.depths, velocity, not args.no_velocity).astype(bool)

    rng = np.random.default_rng(0)
    d0 = rng.standard_normal(Y.shape[1])
    eps = args.eps

    delta = None
    prev_dir = None
    log_growth, window, cosines, smooth_share = [], [], [], []
    for step in range(args.end_step):
        for sim in (sim_a, sim_b):
            sim.step = step
            sim.advance_time()

        if step == args.start_step:
            delta = eps * d0 / np.linalg.norm(d0)
        elif delta is not None:
            x_a, anchor_a = _model_inputs(sampler.sample(sim_a.u_next.x.array), sampler.sample(sim_a.v_next.x.array), args)
            x_b, anchor_b = _model_inputs(sampler.sample(sim_b.u_next.x.array), sampler.sample(sim_b.v_next.x.array), args)
            if frozen is not None:
                x_b[frozen] = x_a[frozen]
            response = (carry * delta + (anchor_b - anchor_a)
                        + predict_with_internal(model, x_b, internal_sim[step]) - predict_with_internal(model, x_a, internal_sim[step]))
            # --relax a: applied interface = (1 - a) * previous applied + a * prediction
            delta = (1.0 - args.relax) * delta + args.relax * response
            if projection is not None:
                delta = projection @ delta

            # Shape of the propagated error: step-to-step direction change and share in the rigid translation
            direction = delta / np.linalg.norm(delta)
            if prev_dir is not None:
                cosines.append(float(direction @ prev_dir))
            prev_dir = direction
            mean_part = np.tile(delta.reshape(-1, 3).mean(0), len(delta) // 3)
            smooth_share.append(float(np.linalg.norm(mean_part) / np.linalg.norm(delta)))

            if (step - args.start_step) % args.renorm == 0:
                c = eps / np.linalg.norm(delta)
                log_growth.append(-np.log(c))
                window.append(step)
                state_a, state_b = sim_a.snapshot(), sim_b.snapshot()
                sim_b.restore([a + c * (b - a) for a, b in zip(state_a, state_b)])
                delta = delta * c

        sim_a.next_iface = Y[step]
        sim_b.next_iface = Y[step] + (delta if delta is not None else 0.0)
        sim_a.solve_time_step()
        sim_b.solve_time_step()

        if window and window[-1] == step:
            rate = np.exp(log_growth[-1] / args.renorm)
            logger.info("steps %d-%d: growth per step %.4f", step - args.renorm, step, rate)

    rates = np.exp(np.array(log_growth) / args.renorm)
    print("\nwindow end | growth per step")
    for s, r in zip(window, rates):
        print(f"{s:10d} | {r:.4f}")
    skip = min(3, len(rates) - 1)  # let the perturbation align with the dominant direction first
    print(f"geometric mean (after {skip} windows): {np.exp(np.mean(np.log(rates[skip:]))):.4f}")
    if args.save_mode:
        np.save(args.save_mode, delta / np.linalg.norm(delta))
        logger.info("Saved the dominant error direction to %s", args.save_mode)
    late = slice(len(cosines) // 3, None)
    print(f"error shape: mean cos(delta_n, delta_n-1) = {np.mean(cosines[late]):+.3f} (-1: flips sign every step, +1: persistent); "
          f"share of |delta| in the rigid interface translation = {np.mean(smooth_share[late]):.3f}")


# ---------------------------------------------------------------------------
# Closed-loop evaluation
# ---------------------------------------------------------------------------

def _output_projection(args):
    """Projector onto smooth interface fields for --output-degree, or None."""
    if args.output_degree is None:
        return None
    coords = np.load(cp.DATA_FOLDER / f"interface_topology_3d_{args.law}.npz")["coords"]
    logger.info("Projecting predictions onto polynomials of degree <= %d per component", args.output_degree)
    return smooth_projection(coords, args.output_degree)


def _reference_internal_fn(sim, law: str):
    """
    Callable returning the reference simulation's current internal state in the training data's basis:
    the plastic-strain PCA and, if the dataset has them, the top-half displacement modes. None if elastic.
    """
    if law != "plastic":
        return None
    path = cp.DATA_FOLDER / "internal_basis.npz"
    if path.exists():
        with np.load(path) as b:
            basis = {k: b[k] for k in b.files}
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=cp.PCA_N_COMPONENTS).fit(cp._load_npz(cp.DATA_FOLDER / "plastic_top_3d_for_pca.npz"))
        basis = {"plastic_mean": pca.mean_, "plastic_components": pca.components_}
    top_dofs = sim.get_dofs(sim.top_half_nodes) if "u_top_mean" in basis else None

    def current() -> np.ndarray:
        parts = [(sim.eq_epsilon_p_next.x.array[sim.top_half_cells] - basis["plastic_mean"]) @ basis["plastic_components"].T]
        if top_dofs is not None:
            parts.append((sim.u_next.x.array[top_dofs] - basis["u_top_mean"]) @ basis["u_top_components"].T)
        return np.concatenate(parts)

    return current


class _Fixed:
    """Stand-in predictor that returns a given interface vector."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def predict(self, x) -> np.ndarray:
        return self.value


def _dt_sequence(dt_max: float, jitter: float, time_total: float, seed: int) -> np.ndarray:
    """Explicit steps drawn uniformly from [(1 - jitter) dt_max, dt_max] until they cover time_total."""
    dt_min = dt_max * (1.0 - jitter)
    n_max = int(np.ceil(time_total / dt_min)) + 1
    steps = dt_max - (dt_max - dt_min) * np.random.default_rng(seed).random(n_max) if jitter > 0 else np.full(n_max, dt_max)
    return steps[: np.searchsorted(np.cumsum(steps), time_total * (1 - 1e-9)) + 1]


def _explicit_info_path(law: str, sim_index: int) -> Path:
    return ROWS_FOLDER / f"explicit_grid_3d_{law}_sim{sim_index:02d}.npz"


def generate_explicit(args) -> None:
    """
    Explicit data generation: one full-domain run per frequency with explicit steps (optionally varying,
    --dt-jitter), streamed onto the surrogate grid. Saves the rows in the format of --generate and the
    interface displacement and top-half plastic field on the grid; --assemble-explicit then builds the
    dataset files that load_dataset() reads.
    """
    if not args.explicit_dt:
        raise ValueError("--generate-explicit needs --explicit-dt")
    ROWS_FOLDER.mkdir(parents=True, exist_ok=True)
    time_total = args.data_time_total or cp.Sim3DConfig.time_total
    sims = args.sims if args.sims is not None else list(range(len(cp.FREQUENCY_RANGE)))
    topo_path = cp.DATA_FOLDER / f"interface_topology_3d_{args.law}.npz"
    for i in sims:
        info = _explicit_info_path(args.law, i)
        if _rows_path(args.law, i).exists() and info.exists():
            with np.load(info) as d:
                complete = "u_top" in d.files
            if complete:
                logger.info("Skipping sim %d: already saved", i)
                continue
            logger.info("Sim %d: rerunning to add the top-half displacement (same steps, so identical rows)", i)
        steps = _dt_sequence(args.explicit_dt, args.dt_jitter, time_total, seed=100 + i)
        sim = TCTExtractRowsExplicit3D(frequency=cp.FREQUENCY_RANGE[i], constitutive_model=args.law, configuration="scaled",
                                       depths=DEPTHS, dt_grid=SURROGATE_DT)
        sim.time_total = time_total
        logger.info("Explicit run of sim %d (%d Hz): %d steps, dt in [%.3g, %.3g], grid step %.3g",
                    i, cp.FREQUENCY_RANGE[i], len(steps), steps.min(), steps.max(), sim.dt_grid)
        data = sim.generate(steps)
        if not topo_path.exists():
            coords, connectivity = sim.get_interface_topology()
            np.savez_compressed(topo_path, coords=coords, connectivity=connectivity)
        n = len(data["iface"]) - 1  # grid steps; rows[n] -> iface at T_{n+1}
        if not _rows_path(args.law, i).exists():
            np.savez_compressed(_rows_path(args.law, i), u_rows=data["u_rows"][:n], v_rows=data["v_rows"][:n],
                                depths=np.array(DEPTHS), frequency=cp.FREQUENCY_RANGE[i])
        np.savez_compressed(_explicit_info_path(args.law, i), iface=data["iface"], plastic_top=data["plastic_top"],
                            u_top=data["u_top"], dt_min=steps.min(), dt_max=steps.max(), n_steps=len(steps))
        logger.info("Saved %s", _rows_path(args.law, i))


def assemble_explicit(args) -> None:
    """Stack the per-sim explicit grid data into the files load_dataset() reads (in frequency order)."""
    from sklearn.decomposition import PCA

    law = args.law
    n_sims = len(cp.FREQUENCY_RANGE)
    missing = [i for i in range(n_sims) if not _explicit_info_path(law, i).exists()]
    if missing:
        raise FileNotFoundError(f"Explicit data missing for sims {missing}")
    iface, plastic = [], []
    for i in range(n_sims):
        with np.load(_explicit_info_path(law, i)) as d:
            iface.append(d["iface"])
            plastic.append(d["plastic_top"])
    disps = np.array([u[1:] for u in iface])  # disps[n] = interface at T_{n+1}
    np.savez_compressed(cp.DATA_FOLDER / f"training_out_3d_{law}_disps.npz", disps)
    # Forces are not recovered in explicit runs; the loader still expects the file
    np.savez_compressed(cp.DATA_FOLDER / f"training_in_3d_{law}_forces.npz", np.full_like(disps, np.nan))
    np.savez_compressed(cp.DATA_FOLDER / f"training_freqs_3d_{law}_inv.npz", np.array(cp.FREQUENCY_RANGE, dtype=np.int32))
    if law == "plastic":
        # As in the implicit generation: PCA basis from the first simulation's full top-half field
        np.savez_compressed(cp.DATA_FOLDER / "plastic_top_3d_for_pca.npz", plastic[0])
        pca = PCA(n_components=cp.PCA_N_COMPONENTS).fit(plastic[0])
        internal = np.array([pca.transform(p) for p in plastic])
        basis = {"plastic_mean": pca.mean_, "plastic_components": pca.components_}
        if args.top_modes > 0:
            # Memory of the removed region's configuration: PCA modes of the top-half displacement field,
            # fitted on the training sims (every 10th grid step), appended to the plastic state
            u_top = []
            for i in range(n_sims):
                with np.load(_explicit_info_path(law, i)) as d:
                    u_top.append(d["u_top"])
            pca_u = PCA(n_components=args.top_modes).fit(np.concatenate([u_top[i][::10] for i in cp.DEFAULT_TRAIN_SIMS]))
            logger.info("Top-half displacement PCA: %d modes, explained variance %s", args.top_modes, np.round(pca_u.explained_variance_ratio_, 4))
            internal = np.concatenate([internal, np.array([pca_u.transform(u) for u in u_top])], axis=2)
            basis.update({"u_top_mean": pca_u.mean_, "u_top_components": pca_u.components_})
        np.savez_compressed(cp.DATA_FOLDER / f"plastic_top_reduced_3d_{law}.npz", internal)
        np.savez_compressed(cp.DATA_FOLDER / "internal_basis.npz", **basis)
    logger.info("Assembled explicit dataset in %s: disps %s", cp.DATA_FOLDER, disps.shape)


def generate_rows_explicit_perturbed(args) -> None:
    """
    Explicit counterpart of the perturbed replays (--generate --perturb-scale): the bottom half integrates
    explicitly with its own step sequence, its interface follows the perturbed grid values (truth + E)
    interpolated linearly in time, as in the coupled run, and the rows are streamed onto the grid the same
    way as in the clean data. Saves the row deviations from the clean rows.
    """
    time_total = args.data_time_total or cp.Sim3DConfig.time_total
    coords = np.load(cp.DATA_FOLDER / f"interface_topology_3d_{args.law}.npz")["coords"]
    sims = args.sims if args.sims is not None else list(range(len(cp.FREQUENCY_RANGE)))
    for i in sims:
        path = _rows_pert_path(args.law, i, args.pert_tag)
        if path.exists():
            logger.info("Skipping sim %d: %s already saved", i, path.name)
            continue
        with np.load(_explicit_info_path(args.law, i)) as d:
            truth = d["iface"]
        n = len(truth) - 1
        errors = args.perturb_scale * cp.interface_perturbation(
            coords, n, np.random.default_rng(1000 + i), fast_rms=args.fast_rms, persistent_rms=args.persistent_rms
        )
        grid_iface = truth.copy()
        grid_iface[1:] += errors
        logger.info("Perturbation of sim %d: |E| rms %.2e mm", i, np.sqrt(np.mean(errors**2)))

        steps = _dt_sequence(args.explicit_dt, args.dt_jitter, time_total, seed=2000 + i)
        sim = TCTReplayDisplacements3D(frequency=cp.FREQUENCY_RANGE[i], constitutive_model=args.law, configuration="scaled")
        sim.time_integration = "explicit"
        sim.time_total = time_total
        sim.dt = float(steps.min())
        sim.setup()
        sampler = InterfaceRowSampler(sim.V, sim.interface_nodes, DEPTHS)
        dt_grid = SURROGATE_DT
        resampler = GridResampler(dt_grid, n + 1)
        grid_t = dt_grid * np.arange(n + 1)
        old_u, old_v = sampler.sample(sim.u_next.x.array), sampler.sample(sim.v_next.x.array)
        for step in progressbar(range(len(steps))):
            sim.step = step
            sim.dt = float(steps[step])
            t_old = sim.time
            sim.advance_time()
            k = min(int(np.searchsorted(grid_t, sim.time)), n)
            w = 1.0 if k == 0 else np.clip((sim.time - grid_t[k - 1]) / dt_grid, 0.0, 1.0)
            sim.next_iface = grid_iface[k] if k == 0 else (1 - w) * grid_iface[k - 1] + w * grid_iface[k]
            sim.solve_time_step()
            new_u, new_v = sampler.sample(sim.u_next.x.array), sampler.sample(sim.v_next.x.array)
            resampler.step(t_old, sim.time, extrap={"u_rows": (old_u, old_v), "v_rows": (old_v, 0.0)})
            old_u, old_v = new_u, new_v
            if resampler.complete:
                break
        resampler.finish(sim.time, {"u_rows": old_u, "v_rows": old_v})
        with np.load(_rows_path(args.law, i)) as clean:
            du_rows = resampler.data["u_rows"][:n] - clean["u_rows"]
            dv_rows = resampler.data["v_rows"][:n] - clean["v_rows"]
        np.savez_compressed(path, du_rows=du_rows, dv_rows=dv_rows, errors=errors, depths=np.array(DEPTHS),
                            frequency=cp.FREQUENCY_RANGE[i])
        logger.info("Saved %s (row deviation rms %.2e mm at depth 2.5 mm)", path, np.sqrt(np.mean(du_rows[:, 1] ** 2)))


def apply(args) -> None:
    config = cp.Sim3DConfig(constitutive_law=args.law, frequency=args.freq, time_total=args.time_total)
    if args.robin_oracle:
        model = _Fixed(None)  # replaced every step by the reference target below
    else:
        model = load_model(args)
        model.initialize_memory_variables()

    tct_predict = TCTApplyRowDisplacements3D(
        model, frequency=args.freq, constitutive_model=args.law, configuration="scaled",
        depths=args.depths, velocity_at_interface=not args.no_iface_velocity, include_velocity=not args.no_velocity, absolute_target=args.target == "absolute",
        output_projection=_output_projection(args),
        input_builder=(lambda u_s, v_s: _model_inputs(u_s, v_s, args)) if args.anchor else None,
        robin_stiffness=args.robin_k,
        sm_dt=SURROGATE_DT if args.explicit_dt else None,
    )
    if args.robin_k is not None:
        logger.info("Spring (Robin) interface coupling, k = %.4g N/mm^3", args.robin_k)
    tct_predict.time_total = args.time_total
    tct_error = cp._make_extractor(config)
    tct_error.store_field_history = False  # not used here, and it grows by ~1.2 MB per step
    dt_sequence = None
    if args.explicit_dt:
        # Both simulations integrate explicitly with the same step sequence; the surrogate keeps its training step
        dt_max = args.explicit_dt
        dt_min = dt_max * (1.0 - args.dt_jitter)
        dt_sequence = _dt_sequence(dt_max, args.dt_jitter, args.time_total, seed=0)
        for sim in (tct_predict, tct_error):
            sim.time_integration = "explicit"
            sim.dt = dt_min  # num_steps from setup() then covers the whole sequence
        logger.info("Explicit integration: dt in [%.3g, %.3g] (%d steps), surrogate step %.3g", dt_min, dt_max, len(dt_sequence), tct_predict.sm_dt)
    tct_predict.setup()
    tct_error.setup()
    if dt_sequence is not None:
        logger.info("Critical time step estimate: %.3g", tct_error.critical_time_step())
        tct_predict.num_steps = tct_error.num_steps = len(dt_sequence)
    t_order = tct_error.traction_to_interface_order()

    predict_dofs = tct_predict.get_dofs(tct_predict.bottom_half_nodes)
    error_dofs = tct_error.get_dofs(tct_error.bottom_half_nodes)
    errors, truth_norms, iface_errors, iface_truth_norms = [], [], [], []

    # Reference internal state in the basis of the training data (plastic PCA, plus top-half displacement
    # modes if the dataset has them), at the state before each step as the internal input in training
    reference_internal = _reference_internal_fn(tct_error, args.law)
    ref_internal = []
    if args.true_internal:
        logger.info("Using the true internal state of the reference simulation (diagnostic)")

    for step in progressbar(range(tct_predict.num_steps)):
        tct_predict.step = step
        tct_error.step = step
        if dt_sequence is not None:
            tct_predict.dt = tct_error.dt = dt_sequence[step]
        tct_predict.advance_time()
        tct_error.advance_time()
        if reference_internal is not None:
            ref_internal.append(reference_internal())
        tct_error.solve_time_step()
        if args.robin_oracle:
            # Diagnostic: spring target from the reference, u + t / k, which the springs should reproduce exactly
            target = tct_error.u_next.x.array[tct_error.interface_dofs].copy()
            if args.robin_k is not None:
                target += tct_error.calculate_interface_tractions()[t_order] / args.robin_k
            tct_predict.predictor = _Fixed(target)
        if args.true_internal and reference_internal is not None:
            model.hidden_state = torch.tensor(model.internal_scaler.transform(ref_internal[-1][None]), dtype=torch.float64, device=model.device)
        tct_predict.solve_time_step()

        u_true, u_pred = tct_error.u_next.x.array[error_dofs], tct_predict.u_next.x.array[predict_dofs]
        errors.append(np.linalg.norm(u_true - u_pred))
        truth_norms.append(np.linalg.norm(u_true))
        ui_true = tct_error.u_next.x.array[tct_error.interface_dofs]
        ui_pred = tct_predict.u_next.x.array[tct_predict.interface_dofs]
        iface_errors.append(np.linalg.norm(ui_true - ui_pred))
        iface_truth_norms.append(np.linalg.norm(ui_true))

    if dt_sequence is not None:
        # The arrays were allocated for the smallest step; keep the steps that were taken
        n = len(dt_sequence)
        for sim in (tct_predict, tct_error):
            for name in ("data_in", "data_out", "data_internal"):
                if hasattr(sim, name):
                    setattr(sim, name, getattr(sim, name)[:n])
        np.save(cp.RESULTS_FOLDER / f"times_3d_{args.law}_f{args.freq}_{args.run_tag}.npy", np.cumsum(dt_sequence))

    errors, truth_norms = np.array(errors), np.array(truth_norms)
    logger.info("Global Relative Error: %.4f%%", 100 * np.linalg.norm(errors) / max(np.linalg.norm(truth_norms), 1e-10))
    logger.info("Interface Relative Error: %.4f%%", 100 * np.linalg.norm(iface_errors) / max(np.linalg.norm(iface_truth_norms), 1e-10))

    tag = f"_{args.run_tag}" if args.run_tag else ""
    suffix_tct = f"3d_{args.law}_rows_f{args.freq}{tag}"
    suffix_sm = f"3d_{args.law}_{METHOD}_v{args.version}_f{args.freq}{tag}"
    n_iface = len(tct_error.interface_dofs)
    np.save(cp.RESULTS_FOLDER / f"tct_results_in_{suffix_tct}.npy", tct_error.data_in)
    np.save(cp.RESULTS_FOLDER / f"tct_results_out_{suffix_tct}.npy", tct_error.data_out[:, :n_iface])
    np.save(cp.RESULTS_FOLDER / f"sm_results_error_{suffix_sm}.npy", errors / max(np.max(truth_norms), 1e-10))
    np.save(cp.RESULTS_FOLDER / f"sm_results_in_{suffix_sm}.npy", tct_predict.data_in)
    np.save(cp.RESULTS_FOLDER / f"sm_results_out_{suffix_sm}.npy", tct_predict.data_out)
    np.save(cp.RESULTS_FOLDER / f"sm_results_internal_{suffix_sm}.npy", tct_predict.data_internal)
    if ref_internal:
        # The basis depends on the dataset, so the file name carries the data tag
        basis = f"_{args.data_tag}" if args.data_tag else ""
        ref = np.array(ref_internal)
        np.save(cp.RESULTS_FOLDER / f"tct_results_internal_{suffix_tct}{basis}.npy", ref[:, 0] if ref.shape[1] == 1 else ref)
    logger.info("Results saved to %s/ (suffix %s)", cp.RESULTS_FOLDER, suffix_sm)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--law", type=str, choices=["elastic", "plastic"], default="plastic")
    parser.add_argument("--version", type=str, default="01")
    parser.add_argument("--depths", type=float, nargs="+", default=list(DEFAULT_INPUT_DEPTHS),
                        help="Input depths below the interface in mm (default 2.5 5 7.5 10; include 0 for the interface itself)")
    parser.add_argument("--target", type=str, choices=["absolute", "increment"], default="absolute",
                        help="Predict u_iface^{n+1} (absolute) or u_iface^{n+1} - u_iface^n (increment)")
    parser.add_argument("--no-iface-velocity", action="store_true", help="With depth 0 in --depths: drop its velocity")
    parser.add_argument("--no-velocity", action="store_true", help="Use only displacements at the input depths")
    parser.add_argument("--no-local-skip", action="store_true", help="Train without the per-node local skip")
    parser.add_argument("--true-internal", action="store_true",
                        help="--apply: feed the surrogate the true internal state each step (diagnostic)")
    parser.add_argument("--mean-weight", type=float, default=0.0,
                        help="--train: weight of the patch-mean interface error term (against slow interface drift)")
    parser.add_argument("--mean-patches", type=int, default=1,
                        help="--train: patches for --mean-weight (k-means on the node positions; 1 = whole interface)")
    parser.add_argument("--smoothness-gamma", type=float, default=0.0,
                        help="--train: weight of the cotangent Dirichlet smoothness term on physical errors")
    parser.add_argument("--internal-noise", type=float, default=0.0,
                        help="--train: std of noise on the scaled internal-state input (targets unchanged)")
    parser.add_argument("--anchor", action="store_true",
                        help="Inputs: differences between neighbouring rows; output: u(first row) + predicted jump")
    parser.add_argument("--tuned", action="store_true",
                        help="--train: no output tanh, local skip on the output, node positions, no dropout, lr 1e-3")
    parser.add_argument("--sims", type=int, nargs="+", default=None, help="Sims for --generate (default: all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--explicit-dt", type=float, default=None,
                        help="--apply: integrate explicitly with this (maximum) step and run the surrogate subcycled at its training step")
    parser.add_argument("--dt-jitter", type=float, default=0.0,
                        help="--apply with --explicit-dt: draw each step uniformly from [(1 - jitter) dt, dt]")
    parser.add_argument("--robin-k", type=float, default=None,
                        help="--apply: tie the interface to the prediction with springs of this stiffness [N/mm^3] instead of prescribing it")
    parser.add_argument("--robin-oracle", action="store_true",
                        help="--apply: replay the reference interface instead of a model (u, or u + t/k with --robin-k); exactness check")
    parser.add_argument("--internal-weight", type=float, default=None,
                        help="--train: weight of the internal-state loss term (default: the configuration's loss_beta)")
    parser.add_argument("--patience", type=int, default=None,
                        help="--train: early-stopping patience in epochs (default: the configuration's, 100 with --tuned)")
    parser.add_argument("--seed", type=int, default=None, help="--train: training seed (default: the import-time seed 8)")
    parser.add_argument("--resume-lr", type=float, default=None,
                        help="learning rate to resume a checkpoint that has no saved scheduler state")
    parser.add_argument("--perturb-scale", type=float, default=0.0,
                        help="--generate: drive the interface with truth + scale * perturbation and save row deviations")
    parser.add_argument("--fast-rms", type=float, default=0.0,
                        help="--generate with --perturb-scale: rms of the white-in-time low-order-mode component (before scaling)")
    parser.add_argument("--persistent-rms", type=float, default=0.0,
                        help="--generate with --perturb-scale: rms of the slow rigid-mode component (before scaling)")
    parser.add_argument("--pert-tag", type=str, default="",
                        help="Tag of the perturbed-row files written by --generate / read by --denoise")
    parser.add_argument("--denoise", action="store_true",
                        help="--train: add perturbed-row copies of the training sims with the true target")
    parser.add_argument("--sim-index", type=int, default=7, help="--loop-gain: training sim to replay (7 = 997 Hz)")
    parser.add_argument("--start-step", type=int, default=50)
    parser.add_argument("--end-step", type=int, default=450)
    parser.add_argument("--renorm", type=int, default=10, help="--loop-gain: steps between renormalisations")
    parser.add_argument("--eps", type=float, default=1e-6, help="--loop-gain: perturbation size [mm]")
    parser.add_argument("--relax", type=float, default=1.0,
                        help="--loop-gain: relaxation a of the applied interface displacement (1 = none)")
    parser.add_argument("--output-degree", type=int, default=None,
                        help="--loop-gain/--apply: project predictions onto polynomial interface fields of this degree")
    parser.add_argument("--save-mode", type=str, default="", help="--loop-gain: save the final error direction (.npy)")
    parser.add_argument("--freeze", type=str, choices=["none", "u", "v", "all"], default="none",
                        help="--loop-gain: hold this input channel of the perturbed run at its true value")
    parser.add_argument("--freq", type=int, default=1000)
    parser.add_argument("--time-total", type=float, default=0.0003)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generate-explicit", action="store_true",
                        help="explicit full runs streamed onto the surrogate grid (needs --explicit-dt; use with --data-tag)")
    parser.add_argument("--assemble-explicit", action="store_true", help="build the dataset files from --generate-explicit output")
    parser.add_argument("--top-modes", type=int, default=0,
                        help="--assemble-explicit: append this many PCA modes of the top-half displacement to the internal state")
    parser.add_argument("--data-time-total", type=float, default=None,
                        help="--generate-explicit / explicit perturbed replays: simulated time [s] (default: the dataset's 0.001)")
    parser.add_argument("--data-tag", type=str, default="",
                        help="read/write the dataset in <data folder>_<tag> (e.g. 'explicit'); models and results are unchanged")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--loop-gain", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.data_tag:
        global ROWS_FOLDER
        cp.DATA_FOLDER = Path(f"{cp.DATA_FOLDER}_{args.data_tag}")
        cp.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        ROWS_FOLDER = cp.DATA_FOLDER / "rows"
        logger.info("Dataset folder: %s", cp.DATA_FOLDER)

    if args.generate_explicit:
        generate_explicit(args)
    if args.assemble_explicit:
        assemble_explicit(args)
    if args.generate:
        if args.explicit_dt:
            if args.perturb_scale <= 0:
                raise ValueError("With --explicit-dt, --generate makes the perturbed replays; clean rows come from --generate-explicit")
            generate_rows_explicit_perturbed(args)
        else:
            generate_rows(args)
    if args.train:
        train(args)
    if args.loop_gain:
        loop_gain(args)
    if args.apply:
        apply(args)
    if not (args.generate or args.generate_explicit or args.assemble_explicit or args.train or args.loop_gain or args.apply):
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
