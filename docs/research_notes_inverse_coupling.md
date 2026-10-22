# Research notes: displacement-coupled SPILS-Net surrogates (3D solid and shell)

State as of 2026-10-22. These notes cover the inverse (displacement-prescribing) coupling work on branch
`3D_example` of SPILS-Net and branch `update_3d_laplacian` of spilsnet-torch. They are meant as a
hand-over for implementing the same approach in OpenCourant (the new OpenRadioss), first on a simplified
model and later on crash simulations.

## 1. Goal and setting

- **Application.** Car-crash simulations with shell elements and explicit time stepping (OpenCourant /
  OpenRadioss). Part of the structure is replaced by a surrogate model (SM) that acts on the cut interface.
- **Why displacements.** Applying the interface *forces* of the removed part was unstable at shell interfaces,
  even with ground-truth forces. Prescribing the interface *displacements* is the user's preferred option, and
  it is the more stable one in explicit runs.
- **SPILS-Net.** A GNN encoder, a GRU core with a *physically informed internal state* (trained with teacher
  forcing), and a decoder. The GRU hidden state is rebuilt from the internal state at every step, so **the
  internal state is the model's only memory between steps.**

## 2. Test problems in this repository

| | 3D solid (TCT) | Shell tube |
|---|---|---|
| Geometry | Cylinder R = 25 mm, 100 mm high, tets | Square tube 50 × 50 mm, 100 mm high, t = 1.5 mm, h = 2.5 mm |
| Loading | Bottom driven `1 mm · sin(ωt)` in z, top clamped | Bottom ring driven `0.5 mm · sin(ωt)` axially (rotations fixed), top clamped |
| Interface | Plane z = 50 mm, 151 nodes | Ring at z = 50 mm, 80 nodes |
| Material | J2, linear isotropic hardening (σy = 250 MPa, H = 0.05 E), E = 200 GPa, ν = 0.3, ρ = 7.85e-9 t/mm³ | Same, in plane stress with 5 Gauss points through the thickness |
| Damping | Mass-proportional, c = 0.02 (heavily damped) | Same ratio |
| Frequencies | 22 sims, 500–2000 Hz (`FREQUENCY_RANGE`); 1000 Hz is used as the unseen test case | Same |
| Retained / replaced | Bottom half kept (FE); top half replaced by the SM | Same |

Units throughout: mm, s, t, N, MPa.

The removed top half is **passive**: it is driven only through the interface and clamped at the top. So in
principle the SM can learn the top half's reaction from the interface motion alone, without external
information.

## 3. Formulation that works (current best)

**Inputs.** Displacements of the retained structure sampled at 2.5, 5, 7.5 and 10 mm below every interface
node ("rows"). The features are node-major, so they map one-to-one onto the interface nodes.
- **Exclude the interface itself** from the inputs; it stays on the output side only. It is the quantity being
  prescribed, so feeding it back makes its own errors circular.
- **No velocities.** Including them was unstable (loop gain 4.2 → 2.3).

**Output.** The absolute interface displacement at the next surrogate time. Increments were worse.

**Internal state** (teacher-forced, predicted by the model):
- the PCA of the equivalent plastic strain of the replaced region;
- **plus PCA modes of the replaced region's displacement field** (4 modes for the solid; see §5.7).

**Training recipe** ("tuned" GNN, `--tuned`):
- unbounded output (`output_activation="none"`), per-node local skip on the output, no dropout;
- lr 1e-3, early-stopping patience 100;
- loss weights α = 0.9 (interface), β = 0.1 (internal state).

**Robustness during training:**
- *Denoising* (§5.3): training copies whose rows come from replays with a perturbed interface, paired with
  the unchanged true target.
- *Internal-state noise* (0.15, under test at 0.3): Gaussian noise on the internal-state input during training.

**Coupling.** Prescribe the interface displacements (Dirichlet). For shells, **prescribe displacements only;
leave the interface rotations free**. Prescribing rotations as well over-constrained the boundary in the user's
crash experiments.

## 4. Running a surrogate inside an explicit solver with varying Δt

This is the part that transfers most directly to OpenCourant.

- **Fixed surrogate grid.** The SM keeps a fixed step Δ (here 5e-7 s, its training step) on its own grid
  `T_k = kΔ`. It is called whenever the solver time crosses the next `T_k`. The explicit solver takes any step
  sizes in between (here ~1.25e-7 s, random in [0.625, 1.25]e-7 s).
- **Inputs at T_k.** Take the last solver state at `t_n ≤ T_k` and extrapolate it with its velocity:
  `u_rows(T_k) = u_rows(t_n) + (T_k − t_n) · v_rows(t_n)`. This only needs the current state.
- **Prescribed interface between calls.** Interpolate linearly between the predictions for `T_k` and `T_{k+1}`.
- **The GRU advances once per surrogate call**, which keeps the internal state physically consistent.
- **Training data must be resampled onto the same grid, with the same rules:**
  - inputs extrapolated from the last state before `T_k`;
  - targets and internal state interpolated linearly between the bracketing states.

  Using identical rules in training and application matters. If the solver's output is written at "the first
  cycle after a requested time", also save the exact time and the velocities, so every output can be moved
  back onto the grid.
- **Result.** v09 (trained on implicit data) coupled into the explicit solver with subcycling: 0.36% domain /
  3.4% interface error, the same with constant and random Δt. Subcycling therefore costs no accuracy.
- **Bug to avoid in any explicit implementation: prescribed dofs under varying Δt.**
  - Wrong: matching the acceleration to the prescribed displacement and then integrating the velocity. That
    makes the velocity error flip sign and get scaled by `dt_n / dt_{n−1}` every two steps, which grows without
    bound for random Δt. The run blew up to 10^23 %.
  - Right: compute v and a at prescribed dofs from divided differences of the prescribed positions, with no
    recursion.

  This matters here because the rows at 2.5 mm interpolate from interface nodes, so their velocities feed the
  input extrapolation.

## 5. Findings, in roughly chronological order

### 5.1 Force-input inverse models are unstable or neutral

- Force → displacement models: closed-loop spectral radius 24–78 (a flip-flop mode).
- Incremental variants: ≈ 1.0, i.e. neutral drift.

The underlying reason is the partitioned Dirichlet–Neumann coupling of equal-stiffness halves, which has a
loop factor of about −1.

### 5.2 Judge by closed-loop behaviour, never by one-step accuracy

- The **loop gain** is the growth per surrogate step of a small interface error, fed through the retained FE
  model and back through the SM. Measure it with two coupled models in lockstep, renormalising every N steps
  (Benettin).
- **Pitfall:** call `eval()` on the model before any diagnostic. Otherwise dropout stays active; that made all
  early loop-gain numbers invalid.
- **Example:** v07 had the best test error (0.71%) but a loop gain of 1.18, and its coupled run was 3–4× worse
  than the others.

### 5.3 Denoising works

**Method.**
1. Replay the retained model with the interface driven by truth + a known error E.
2. Record the resulting row deviations.
3. Train on clean rows + α · deviation (α random, log-uniform in [0.03, 1], random sign), with the *true*
   target.

The error E combines smooth slow drift, fast smooth modes and persistent rigid-like modes. It costs one extra
replay per training simulation.

**Effect on loop gain:**

| model | loop gain |
|---|---|
| clean training | 1.022 |
| denoised, slow perturbations | 1.004 |
| + fast and persistent modes | 0.996 |

### 5.4 Internal-state drift and internal-state noise

In the closed loop, the internal state drifts away from its true trajectory.

| model | domain / interface error |
|---|---|
| v06 | 1.96% / 17.7% |
| v06 with the true internal state fed in | 1.01% / 9.1% |
| v08 | 1.0% / 8.5% |
| v09 = v08 + internal-state input noise 0.15 | **0.38% / 3.7%** |

### 5.5 The remaining error: a small, persistent bias amplified by a near-neutral loop

- The weakest closed-loop mode is a **rigid interface translation**, with loop gain ≈ 0.99–0.998. With
  prescribed displacements, nothing physical restores it.
- A persistent one-step bias b therefore builds up to ~b / (1 − g), i.e. 100–500× b. A bias of 0.01–0.3 µm,
  invisible in test metrics, becomes a 5–25 µm offset.
- **On implicit data the seed decides a lot:** v09 / v12 / v13 gave 3.7% / 6.9% / 17.1% interface error.
- **On explicit data the seeds agree much better:** e01 / e05 / e06 gave 4.4–6.7%.
- **Diagnostics that separate the causes:**
  - open loop (true rows and true internal state fed in);
  - free-running internal state with true rows;
  - coupled run with the true internal state fed in;
  - loop gain.

  The large coupled offsets came from a small open-loop bias (±0.3 µm) in the fast phase after the first
  plastic loading, around 0.4–0.55 ms, amplified by the loop.

### 5.6 Things tried and not kept

- **Robin / penalty springs instead of prescribed displacements.**
  - It is exact with the target `u + t/k`. The test of this exposed the traction-ordering bug in §6.
  - Useful stabilisation needs soft springs, which makes the force part of the target large.
  - The user rejected it: springs caused long-term problems in their explicit crash runs.
- **Laplacian / Dirichlet smoothness loss.**
  - The original Laplacian was unnormalised in 3D: about 19× the MSE.
  - A corrected cotangent Dirichlet form improves smoothness and one-step error, but made the coupled runs worse.
- **Patch-mean loss (weight 30, 4 k-means patches)**, penalising the patch-averaged error:
  - slightly narrower spread between seeds, no better on average;
  - costs one-step accuracy (1.0–1.4% vs 0.6–0.9% test error);
  - not worth keeping as a default.
- **Absolute position references or extra external inputs**: rejected because they do not generalise to crash
  models.
- **Ensembles of seeds**: rejected; the user does not want to train several models.

### 5.7 Adding the replaced region's configuration to the internal state

**Motivation.** The internal state was only the plastic strain, a monotone scalar. It records *how much* the
top half has yielded, not *where* it is, so an interface plus rows that drift together look like a valid state.

**Change.** Add 4 PCA modes of the top half's displacement field (explaining 97.8 / 1.9 / 0.19 / 0.09% of the
variance). Data tag `explicit_ut`, `--assemble-explicit --top-modes 4`.

**Results** (explicit 2000-step coupled run, random Δt):

| model | interface error | rms z-shift | z-shift at 0.5 ms | note |
|---|---|---|---|---|
| e01 / e05 / e06 (plastic only) | 4.4–6.7% | 3.4–5.1 µm | ±8–10 µm | |
| **u02 / u03** | **4.4–4.8%** | **2.7–3.1 µm** | **−1.8 / +2.9 µm** | after only 75–80 epochs |
| u01 | 13.5% | 10.2 µm | | its displacement memory runs away |

**Interface accuracy follows internal-state accuracy.** Next test: u04–u06 with internal-state noise 0.3.

### 5.8 Training data must cover the application range

- v09 / v12 / e06 (trained on 2000 steps = one period at 1000 Hz), run for 6000 steps: interface error
  3.7% → 18–20% → 31–39% per 2000-step block.
  - The internal state reaches 4× the largest training value.
  - The motion amplitude doubles.
- x01 (trained on 6000-step data, with the patch-mean term, stopped at epoch 124): 7.3% / 8.4% in blocks 2–3.
  So longer data fixes the extrapolation.
- **But x01 is poor in block 1** (27.6%, offset up to 37 µm), which is part of its training data.
  - Suspected cause: the plastic PCA and its min-max scaling are dominated by the late, large strains, which
    compresses the early state.
  - Not yet resolved.
- For crash: the training data must cover the full deformation and plastic range the SM will see.

### 5.9 Explicit vs implicit training data

- All the early data used Newmark β = 0.5, γ = 1 (numerically damped, first order). A switch to β = 0.25,
  γ = 0.5 exists but is off.
- The converged explicit solution differs from that implicit data by **2.7%** at the interface. That is about
  the size of the coupled error, so for explicit applications, generate the training data explicitly.

### 5.10 Shell tube results

**Element** (`fem_sim/shell_explicit.py`):
- flat linear triangles, Reissner–Mindlin, 6 DOFs per node (global u and θ);
- centroid transverse shear, drilling penalty;
- plane-stress J2 with Newton return mapping, 5 Gauss points through the thickness;
- lumped translational and rotary mass, central differences.

**Validated:**

| check | result |
|---|---|
| frame invariance | 1e-11 |
| uniaxial plasticity | exact |
| tube cantilever | within 6% of beam theory |
| critical Δt | 3.0e-7 s (membrane) |
| speed | 27 ms per plastic step |

**Limitation:** linear triangles are 11–19% too stiff in plate bending at h = 2.5 mm, converging under
refinement. MITC3 would fix this.

**Plastic distribution:** plastic strain concentrates at the driven end (13%); the top half sees only ~1e-3.

**Internal-state PCA.** One plastic PCA component fitted on one simulation captures only 57–85% of the shell
field at other frequencies, against 98–99% for the solid. The shell needs **2 components fitted on all training
simulations** (97%).

**Coupled behaviour:**

| | interface error |
|---|---|
| open loop | 1.7–2.0% |
| coupled | 15–24% |

- Loop gain 0.99. It is the same bias-amplification mechanism as §5.5, with a relatively larger bias (0.3% of
  the 64 µm motion) and in-plane slow modes.
- The patch-mean term halved the shell drift (s02).
- The shell has not yet been tested with displacement modes in the internal state.

## 6. Bugs and pitfalls found

- **Interface traction vs displacement node order (3D solid).**
  - `calculate_interface_tractions()` returns tractions in the node order of the traction space, which differs
    from `u[interface_dofs]` and the interface topology. Only 2 of 151 nodes coincide.
  - Use `TCTSimulation3D.traction_to_interface_order()`.
  - Force application itself is self-consistent: replaying the true forces is exact to 1e-15.
  - Every dataset pairing forces with displacements or the graph is affected, including the **forward 3D GNN**:
    its decoder convolution operates on scrambled outputs. It is not fixed in `create_predictor_3d.py` yet.
- **Dropout in diagnostics:** `SPILSNetGraph.load()` returns train mode. Call `model._model.eval()`.
- **Memory leak in long explicit runs:** the extractor stored ε and σ fields every step (1.2 MB/step, 13 GB
  per run). Now switchable with `store_field_history`.
- **Training resume:** optimizer param groups, scheduler state, next epoch and patience are now saved. Older
  checkpoints are resumed with `--resume-lr`.
- **Analysis:**
  - compare explicit runs against the lockstep explicit reference, plotted against time (not step index);
  - each PCA basis is dataset-specific;
  - container logs are in UTC.
- **Operations:** the Mac sleeps on battery and pauses Docker. Use `caffeinate -dims` and power. Run one
  Python process per solid FE simulation, because PETSc segfaults when creating several.

## 7. Recommendations for the OpenCourant implementation

1. **Coupling.** Prescribe interface displacements only; leave shell rotations free. Call the SM at fixed
   physical times Δ (its training step) and interpolate the prescribed motion between calls (§4).
2. **Inputs.** Displacements of node rows at fixed distances from the interface on the retained side. On a
   shell, follow the surface; use distances that land on node rows if possible. Never the interface itself,
   never velocities.
3. **Internal state.**
   - plastic-strain PCA of the replaced region (several components if the plastic pattern shifts between
     load cases);
   - plus a few PCA modes of the replaced region's displacement field;
   - fit each PCA on all training runs, over the full range the application will reach.
4. **Training data.** Generate it with the same explicit solver, resampled onto the surrogate grid with the
   rules of §4. Add denoising replays with perturbed interface motion, and internal-state noise. Cover the
   full deformation and plastic range.
5. **Diagnostics before trusting a model:**
   - open-loop error;
   - free-running internal state with true inputs;
   - coupled run with the true internal state;
   - loop gain (should be clearly below 1);
   - at least 2–3 seeds.

   For quick tests use `--patience 30`; the user fine-tunes the final version.
6. **Crash-like loading** (planned next): non-periodic loading such as constant-velocity crushing, drop-weight
   impact, random smooth velocity histories or oblique loading. This needs large-deformation shell kinematics
   (co-rotational) and self-contact, and an internal state that covers plastic strains of order 1.

## 8. Code map

**SPILS-Net `workspace/`**

| file | role |
|---|---|
| `fem_sim/structural_sims.py` | FE base: Newmark constants, `time_integration="explicit"`, `critical_time_step()` |
| `fem_sim/tct_sims_3d.py` | TCT setup, `traction_to_interface_order()` |
| `fem_sim/tct_displacements_3d.py` | Extraction / replay / row-surrogate application (subcycling `sm_dt`, Robin option), `TCTExtractRowsExplicit3D` (grid data incl. `u_top`) |
| `fem_sim/interface_rows.py` | Exact P1 row sampling below interface nodes |
| `fem_sim/time_grid.py` | `GridResampler` (streaming onto the surrogate grid) |
| `fem_sim/shell_explicit.py`, `fem_sim/shell_tct.py` | Explicit RM shell and the square-tube problem |
| `create_predictor_3d_inv.py` | Inverse datasets, `interface_perturbation()`, force-input models |
| `create_predictor_3d_rows.py` | Row-surrogate pipeline (see below) |
| `create_predictor_shell_rows.py` | Shell pipeline: `--generate`, `--assemble`, `--generate-perturbed`, `--train`, `--apply`, `--loop-gain`, `--true-internal` |
| `diag_loop_gain_3d_inv.py` | Loop gain of force-input models |
| `analyse_results_3d_explicit.ipynb` | Explicit coupled runs vs reference over time |

Main commands of `create_predictor_3d_rows.py`:
- `--generate-explicit`, `--assemble-explicit [--top-modes K]`;
- `--generate --explicit-dt … --perturb-scale …`;
- `--train`;
- `--apply [--explicit-dt --dt-jitter --true-internal]`;
- `--loop-gain`.

Key options: `--data-tag`, `--seed`, `--patience`, `--internal-noise`, `--internal-weight`, `--mean-weight`,
`--resume[-lr]`.

**spilsnet-torch `spilsnet/`:**
- multi-feature nodes, local skip, unbounded output;
- Dirichlet / cotangent smoothness and patch-mean losses;
- internal-state noise;
- proper resume.

**Data, models, results** are git-ignored (several GB) and regenerated by the scripts:

| folder pattern | contents |
|---|---|
| `training_data_3d_inv[_explicit, _explicit6k, _explicit_ut]` | solid datasets |
| `training_data_shell` | shell dataset |
| `surrogate_models_3d_inv`, `surrogate_models_shell` | trained models |
| `results_3d_inv`, `results_shell` | coupled-run results |

Model naming:

| name | meaning |
|---|---|
| v08–v13 | implicit data |
| e01–e06 | explicit data |
| u01–u06 | explicit data + displacement modes |
| x01 | 6000-step explicit data |
| s01–s02 | shell |

## 9. Open items

- u04–u06: does internal-state noise 0.3 stop the displacement memory from running away (u01)?
- x01's poor first period on 6000-step data: internal-state scaling, or under-training?
- Shell: add displacement modes to its internal state; retest the coupling.
- Forward 3D GNN: retrain with the tractions put into interface order.
- Shell element: MITC3 if bending accuracy matters; co-rotational kinematics and contact for crash loading.
- Newmark switch: deferred (implicit data); explicit data is preferred anyway.
