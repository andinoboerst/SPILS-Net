# SPILS-Net — Results Reproduction Repository

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

> **Companion repository** to the paper:
>
> *"SPILS-Net: A Physics-Informed LSTM Surrogate for Interface Traction Prediction in Domain-Decomposition Simulations"*
> — *[Authors]*, *[Journal]*, [Year]. DOI: `[to be added upon publication]`

This repository provides all the code needed to **reproduce the numerical results** presented in the paper. It is designed to be fully reproducible — either inside the provided Docker Dev Container (recommended for FEM simulations) or in a local Python virtual environment (for training and evaluating the neural-network predictors only).

The SPILS-Net neural-network architecture itself lives in a **separate, dedicated repository**: [spilsnet-torch](https://github.com/andinoboerst/spilsnet-torch).

---

## Repository Structure

```
SPILS-Net/
├── workspace/
│   ├── create_predictor.py      # Main entry point: generate data, train, and evaluate
│   ├── fem_sim/                 # FEniCSx-based FEM simulation code
│   │   ├── structural_sims.py   # Base FEM formulations (linear/nonlinear structural dynamics)
│   │   ├── tct_sims.py          # TCT (Traction Coupling Test) problem setup
│   │   ├── tct_tractions.py     # Traction extraction and application classes
│   │   ├── tct_tractions_comp.py
│   │   ├── plotting.py          # Mesh animation utilities
│   │   └── progress_bar.py
│   ├── nn_predictors/           # Neural network predictor wrappers
│   │   ├── lstm.py              # LSTM baseline
│   │   └── misc.py              # Data utilities (scaling, splitting, datasets)
│   ├── mesh_files/              # Auto-generated FEM mesh files (XDMF)
│   ├── training_data/           # Pre-generated training datasets (.npz) — see Data section
│   ├── surrogate_models/        # Saved model checkpoints (git-ignored)
│   └── results/                 # Simulation output files (git-ignored)
├── tests/
│   └── smoke_test.py            # Lightweight tests (no FEM required)
├── logs/                        # Training logs
├── .devcontainer/               # Docker Dev Container configuration
│   ├── Dockerfile
│   └── devcontainer.json
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment for local ML testing
├── CITATION.cff                 # Machine-readable citation
└── LICENSE
```

---

## Requirements

The code has **two distinct dependency layers**:

| Layer | Dependencies | Installation |
|---|---|---|
| **FEM simulations** | FEniCSx (dolfinx), mpi4py, petsc4py, UFL, pygmsh | Docker (see below) |
| **ML predictors** | PyTorch, scikit-learn, numpy, lion-pytorch | `requirements.txt` / `environment.yml` |

> **`spilsnet-torch`**: The SPILS-Net architecture (`nn_predictors/spils_net.py`) requires the `spilsnet-torch` package. Install it separately following the instructions in the [spilsnet-torch repository](https://github.com/andinoboerst/spilsnet-torch).

---

## Getting Started

### Option 1: Dev Container (Recommended — Full Reproduction)

This provides the complete environment including FEniCSx for running FEM simulations.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

1. Clone the repository:
   ```bash
   git clone https://github.com/andinoboerst/SPILS-Net.git
   cd SPILS-Net
   ```

2. Open in VS Code and click **"Reopen in Container"** when prompted (or use `Ctrl+Shift+P` → `Dev Containers: Reopen in Container`).

3. Install `spilsnet-torch` inside the container:
   ```bash
   pip install git+https://github.com/andinoboerst/spilsnet-torch.git
   ```

4. The working directory inside the container is `/workspace` (mapped to the repository root).

### Option 2: Docker Compose (Standalone — No VS Code required)

If you don't use VS Code, you can run the entire environment using Docker Compose:

1. **Build the image**:
   ```bash
   make build   # or: docker compose build
   ```

2. **Run a shell in the container**:
   ```bash
   make run     # or: docker compose run --rm spils-net /bin/bash
   ```

3. **Run specific tasks from your host**:
   ```bash
   make train-lstm   # Builds and runs training
   make simulate     # Runs the evaluation simulation
   ```

### Option 3: Local Virtual Environment (ML predictors only)

This option runs the neural-network training and evaluation **without** FEM simulations. You will need the pre-generated training data (see [Data Availability](#data-availability)).

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate spils-net
pip install git+https://github.com/andinoboerst/spilsnet-torch.git
```

**Using pip + venv:**
```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install git+https://github.com/andinoboerst/spilsnet-torch.git
```

---

## Usage

All commands should be run from the **repository root** (or `/workspace` inside the container).

> **Note:** In the Dev Container, the working directory is `/workspace`, so paths like `workspace/training_data/...` resolve correctly.

### 1. Generate Training Data (requires FEM / Dev Container)

Edit the configuration at the top of `workspace/create_predictor.py`:
```python
constitutive_law = "plastic"        # "elastic" or "plastic"
problem_configuration = "benchmark" # "benchmark" or "scaled"
training_set_exists = False         # Set to False to generate
```
Then run:
```bash
cd workspace
python create_predictor.py
```
This will populate `workspace/training_data/` with `.npz` files.

### 2. Train a predictor

Set `training_set_exists = True` and choose the method:
```python
predictor_method = "spils_net"  # or "lstm"
predictor_model_exists = False
```
Then run `python workspace/create_predictor.py` again.

Training logs and checkpoints are saved to `workspace/surrogate_models/`.

### 3. Evaluate a predictor

Set `predictor_model_exists = True` to skip training and only run the FEM evaluation:
```bash
python workspace/create_predictor.py
```
Results are saved to `workspace/results/`.

---

## Data Availability

The pre-generated training datasets (`workspace/training_data/*.npz`, ~475 MB total) are too large to store in Git. They are available at:

> **[Zenodo / Figshare — DOI: `to be added`]**

Download and place the files in `workspace/training_data/` before running training.

Alternatively, you can generate the data from scratch using the Dev Container (see Step 1 above).

---

## Reproducing Paper Results

The paper presents two experimental configurations:

| Experiment | `constitutive_law` | `problem_configuration` | `predictor_method` |
|---|---|---|---|
| Benchmark – LSTM | `plastic` | `benchmark` | `lstm` |
| Benchmark – SPILS-Net | `plastic` | `benchmark` | `spils_net` |
| Scaled – LSTM | `plastic` | `scaled` | `lstm` |
| Scaled – SPILS-Net | `plastic` | `scaled` | `spils_net` |

Training logs for all four experiments are archived in `logs/`.

---

## Testing

A lightweight smoke test (no FEM dependencies required) can be run locally:

```bash
conda activate spils-net
python -m pytest tests/smoke_test.py -v
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{spilsnet2026,
  title   = {SPILS-Net: A Physics-Informed LSTM Surrogate for Interface Traction Prediction in Domain-Decomposition Simulations},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI]}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 — see the [LICENSE](LICENSE) file for details.