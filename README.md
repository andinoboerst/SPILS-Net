# SPILS-Net — Results Reproduction Repository

[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1234/zenodo.1234567-blue.svg)](https://doi.org/10.5281/zenodo.21236798)

> **Companion repository** to the paper:
>
> *Accelerating Transient Structural Dynamics via SPILS-Net, a Physics-Derived Latent Space Subdomain Surrogate"*
> — *Andino Börst et al.*, *Computer Methods in Applied Mechanics and Engineering*, 2026. DOI: `[to be added upon publication]`

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
│   ├── surrogate_models/        # Saved model checkpoints
│   └── results/                 # Simulation output files
├── tests/
│   └── smoke_test.py            # Lightweight tests (no FEM required)
├── logs/                        # Training logs
├── .devcontainer/               # Docker Dev Container configuration
│   ├── Dockerfile
│   └── devcontainer.json
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment for local ML testing
├── docker-compose.yml           # Docker Compose configuration
├── CITATION.cff                 # Machine-readable citation
├── Makefile                     # Build and run automation
└── LICENSE
```

---

## Requirements

The code has **two distinct dependency layers**:

| Layer | Dependencies | Installation |
|---|---|---|
| **FEM simulations** | FEniCSx (dolfinx), mpi4py, petsc4py, UFL, pygmsh | Docker (see below) |
| **ML predictors** | PyTorch, scikit-learn, numpy, lion-pytorch | `requirements.txt` / `environment.yml` |

> **`spilsnet-torch`**: The SPILS-Net architecture requires the `spilsnet-torch` package. By default, the official PyPI release (`spilsnet-torch==1.0.1`) is installed. To switch to using a local sibling directory for development, see [Local Development vs. PyPI Release](#local-development-vs-pypi-release) below.

---

## Getting Started

### Option 1: Docker Compose (Recommended — Full Reproduction)

This provides the complete environment including FEniCSx for running FEM simulations and is the recommended approach for most users.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/).

1. Clone the repository:
   ```bash
   git clone https://github.com/andinoboerst/SPILS-Net.git
   cd SPILS-Net
   ```

2. **Build the image**:
   ```bash
   make build   # or: docker compose build
   ```

3. **Run a shell in the container**:
   ```bash
   make run     # or: docker compose run --rm spils-net /bin/bash
   ```

4. **Run specific tasks from your host**:
   ```bash
   make train-lstm   # Builds and runs training
   make simulate     # Runs the evaluation simulation
   ```

### Option 2: Local Virtual Environment (ML predictors only)

This option runs the neural-network training and evaluation **without** FEM simulations. You will need the pre-generated training data (see [Data Availability](#data-availability)).

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate spils-net
```

**Using pip + venv:**
```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Option 3: Dev Container (VS Code specific)

If you use VS Code and prefer an integrated development environment, you can use the Dev Container.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

1. Clone the repository:
   ```bash
   git clone https://github.com/andinoboerst/SPILS-Net.git
   cd SPILS-Net
   ```

2. Open in VS Code and click **"Reopen in Container"** when prompted (or use `Ctrl+Shift+P` → `Dev Containers: Reopen in Container`).

3. The working directory inside the container is `/workspace` (mapped to the repository root).

### Local Development vs. PyPI Release

By default, this repository uses the PyPI-released `spilsnet-torch==1.0.1` package. However, if you are developing changes locally inside the sibling `spilsnet-torch` repository (`../spilsnet-torch`), you can configure the environment to use your local implementation instead:

#### Option A: Inside Docker (Docker Compose / Dev Container)
1. In `docker-compose.yml`, uncomment the developer `PYTHONPATH` line:
   ```yaml
   # - PYTHONPATH=/workspace:/spilsnet-torch:/usr/local/lib:/usr/local/dolfinx-real/lib/python3.12/dist-packages
   ```
   This will prioritize the `/spilsnet-torch` volume mount containing your local files.
2. Alternatively, inside the container, run:
   ```bash
   make dev-local
   ```
   This will install the local directory in editable mode (`pip install -e /spilsnet-torch`).
3. To switch back to the official PyPI release, run:
   ```bash
   make dev-pypi
   ```

#### Option B: Local Virtual Environment (Conda / venv)
1. Run the Makefile target:
   ```bash
   make dev-local
   ```
   This will automatically detect the local environment and run `pip install -e ../spilsnet-torch`.
2. To revert to the PyPI package, run:
   ```bash
   make dev-pypi
   ```

---

## Makefile Commands

The repository includes a `Makefile` to simplify common tasks. All commands can be run from the host machine (they use Docker internally) or inside the container.

| Command | Description |
|---------|-------------|
| `make build` | Build the Docker image for the reproduction environment. |
| `make run` | Open an interactive shell inside the Docker container. |
| `make train` | Train the SPILS-Net model using Docker. Accepts `ARGS` for additional arguments. |
| `make train-lstm` | Train the LSTM baseline model using Docker. Accepts `ARGS` for additional arguments. |
| `make train-locally` | Train SPILS-Net locally (requires local Python environment). Accepts `ARGS` for additional arguments. |
| `make train-lstm-locally` | Train LSTM locally. Accepts `ARGS` for additional arguments. |
| `make apply` | Apply a trained SPILS-Net model to new data using Docker. Accepts `ARGS` for additional arguments. |
| `make apply-lstm` | Apply a trained LSTM model using Docker. Accepts `ARGS` for additional arguments. |
| `make simulate` | Run the full FEM simulation using Docker. Accepts `ARGS` for additional arguments. |
| `make smoke-test` | Run lightweight tests (no FEM required) using Docker. |
| `make clean` | Remove temporary files, caches, and virtual environments. |

**Using ARGS:** Several commands accept an `ARGS` variable to pass additional command-line arguments to the underlying `create_predictor.py` script. For example:
- `make train ARGS="--data-version scaled --law plastic --version 02"`
- `make apply ARGS="--freq 1000"`

Run `python workspace/create_predictor.py --help` to see all available options.

---

## Usage

The recommended way to run the code is through the [Makefile commands](#makefile-commands) described above, which handle the Docker setup and argument passing automatically.

For advanced users or custom configurations, you can run the main script directly:

```bash
cd workspace
python create_predictor.py --help  # See available options
```

Common usage patterns:
- Generate training data: `python create_predictor.py --generate --data-version benchmark --law plastic`
- Train a model: `python create_predictor.py --train --method spils_net --data-version benchmark`
- Evaluate a model: `python create_predictor.py --apply --method spils_net --data-version benchmark`

All commands should be run from the **repository root** (or `/workspace` inside the container).

---

## Data Availability

The pre-generated training datasets are included in the repository at `workspace/training_data/*.npz`. These datasets contain the training data for both benchmark and scaled configurations.

If you need to regenerate the data from scratch (e.g., for different parameters), you can do so using the Dev Container or Docker Compose setup with FEM capabilities.

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

If you use this code in your research, please cite the associated paper and this repository.

### Paper Citation
```bibtex
@article{boerst2026spilsnet,
  title={Accelerating Transient Structural Dynamics via SPILS-Net, a Physics-Derived Latent Space Subdomain Surrogate},
  author={Börst, Andino and Díez, Pedro and Zlotnik, Sergio and Cavaliere, Fabiola and Curtosi, Gabriel and Larráyoz, Xabier},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2026},
  doi={[DOI — to be added upon publication]}
}
```

### Software Citation

**Results-reproduction repository** (this repository):
```bibtex
@software{boerst_spils_net_2026,
  author={Börst, Andino},
  title={SPILS-Net — Results Reproduction Repository},
  year={2026},
  publisher={Zenodo},
  url={https://doi.org/10.5281/zenodo.21236797},
  doi={10.5281/zenodo.21236797},
  version={1.0.1}
}
```

**Neural-network architecture package** ([spilsnet-torch](https://github.com/andinoboerst/spilsnet-torch), available on [PyPI](https://pypi.org/project/spilsnet-torch/)):
```bibtex
@software{boerst_spilsnet_torch_2026,
  author={Börst, Andino},
  title={spilsnet-torch: PyTorch Implementation of SPILS-Net},
  year={2026},
  publisher={Zenodo},
  url={https://doi.org/10.5281/zenodo.21236780},
  doi={10.5281/zenodo.21236780},
  version={1.0.1}
}
```

For machine-readable citation metadata, see [`CITATION.cff`](CITATION.cff).

---

## License

This project is licensed under the GNU Affero General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
