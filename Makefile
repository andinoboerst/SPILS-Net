# Makefile for SPILS-Net
# Provides easy commands for building and running the reproduction environment.

IMAGE_NAME = spils-net-repro
CONTAINER_NAME = spils-net-instance

.PHONY: build run train-lstm train-spilsnet simulate smoke-test test clean help dev-local dev-pypi \
        simulate-3d verify-fem-3d generate-3d train-3d train-lstm-3d apply-3d apply-lstm-3d \
        verify-fem-3d-inv generate-inv-3d train-inv-3d train-gnn-inv-3d train-lstm-inv-3d apply-inv-3d apply-gnn-inv-3d apply-lstm-inv-3d \
        verify-fem-3d-disc generate-3d-disc train-3d-disc train-gnn-3d-disc train-lstm-3d-disc apply-3d-disc apply-gnn-3d-disc apply-lstm-3d-disc \
        verify-fem-3d-disc-inv generate-inv-3d-disc train-inv-3d-disc train-gnn-inv-3d-disc train-lstm-inv-3d-disc apply-inv-3d-disc apply-gnn-inv-3d-disc apply-lstm-inv-3d-disc

help:
	@echo "SPILS-Net Reproduction Makefile"
	@echo "Usage:"
	@echo "  make build                  Build the Docker image"
	@echo "  make run                    Open a shell inside the container"
	@echo "  make train                  Train the SPILS-Net model (runs inside Docker)"
	@echo "  make train-lstm             Train the LSTM baseline (runs inside Docker)"
	@echo "  make simulate               Run the full FEM simulation (runs inside Docker)"
	@echo "  make simulate-3d            Run the 3D FEM simulation + GIF export"
	@echo "  make verify-fem-3d          Extract tractions, re-apply (no ML), report error"
	@echo "  make verify-fem-3d-inv      Extract displacements, re-apply (no ML), report error"
	@echo "  make generate-3d            Generate 3D multi-frequency training dataset"
	@echo "  make train-3d               Train the 3D SPILS-Net model"
	@echo "  make train-lstm-3d          Train the 3D LSTM baseline"
	@echo "  make apply-3d               Evaluate 3D SPILS-Net against FEM"
	@echo "  make apply-lstm-3d          Evaluate 3D LSTM against FEM"
	@echo "  make generate-inv-3d        Generate 3D inverse training dataset (Forces -> Displacements)"
	@echo "  make train-gnn-inv-3d       Train 3D inverse SPILS-Net GNN model"
	@echo "  make train-lstm-inv-3d      Train 3D inverse LSTM baseline"
	@echo "  make apply-gnn-inv-3d       Evaluate 3D inverse SPILS-Net GNN against FEM"
	@echo "  make verify-fem-3d-disc     Verify discontinuous 3D forward FEM traction coupling"
	@echo "  make generate-3d-disc       Generate discontinuous 3D forward dataset (u -> F)"
	@echo "  make train-gnn-3d-disc      Train discontinuous 3D forward SPILS-Net GNN"
	@echo "  make apply-gnn-3d-disc      Evaluate discontinuous 3D forward GNN against FEM"
	@echo "  make verify-fem-3d-disc-inv Verify discontinuous 3D inverse FEM displacement coupling"
	@echo "  make generate-inv-3d-disc   Generate discontinuous 3D inverse dataset (F -> u)"
	@echo "  make train-gnn-inv-3d-disc  Train discontinuous 3D inverse SPILS-Net GNN"
	@echo "  make apply-gnn-inv-3d-disc  Evaluate discontinuous 3D inverse GNN against FEM"
	@echo "  make smoke-test             Run lightweight ML tests"
	@echo "  make dev-local              Install the local spilsnet-torch in editable mode for dev"
	@echo "  make dev-pypi               Install the PyPI-released spilsnet-torch (v1.0.1)"
	@echo "  make clean                  Remove temporary files and caches"

build:
	docker compose build

run:
	docker compose run --rm spils-net /bin/bash

train:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --train $(ARGS)"

train-lstm:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --method lstm --train $(ARGS)"

train-locally:
	cd workspace && python3 create_predictor.py --train $(ARGS)

train-lstm-locally:
	cd workspace && python3 create_predictor.py --method lstm --train $(ARGS)

apply:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --apply $(ARGS)"

apply-lstm:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --method lstm --apply $(ARGS)"

simulate:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --simulate $(ARGS)"

simulate-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 simulate_3d.py $(ARGS)"

verify-fem-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --verify-fem $(ARGS)"

generate-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --generate $(ARGS)"

train-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --train $(ARGS)"

train-gnn-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --method spils_net_gnn --train $(ARGS)"

train-gnn-pos-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --method spils_net_gnn_pos --train $(ARGS)"

train-lstm-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --method lstm --train $(ARGS)"

apply-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --apply $(ARGS)"

apply-gnn-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --method spils_net_gnn --apply $(ARGS)"

apply-gnn-pos-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --method spils_net_gnn_pos --apply $(ARGS)"

apply-lstm-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d.py --method lstm --apply $(ARGS)"

verify-fem-3d-inv:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --verify-fem $(ARGS)"

generate-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --generate $(ARGS)"

train-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --train $(ARGS)"

train-gnn-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --method spils_net_gnn --train $(ARGS)"

train-lstm-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --method lstm --train $(ARGS)"

apply-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --apply $(ARGS)"

apply-gnn-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --method spils_net_gnn --apply $(ARGS)"

apply-lstm-inv-3d:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_inv.py --method lstm --apply $(ARGS)"

verify-fem-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --verify-fem $(ARGS)"

generate-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --generate $(ARGS)"

train-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --train $(ARGS)"

train-gnn-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --method spils_net_gnn --train $(ARGS)"

train-lstm-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --method lstm --train $(ARGS)"

apply-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --apply $(ARGS)"

apply-gnn-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --method spils_net_gnn --apply $(ARGS)"

apply-lstm-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc.py --method lstm --apply $(ARGS)"

verify-fem-3d-disc-inv:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --verify-fem $(ARGS)"

generate-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --generate $(ARGS)"

train-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --train $(ARGS)"

train-gnn-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --method spils_net_gnn --train $(ARGS)"

train-lstm-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --method lstm --train $(ARGS)"

apply-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --apply $(ARGS)"

apply-gnn-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --method spils_net_gnn --apply $(ARGS)"

apply-lstm-inv-3d-disc:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor_3d_disc_inv.py --method lstm --apply $(ARGS)"


time-predictor:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --time-predictor $(ARGS)"

time-fem:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --time-fem $(ARGS)"

smoke-test:
	docker compose run --rm spils-net /bin/bash -c "PYTHONPATH=/workspace pytest tests/smoke_test.py -v"

# ================================================================== #
# Development & Mode Switching
# ================================================================== #

dev-local:
	@echo "Configuring environment to use the local spilsnet-torch codebase..."
	docker compose run --rm spils-net /bin/bash -c "if [ -d /spilsnet-torch ]; then pip install -e /spilsnet-torch; elif [ -d ../spilsnet-torch ]; then pip install -e ../spilsnet-torch; else echo 'Error: /spilsnet-torch not found.'; exit 1; fi"

dev-pypi:
	@echo "Configuring environment to use PyPI release spilsnet-torch==1.1.0..."
	docker compose run --rm spils-net /bin/bash -c "pip install --force-reinstall spilsnet-torch==1.1.0"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .venv
