# Makefile for SPILS-Net
# Provides easy commands for building and running the reproduction environment.

IMAGE_NAME = spils-net-repro
CONTAINER_NAME = spils-net-instance

.PHONY: build run train-lstm train-spilsnet simulate smoke-test test clean help

help:
	@echo "SPILS-Net Reproduction Makefile"
	@echo "Usage:"
	@echo "  make build          Build the Docker image"
	@echo "  make run            Open a shell inside the container"
	@echo "  make train          Train the SPILS-Net model (runs inside Docker)"
	@echo "  make train-lstm     Train the LSTM baseline (runs inside Docker)"
	@echo "  make simulate       Run the full FEM simulation (runs inside Docker)"
	@echo "  make smoke-test     Run lightweight ML tests (can be run locally or in Docker)"
	@echo "  make clean          Remove temporary files and caches"

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

time-predictor:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --time-predictor $(ARGS)"

time-fem:
	docker compose run --rm spils-net /bin/bash -c "cd workspace && python3 create_predictor.py --time-fem $(ARGS)"

smoke-test:
	docker compose run --rm spils-net /bin/bash -c "PYTHONPATH=/workspace pytest tests/smoke_test.py -v"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .venv
