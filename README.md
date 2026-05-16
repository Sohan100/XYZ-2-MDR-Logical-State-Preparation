# $XYZ^2$ MDR Logical State Preparation

This repository implements and benchmarks a Measurement-Decoding-Recovery
(MDR) state-preparation workflow for the $XYZ^2$ hexagonal stabilizer code.
It is a research-code package refactor from legacy project artifacts.

## Motivation

The project is motivated by two linked goals:

- study the $XYZ^2$ code family `[[2d^2, 1, d]]` on a honeycomb lattice,
  with weight-2 XX links, weight-6 XYZXYZ plaquettes, and weight-3
  boundary checks
- evaluate whether mixed-Pauli logical structure gives stronger resilience
  under biased noise channels than under more unbiased channels

This repository is meant to keep the active MDR implementation, workflows,
and tests on `main`; historical extraction artifacts and generated run outputs
stay outside version control.

## Background

The MDR protocol prepares a target logical state by:

1. preparing an ancilla and entangling it with stabilizer/logical checks
2. measuring syndrome outcomes
3. applying classically conditioned Pauli toggles to project into the
   desired logical eigenspace

In this repository, the protocol is classically simulated in Stim with
SPAM, 1-qubit, and 2-qubit Pauli-channel noise models.

## Project Goals

- one class per file under the canonical packages in `src/`
- orchestration-only entry scripts under `scripts/`
- a Slurm workflow under `slurm/`
- `pytest` tests under `tests/`
- systematic output folders under `data/`, with generated outputs ignored by
  git

## Layout

- `src/xyz2/stabilizer_generator.py` -> `XYZ2StabilizerGenerator`
- `src/xyz2/logical_generator.py` -> `XYZ2LogicalGenerator`
- `src/mdr/robust_toggle_generator.py` -> `RobustToggleGenerator`
- `src/mdr/mdr_table.py` -> `MDRTable`
- `src/mdr/mdr_circuit.py` -> `MDRCircuit`
- `src/mdr/mdr_simulation.py` -> `MDRSimulation` (round-by-round
  expectation simulation core)
- `src/mdr/mdr_noise_sweep.py` -> `MdrNoiseSweep`
- `src/mdr/workflows.py` -> helper functions to wire classes together

## Install

```bash
python -m pip install -e .[dev]
```

## Data Saving and Caching (Spec-Based)

Simulation outputs are keyed by an exact parameter specification, including:

- distance
- noise model and parameter names
- probability list
- rounds
- shots
- replicates
- SPAM probability

Each run writes:

- CSV: `data/simulation_results/results_<...>_spec-<hash>.csv`
- sidecar spec: same path with `.spec.json`

Behavior:

- if an exact same spec already exists, the code loads cached results and
  reports that the simulation already exists
- if any parameter differs, a new spec hash is produced and a new simulation
  is run
- if you want to re-run an existing exact spec anyway, pass `--force-rerun`

## Run Full Distance Sweeps (Local)

This runs (or loads cached) sweeps:

```bash
python scripts/run_distance_sweeps.py \
  --distances 3 5 7 9 11 \
  --noise-models z_type pure_z unbiased
```

Force recomputation of exact-matching specs:

```bash
python scripts/run_distance_sweeps.py \
  --distances 3 5 7 9 11 \
  --noise-models z_type pure_z unbiased \
  --force-rerun
```

Outputs:

- `data/tables/<code_family>/mdr_table_<code_family>_d{d}.csv`
- `data/simulation_results/<code_family>/results_*_spec-<hash>.csv`
- `data/simulation_results/<code_family>/results_*_spec-<hash>.spec.json`

## Regenerate MDR Noise-Sweep Plots From CSV

```bash
python scripts/plot_thresholds_from_csv.py \
  --distances 3 5 7 9 11 \
  --input-dir data/simulation_results \
  --output-dir data/plots
```

Filter by SPAM setting:

```bash
python scripts/plot_thresholds_from_csv.py \
  --distances 3 5 7 9 11 \
  --input-dir data/simulation_results \
  --output-dir data/plots \
  --p-spam 1.339e-3
```

The plotting script supports both legacy naming and spec-hash naming.
When `--p-spam` is set, it resolves CSVs by reading the `.spec.json`
sidecars and selecting the newest match for each
`(noise_model, distance, p_spam)`.

By default, regenerated MDR noise-sweep plots are written under
`data/plots/<code_family>/thresholds/`.

## Slurm Workflow

### 1) Submit No-SPAM Simulation

```bash
sbatch slurm/xyz2/run_parallel_no_spam.sh
```

### 2) Submit With-SPAM Simulation

```bash
sbatch slurm/xyz2/run_parallel_with_spam.sh
```

Surface-code sweeps use the parallel entry points under `slurm/surface/`:

```bash
sbatch slurm/surface/run_parallel_no_spam.sh
sbatch slurm/surface/run_parallel_with_spam.sh
```

The family folders under `slurm/` are self-contained:
- create one run config per `(noise_model, distance)` pair in the default
  sweep `z_type`, `pure_z`, `unbiased` x `3 5 7 9 11`
- launch one process per probability index in parallel for each pair
- merge partial CSV outputs after each pair completes
- copy canonical spec-keyed results into `data/simulation_results/`
- copy tables into `data/tables/`

### 3) Final outputs

These paths are generated locally or on the cluster and are intentionally not
tracked on `main`:

- `XYZ2-experiment-data-slurm/<RUN_NAME>/partials/result_idx*.csv`
- `XYZ2-experiment-data-slurm/<RUN_NAME>/results_<noise_model>_d<distance>.csv`
- `data/simulation_results/<code_family>/results_<code_family>_<noise_model>_d<distance>_pspam..._spec-<hash>.csv`
- `data/simulation_results/<code_family>/results_<...>.spec.json`
- `data/tables/<code_family>/mdr_table_<code_family>_d<distance>.csv`

## Tests

```bash
pytest
```

The suite includes class-focused tests and save/load smoke tests.
