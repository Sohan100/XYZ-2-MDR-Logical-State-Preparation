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

This aligns with the project note in
`legacy/XYZ_2_Code_9_excerpt.txt` (generated from your PDF) and with the
hardware-oriented objective of characterizing encoded-state preparation
fidelity before/alongside QPU experiments.

## Background

The MDR protocol prepares a target logical state by:

1. preparing an ancilla and entangling it with stabilizer/logical checks
2. measuring syndrome outcomes
3. applying classically conditioned Pauli toggles to project into the
   desired logical eigenspace

In this repository, the protocol is classically simulated in Stim with
SPAM, 1-qubit, and 2-qubit Pauli-channel noise models.

## Project Goals

- one class per file under `src/xyz2_mdr/`
- orchestration-only entry scripts under `scripts/`
- a Slurm workflow under `slurm/`
- `pytest` tests under `tests/`
- systematic output folders under `data/`

## Layout

- `src/xyz2_mdr/xyz2_stabilizer_generator.py` -> `XYZ2StabilizerGenerator`
- `src/xyz2_mdr/xyz2_logical_generator.py` -> `XYZ2LogicalGenerator`
- `src/xyz2_mdr/robust_toggle_generator.py` -> `RobustToggleGenerator`
- `src/xyz2_mdr/mdr_table.py` -> `MDRTable`
- `src/xyz2_mdr/mdr_circuit.py` -> `MDRCircuit`
- `src/xyz2_mdr/mdr_simulation.py` -> `MDRSimulation` (round-by-round
  expectation simulation core)
- `src/xyz2_mdr/mdr_noise_sweep.py` -> `MdrNoiseSweep`
- `src/xyz2_mdr/workflows.py` -> helper functions to wire classes together

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

- `data/tables/mdr_table_d{d}.csv`
- `data/simulation_results/results_*_spec-<hash>.csv`
- `data/simulation_results/results_*_spec-<hash>.spec.json`

## Regenerate Threshold Plots From CSV

```bash
python scripts/plot_thresholds_from_csv.py \
  --distances 3 5 7 9 11 \
  --input-dir data/simulation_results \
  --output-dir data/plots
```

The plotting script supports both legacy naming and the new
spec-hash naming. For spec-hash files, it picks the newest matching CSV
for each `(noise_model, distance)` pair.

## Slurm Workflow

### 1) Submit No-SPAM Simulation

```bash
sbatch slurm/run_xyz2_parallel_no_spam.sh
```

### 2) Submit With-SPAM Simulation

```bash
sbatch slurm/run_xyz2_parallel_with_spam.sh
```

Both Slurm files are self-contained:
- create run config for the chosen `DISTANCE`/`NOISE_MODEL`
- launch one process per probability index in parallel
- merge partial CSV outputs at the end

### 3) Final outputs

- `XYZ2-experiment-data-slurm/<RUN_NAME>/partials/result_idx*.csv`
- `XYZ2-experiment-data-slurm/<RUN_NAME>/results_<noise_model>_d<distance>.csv`
- copied merged file into `data/simulation_results/`

## Tests

```bash
pytest
```

The suite includes class-focused tests and save/load smoke tests.
