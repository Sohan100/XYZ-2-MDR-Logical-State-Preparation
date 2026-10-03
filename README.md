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

The named `si1000` noise model is the Stim/q=2 specialization of the
generalized SI1000 qudit model. It sweeps one physical parameter `p`, applies
one-qubit depolarizing noise at `p/10`, two-qubit depolarizing noise at `p`,
reset shift errors at `2p`, measurement depolarizing noise at `p`, and
measurement-result flips at `5p`. Because the measurement-result rate is
`5p`, SI1000 sweeps must use `p <= 0.2`.

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

Run the SI1000 circuit-level model with an explicit valid probability grid:

```bash
python scripts/run_distance_sweeps.py \
  --distances 3 5 7 \
  --noise-models si1000 \
  --probabilities 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 2e-1
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
- submit one Slurm array task per `(noise_model, distance, probability)` in
  the default sweep `z_type`, `pure_z`, `unbiased` x `3 5 7 9 11` x 29
  probabilities
- create one shared run config per `(noise_model, distance)` pair under a
  lock
- merge partial CSV outputs automatically once the final probability for a
  pair completes
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

## Fault-tolerant MDR (frame initialization + space-time matching)

`src/mdr/ft/` contains a fault-tolerant version of the MDR preparation of
$|\bar{+}\rangle$. The older `full_mdr` and `link_logical_plus` paths have
circuit-level fault distance 1, which is why their logical fidelity saturates
after one round and gets worse with distance. The new path reaches fault
distance $d+1$, which we verified exactly for $d = 3, 5$ and for $d = 7$ with one
round:

1. prepare data in the block-CSS product frame (even blocks $|+\rangle|+\rangle$,
   odd blocks $|0\rangle|{+i}\rangle$), which makes $\bar{X}$ and $d^2$
   stabilizers deterministic,
2. measure all $2d^2-1$ checks with a depth-6 interleaved schedule,
3. decode the whole detector history with PyMatching on the
   frame-deterministic detectors,
4. recover with a Pauli-frame update.

Under the one-parameter Helios model at its device point (`CircuitNoise.trapped_ion`,
two-qubit error rate $p=10^{-3}$) one round gives $p_L \approx 7.5\times10^{-4}$
at $d=3$, $9.0\times10^{-5}$ at $d=5$ and $2.2\times10^{-5}$ at $d=7$ with matching.
`docs/ft_mdr_protocol.md` has the details, figures and the hardware plan.

```bash
python scripts/report_ft_mdr_fault_distance.py --distances 3 5 7
python scripts/run_ft_mdr_sweep.py --noise helios --rounds 1 --distances 3 5 7 --values 5e-4 1e-3 2e-3
python scripts/plot_ft_mdr_results.py
python scripts/export_ft_mdr_qasm.py --distances 3 5 --rounds 1 3 --share-ancillas 4
pytest tests/test_ft_mdr.py
```

### Decoders and circuit-level thresholds

`mdr.ft.TwoLevelDecoder` adds decoders that use the gauge detectors of
`FTMDRCircuit(..., detectors="combined")` on top of matching on $S_0$.
The recommended decoder is hierarchical correlated matching, selected with
`mode="corr_split", lower="gauge"`. It costs about four times as much as plain
matching and raises the threshold under every noise model we tried.
`docs/ft_mdr_decoders.md` lists all decoders, the noise models
`CircuitNoise.uniform`, `si1000`, `biased`, `trapped_ion` (one-parameter
Helios and H2 models), `quantinuum` and `em3` (pair measurements), and the
threshold table. Threshold sweeps:

```bash
python scripts/run_decoder_threshold_sweep.py --noise sd6 \
    --decoders mwpm corr_gauge --distances 3 5 7 9 \
    --values 3e-3 4e-3 5e-3 6e-3 --out data/ft_mdr/thresholds.csv
```

The decoders need `pymatching>=2.3` for correlated matching and `scipy`.
Belief-matching and sequential BP also need `ldpc`, and the Tesseract
reference needs `tesseract-decoder`. Both come with
`pip install -e ".[decoders]"`. The scripts that make the fits, tables and
result figures of the paper are described in `paper/README.md`.

### Full threshold campaign (NERSC)

`scripts/campaign.py` runs every decoder (with CFE and its OSD-0 variant CFE-0), every number of
rounds (1 to 10 and r = d), every odd distance from 3 to 21 and every noise model, and
`paper/analysis/campaign_report.py` fits all thresholds and draws the figures. `docs/nersc_campaign.md`
is the runbook for Perlmutter:

```bash
bash slurm/ft_mdr/setup_env.sh                       # once
bash slurm/ft_mdr/submit_campaign.sh -A <project> -n 32
python scripts/campaign.py status data/campaign/tasks.jsonl "data/campaign/points_*.csv"
```
