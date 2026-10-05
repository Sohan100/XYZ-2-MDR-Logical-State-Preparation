# Threshold campaign on NERSC Perlmutter

This runbook runs every decoder, number of rounds, distance and noise model of the FT-MDR threshold study
on Perlmutter CPU nodes, then fits every threshold and makes every figure. It is written so that a person
or Claude Code on a login node can follow it step by step.

## What runs

`scripts/campaign.py tasks` writes `data/campaign/tasks.jsonl`, one line per task:

| | values |
|---|---|
| noise models | `sd6`, `si1000`, `biased10`, `biased100`, `purez`, `em3`, `helios_p`, `h2_p`, `helios_p_noxt`, `h2_p_noxt` |
| decoders | `mwpm`, `corr_links`, `corr_gauge` (HCM), `seq_soft`, `seq_match`, `bm` (belief-matching), `tesseract`, `cfe`, `cfe_tn` |
| rounds r | 1, 2, 3, 4, 5, 6, 8, 10 and r = d |
| distances | 3, 5, ..., 21 |
| p | 14 values (10 for the slow decoders) on a geometric grid around the expected threshold |

Limits that come from the decoders, not from the campaign:

- Tesseract stops at d = 9: its beam search grows too fast beyond.
- `cfe` is the CFE decoder of the paper. Its OSD-CS step runs through `src/mdr/ft/fast_osd.py`, which gives
  the same decoding as ldpc's `BpOsdDecoder(osd_method="osd_cs", osd_order=10)` (identical outputs in
  `tests/test_tn_decoder.py`) with memory and time that grow slowly, so CFE reaches d = 21 for every r
  (at d = 21 with r = 21 about 10 GB, most of it to build the decoder, and from 15 s per shot at
  the lowest p to 5 min at the highest; memory measured in `docs/data/campaign/memory_probe.csv`).
- `cfe_tn` is CFE whose decision is replaced, shot by shot, by the maximum-likelihood decision of a
  tensor-network contraction (`src/mdr/ft/tn_decoder.py`) whenever the bond dimension converges
  (chi doubled from 32 up to 256). The bond dimension needed grows quickly with the number of rounds:
  r = 1 converges with chi = 32 up to d = 21 (about 7 s per shot at d = 21), r = 2 with chi = 64 at d = 3,
  while r = 3 at d = 3 is not converged at chi = 32 and takes minutes per shot at chi = 256. The campaign
  makes `cfe_tn` tasks for r = 1 at every d and r = 2 up to d = 5 (`--tn-dmax 1=21 2=5`); elsewhere the
  best decoder is `cfe`. `TwoLevelDecoder.tn_used` counts the shots decided by the network.

Each task stops at a target number of logical errors, a maximum number of shots or a time budget. Budgets grow
with the circuit size, and points far below the expected threshold get less time. Points whose budget exceeds
four hours are split into replicas with independent seeds. A task writes its counts to
`data/campaign/points_<chunk>.csv` every five minutes, so a job that hits its time limit loses at most five
minutes per task, and the next submission continues where it stopped.

`scripts/campaign.py tasks` prints an upper bound of the cost (every task using its whole budget), about
20,000 core-hours (160 node-hours at 128 cores per node) for the whole list. Most tasks stop early on their
error target, so expect roughly half of that. `--scale` multiplies every budget and `--cfe-scale` the CFE
budgets, if the allocation is tight.

### Second stage (fast CFE and CFE + TN)

A campaign started before the fast CFE and the tensor network were added (commit 5e3ea41) ran `cfe` only up
to 16 GB per process (or with ldpc's slow OSD-CS) and had no `cfe_tn`. Its finished tasks stay valid: the
fast OSD-CS gives the same decoding. The second stage adds the rest without repeating any task:

```bash
python scripts/campaign.py tasks --out data/campaign/tasks2.jsonl --decoders cfe cfe_tn \
    --exclude data/campaign/tasks.jsonl
bash slurm/ft_mdr/submit_campaign.sh -A <project> -n 24 -T data/campaign/tasks2.jsonl -P points_s2
```

Resubmitting the first stage after `git pull` runs its remaining `cfe` tasks with the fast OSD-CS.

## Steps

All commands run from the repository root on a Perlmutter login node.

1. Environment, once (about five minutes):

   ```bash
   bash slurm/ft_mdr/setup_env.sh          # makes .venv with the package, ldpc, tesseract-decoder, pytest
   source slurm/ft_mdr/env.sh
   python -m pytest -q tests/test_campaign.py tests/test_coset_decoder.py tests/test_two_level_decoder.py
   ```

2. Smoke test on the login node (two minutes), which must finish without `failed` rows:

   ```bash
   mkdir -p /tmp/$USER/smoke
   python scripts/campaign.py tasks --out /tmp/$USER/smoke/tasks.jsonl --noises sd6 em3 helios_p \
       --rounds 1 d --distances 3 5
   python scripts/campaign.py run /tmp/$USER/smoke/tasks.jsonl /tmp/$USER/smoke/points_0.csv \
       --nchunks 40 --workers 16 --max-seconds 10
   grep failed /tmp/$USER/smoke/points_0.csv || echo "no failures"
   ```

3. Project and nodes. `sacctmgr -nP show assoc user=$USER format=account` lists the projects; use the one
   the user names (or the default). `iris` or `sshare` shows the balance. With N nodes the campaign takes
   about 90 / N hours of wall time plus queue time (N = 32 gives three to six hours).

4. Submit (the task list is written on the first call):

   ```bash
   bash slurm/ft_mdr/submit_campaign.sh -A <project> -n 32 -t 12:00:00
   ```

   This submits an array of N single-node jobs (`slurm/ft_mdr/run_campaign.sh`) and a job
   (`slurm/ft_mdr/analyze_campaign.sh`) that starts after all of them end. Logs go to `logs/`.

   Fastest turnaround: the wall time cannot drop below the longest task (`--rep-hours`, 4 h by default),
   so split long points into one-hour replicas before the first submission and use more nodes with a
   short time limit, which also backfills sooner (about one hour of run time at 384 nodes):

   ```bash
   python scripts/campaign.py tasks --out data/campaign/tasks.jsonl --rep-hours 1
   MEM_GB=396 bash slurm/ft_mdr/submit_campaign.sh -A <project> -n 384 -t 02:00:00
   ```

   `MEM_GB` caps the summed memory estimates of the running tasks of a node (default 85% of it; Slurm allows
   a job 476 of the 503 GiB). The estimates cover 1.3 times the peaks measured on Perlmutter for the largest
   task of every decoder and noise model (`docs/data/campaign/memory_probe.csv`).

5. Watch progress:

   ```bash
   squeue --me
   python scripts/campaign.py status data/campaign/tasks.jsonl "data/campaign/points_*.csv"
   ```

   If array elements end at the time limit with tasks left, submit again with the same `-n`
   (the script keeps the number of chunks in `data/campaign/nchunks`). Rows with `failed` in the `note`
   column name the error; a failed task is not retried unless its rows are removed.

6. After the analysis job (or by hand with `bash slurm/ft_mdr/analyze_campaign.sh` on a login node,
   about 15 minutes):

   - `docs/data/campaign/points.csv`: every point, counts summed over tasks and replicas
   - `docs/data/campaign/thresholds.csv`, `crossings.csv`, `thresholds.md`, `status.txt`
   - `paper/figures/campaign/`: `thr_<noise>_<decoder>.pdf` (every r), `rounds_<noise>.pdf`,
     `crossings_<noise>.pdf`, `fig_thr_<decoder>.pdf` (paper style, r = d, d = 3 to 21),
     `fig_thr_hw_<decoder>.pdf` (Helios and H2), `summary.pdf`, `tab_campaign_rd.tex`, `tab_campaign_r1.tex`

7. Commit the results (not `data/campaign/`, which `.gitignore` keeps out of git) and push:

   ```bash
   git add docs/data/campaign paper/figures/campaign
   git commit -m "Threshold campaign: every decoder, rounds, d = 3 to 21, every noise model"
   git push
   ```

`data/campaign/points_*.csv` hold the raw counts per task. They are not needed for the figures once
`docs/data/campaign/points.csv` is written, but keep them on `$SCRATCH` or `$CFS` until the paper is done.

## Large runs: a work pool on 128-node jobs

The array of single-node jobs works while the machine has room. A full campaign is different: every decoder,
r = 1 to 21 and r = d, d up to 21 and 4x budgets is about 1.5 million core-hours. When the queue is full,
the array barely moves:

- Perlmutter lets only two pending jobs per user and QOS gain age priority (`MaxJobsAccruePU = 2`).
- A job gets a node reservation only from priority 69121 on (`bf_min_prio_reserve`), about a day of age
  above the 67679 of `regular` and `preempt`.

So an array element starts only when a backfill hole fits it, two at a time. In October 2026 that was
about 4 node-hours per hour.

A large job ages just like a small one. The largest `preempt` job (128 nodes, 2 days) therefore gets 128
nodes as soon as it holds one of the two slots, and such jobs started within 3 to 21 hours that month.
`regular` jobs cannot preempt `preempt` jobs (`sacctmgr show qos format=name,preempt`), so they run their
full time.

The pool spreads the tasks over the nodes of such jobs:

```bash
python scripts/campaign.py pool data/campaign/pool1.jsonl --first d --tasks data/campaign/tasks*.jsonl
sbatch -A <project> --export=ALL,POOL=data/campaign/pool1.jsonl slurm/ft_mdr/run_pool.sh   # twice
python scripts/campaign.py pool-status data/campaign/pool1.jsonl
```

How the pool works:

- `pool` writes every unfinished task with its counts so far, longest budget first, in units of 128 tasks.
  `--first d` puts every r = d task ahead of the rest, so the r = d thresholds finish in the first hours.
- Every node runs `campaign.py pool-run`, which works like `run` on one node. It claims a unit whenever
  all tasks claimed so far have started, by creating `pool1.d/claims/<unit>`, and refreshes the claim
  every minute.
- The counts of a unit go to `points_pool1_u<unit>.csv`.
- A claim that has not been refreshed for 15 minutes belongs to a node that stopped. The next node takes
  the unit over and continues from that file.
- Finished units move to `pool1.d/done/`. A node stops when no unit is free and no other node of its job
  holds one.

Submit jobs before the pool exists, so they start aging; a job that starts first waits up to three hours
for the pool. Keep the pool's name: claims and counts refer to it. For a later pool, make
`pool2.jsonl` from the same task files once every job of the first one has ended; it reads the counts
of `points_pool1_*` like any other.

`status`, `merge` and the analysis read the pool's files with every other `points_*.csv`.
