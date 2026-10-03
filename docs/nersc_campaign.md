# Threshold campaign on NERSC Perlmutter

This runbook runs every decoder, number of rounds, distance and noise model of the FT-MDR threshold study
on Perlmutter CPU nodes, then fits every threshold and makes every figure. It is written so that a person
or Claude Code on a login node can follow it step by step.

## What runs

`scripts/campaign.py tasks` writes `data/campaign/tasks.jsonl`, one line per task:

| | values |
|---|---|
| noise models | `sd6`, `si1000`, `biased10`, `biased100`, `purez`, `em3`, `helios_p`, `h2_p`, `helios_p_noxt`, `h2_p_noxt` |
| decoders | `mwpm`, `corr_links`, `corr_gauge` (HCM), `seq_soft`, `seq_match`, `bm` (belief-matching), `tesseract`, `cfe`, `cfe0` |
| rounds r | 1, 2, 3, 4, 5, 6, 8, 10 and r = d |
| distances | 3, 5, ..., 21 |
| p | 14 values (10 for the slow decoders) on a geometric grid around the expected threshold |

Limits that come from the decoders, not from the campaign:

- Tesseract stops at d = 9: its beam search grows too fast beyond.
- CFE with the combination sweep (`cfe`, the decoder of the paper) stores one candidate string per information
  bit, so a process needs about 2 (33 S)^2 bytes with S = (r + 1) d^2: 0.5 GB at d = 9 with r = 9 and 80 GB at
  d = 21 with r = 21. The campaign keeps `cfe` up to 16 GB per process, that is d <= 13 for r = d and d = 21 for
  r <= 5. `cfe0` is the same decoder with OSD-0 candidates (no combination sweep). It runs every case up to
  d = 21, at about a minute per shot for d = 21, r = 21.

Each task stops at a target number of logical errors, a maximum number of shots or a time budget. Budgets grow
with the circuit size, and points far below the expected threshold get less time. Points whose budget exceeds
four hours are split into replicas with independent seeds. A task writes its counts to
`data/campaign/points_<chunk>.csv` every five minutes, so a job that hits its time limit loses at most five
minutes per task, and the next submission continues where it stopped.

`scripts/campaign.py tasks` prints an upper bound of the cost (every task using its whole budget): about
22,000 core-hours, or 175 node-hours at 128 cores per node, of which CFE-0 and CFE are about 75%. Most tasks
stop early on their error target, so expect roughly half of that. `--scale` multiplies every budget and
`--cfe-scale` the CFE budgets, if the allocation is tight.

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
