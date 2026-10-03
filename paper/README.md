# Analysis scripts for the paper

These scripts turn the simulation data in `docs/data` into the fits, tables
and result figures of the paper. `paper/manuscript` holds the LaTeX source as
uploaded to Overleaf, with the rendered figures in `paper/manuscript/figures`
and the TikZ sources of the schematics in `paper/manuscript/figures/src`.

```bash
pip install -e ".[decoders]"
python paper/analysis/thresholds.py   # docs/data/thresholds/fits.csv
python paper/analysis/make_plots.py   # paper/build/figures/*.pdf
python paper/analysis/tables.py       # paper/build/tables/*.tex and *.md
python paper/analysis/timing.py       # paper/build/timing.csv
python paper/analysis/rounds_thresholds.py   # docs/data/rounds_threshold/thresholds.csv
python paper/analysis/rounds_plots.py        # threshold versus number of rounds
python paper/analysis/helios_plots.py        # Quantinuum figure, tab_quantinuum, tab_budget
python paper/analysis/cfe_plots.py           # CFE threshold comparison (fig_cfe_thr, tab_cfe_thr)
python paper/analysis/bm_thresholds.py       # belief-matching crossings (fig_bm, bm_crossings.csv)
python paper/analysis/docs_figures.py        # PNG previews in docs/figures
```

`make_plots.py` renders text with LaTeX, so it needs a TeX installation with
the cm-super fonts. `crossings.py` has the crossing-point estimate used for
belief-matching and the Quantinuum models: a weighted fit of $\ln p_L$ against
$\ln p$ per distance within $\pm15\%$ of the crossing, the intersection of the
two lines and a parametric bootstrap. `overlap_check.py` reports text that
touches other text, legends, lines or markers. Import it before drawing and
every `savefig` prints the overlaps of that figure:

```bash
cd paper/analysis && python -c "import overlap_check, runpy; runpy.run_path('bm_thresholds.py', run_name='__main__')"
```

## Data

| file | content |
|---|---|
| `docs/data/thresholds/sweeps_pooled.csv` | every simulated point of the SD6, SI1000, biased, pure $Z$ and EM3 models, shots and errors summed over runs |
| `docs/data/thresholds/fits.csv` | finite-size scaling fits and belief-matching crossings, one row per noise model and decoder |
| `docs/data/thresholds/bm_pooled.csv` | belief-matching points, dense around the crossings |
| `docs/data/thresholds/bm_crossings.csv` | belief-matching crossings of $d=5/7$ and $d=3/5$ with bootstrap errors and 95% ranges |
| `docs/data/thresholds/timing.csv` | CPU time per shot of each decoder |
| `docs/data/fault_distance.csv` | exact circuit-level fault distance from CP-SAT |
| `docs/data/rounds_threshold/sweeps_pooled.csv` | SD6 sweep for $r=1$ to 6, 8, 10 and $r=d$ rounds, $d=3$ to 19 |
| `docs/data/rounds_threshold/thresholds.csv` | scaling fits per $r$ and fits to three consecutive distances |
| `docs/data/helios/helios_mwpm_sinter.csv` | MWPM runs of the Helios, Helios CB and H2 models (sinter format, `lam` is $p/p_0$) |
| `docs/data/helios/helios_hcm.csv`, `helios_hcm_rounds.csv` | HCM runs of the same models, scans of $p/p_0$ and rounds at the device point $p=p_0$ |
| `docs/data/helios/ion_p_sweeps.csv` | dense $p$ sweeps of `CircuitNoise.trapped_ion` for Helios and H2, with and without crosstalk |
| `docs/data/helios/helios_budget.csv` | Helios error budget: one component removed or kept at a time |
| `docs/data/cfe/*.npz`, `cfe_summary.csv` | CFE threshold sweep with paired MWPM, HCM and belief-matching decoding |

To add new points, run `scripts/run_decoder_threshold_sweep.py` and list its
output files in `XYZ2_EXTRA_SWEEPS`, separated by `os.pathsep`, which is a
colon on Linux and macOS and a semicolon on Windows. Every script then pools
them with the published data.

## Fault distance

`scripts/exact_fault_distance.py` decides with CP-SAT whether an undetectable
logical error with at most $k$ faults exists:

```bash
python scripts/exact_fault_distance.py --distance 5 --rounds 5 --max-faults 5   # INFEASIBLE
python scripts/exact_fault_distance.py --distance 5 --rounds 5 --max-faults 6   # OPTIMAL
```

CP-SAT does not finish for $d=7$ with seven rounds. For that instance
`scripts/fault_distance_lp_bound.py` gives a certified lower bound of seven
from a linear program on the decomposed detector error model, and the
undetectable-error search of Stim finds a logical error with eight faults:

```bash
python scripts/fault_distance_lp_bound.py --distance 7 --rounds 7   # fault distance >= 7
```

## Threshold versus the number of rounds

`scripts/run_rounds_threshold_sweep.py` samples SD6 noise for $r=1$ to 6, 8
and 10 rounds and for $r=d$, with every odd distance up to 19 and MWPM on the
$S_0$ detectors. A coarse pass covers $2\times10^{-3}\le p\le 0.1$, and a
second pass samples eight values of $p$ in a window around the crossing points:

```bash
python scripts/run_rounds_threshold_sweep.py --out coarse.csv --pass coarse
python scripts/run_rounds_threshold_sweep.py --out fine.csv --pass fine --max-shots 60000 --max-errors 1500
python paper/analysis/rounds_thresholds.py coarse.csv fine.csv
python paper/analysis/rounds_plots.py
```

The code needs odd $d$, so $d=19$ is the largest distance below 20.
