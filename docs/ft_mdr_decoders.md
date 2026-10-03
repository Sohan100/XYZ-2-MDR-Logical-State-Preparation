# Decoders and thresholds for fault-tolerant MDR

Code: `src/mdr/ft/two_level_decoder.py`, `src/mdr/ft/circuit_noise.py`.
Sweeps: `scripts/run_decoder_threshold_sweep.py`. Tests:
`tests/test_two_level_decoder.py`. Fault distance:
`scripts/exact_fault_distance.py` and `scripts/fault_distance_lp_bound.py`.

Data in `docs/data/thresholds/`. `sweeps_pooled.csv` has one row per
simulated point, with shots and errors summed over independent runs.
`fits.csv` has the finite-size scaling fits and the belief-matching crossings.
`bm_pooled.csv` has the belief-matching points and `bm_crossings.csv` the
crossings of $d=5/7$ and $d=3/5$ with their bootstrap errors. `timing.csv` has
the CPU time per shot of each decoder on one core at $d=5$, $r=5$ and SD6
noise with $p=4\times10^{-3}$.

## Detectors

Build the circuit with `FTMDRCircuit(d, r, noise, detectors="combined")`.
Besides the $d^2(r+1)$ detectors of the frame-deterministic group $S_0$,
this declares one gauge detector per round from round 2 for every
generator whose first outcome is random. These generators are the odd
links, the class-A hexagons and the class-A boundary checks, which gives
$(d^2-1)(r-1)$ gauge detectors in total. With `r = 1` there are no gauge
detectors and every decoder below reduces to matching on $S_0$.

## Decoders

| `TwoLevelDecoder` mode | paper name | information used |
|---|---|---|
| `mwpm` | MWPM | $S_0$ detectors only |
| `seq_soft` | sequential BP | BP on link detectors, posteriors pushed onto $S_0$ edges, then matching |
| `seq_match` | erasure passing | matching on link detectors, $S_0$ edges of matched faults erased |
| `bp_full` (`bp_method="product_sum"`, `bp_iters=10`) | belief-matching | BP on all detectors, then matching on $S_0$ |
| `corr_split`, `lower="links"` | HCM with links | correlated matching, $S_0$ with odd links |
| `corr_split`, `lower="gauge"` | HCM | correlated matching, $S_0$ with all gauge detectors |
| `tesseract` | Tesseract | search on the full detector error model |
| `cfe` (`kappa=0.5`, `osd_order=10`, `cfe_guided=True`) | coset free-energy (CFE) | two-class BP-OSD on the full detector error model, local descent, entropy correction |

HCM splits each fault of the detector error model into its $S_0$ part and
its gauge part, joins them into one correlated mechanism and runs the
two-pass correlated matching of PyMatching 2.3 or later with
`enable_correlations=True`. It costs about four times as much as plain
matching, 32 µs against 7 µs per shot at $d=5$ and $r=5$.

### Coset free-energy decoder (`src/mdr/ft/coset_decoder.py`)

The optimal decoder compares the total probability $Z_\ell$ of the two
logical classes of errors that explain the syndrome. CFE estimates the free
energies $F_\ell=-\ln Z_\ell$:

1. **Two-class search.** The observable row $L$ is appended to $H$, and BP-OSD
   (product-sum BP, 30 iterations, combination-sweep OSD of order 10) solves
   $(H; L)\,x = (\sigma; \ell)$ for $\ell = 0$ and $1$. A second pair of runs starts
   from priors raised to 0.3 on the faults of the HCM solution
   (`cfe_guided=True`), so each class has two candidates.
2. **Local descent.** Degeneracy moves are sets of three or four faults whose
   detector and observable flips cancel (a $Y$ fault and its $X$ and $Z$
   parts, the two paths around a square). Greedy moves lower the weight
   $w(x)=\sum_j \lambda_j x_j$, with $\lambda_j=\ln[(1-p_j)/p_j]$ the
   log-likelihood ratio of fault $j$, of every candidate until no move helps.
3. **Free energy.** $F_\ell = w(x_\ell) - \kappa \sum_g \ln(1+e^{-\Delta w_g})$ over
   the moves that touch $x_\ell$; the class with the smaller $F_\ell$ wins.
   `kappa=0` is most-likely-error decoding with a two-class search.

On the same 3000 shots at $d=5$, $r=5$, $p=5\times10^{-3}$ (failures):

| decoder | SD6 | biased, $\eta=100$ |
|---|---|---|
| MWPM | 187 | 162 |
| HCM | 109 | 58 |
| belief-matching | 102 | 30 |
| BP-OSD | 117 | 25 |
| two-class BP-OSD | 146 | 23 |
| + local descent | 108 | 14 |
| + free energy, $\kappa=1/2$ | 103 | 13 |
| + guided candidates (CFE) | 84 | 12 |
| Tesseract | 85 | 15 |

Threshold sweeps with paired MWPM, HCM and belief-matching on the same shots:
`scripts/run_cfe_threshold_sweep.py`, analysis in
`paper/analysis/cfe_plots.py`, data in `docs/data/cfe/` (one `.npz` per noise,
distance and $p$, with the weight and entropy of every candidate). Failures of
CFE over failures of belief-matching on the same shots, summed over $p$
($r=d$, bootstrap errors):

| noise | $d=3$ | $d=5$ | $d=7$ |
|---|---|---|---|
| SD6, $p=5$ to $7\times10^{-3}$ | 0.96(2) | 0.94(3) | 1.18(12) |
| biased $\eta=100$, $p=7$ to $11\times10^{-3}$ | 0.89(2) | 0.79(3) | 0.72(6) |

Under strong bias CFE fails on 11% to 28% fewer shots than belief-matching and
the gain grows with $d$, so its crossing point lies above that of
belief-matching. Under depolarizing noise the gain is gone at $d=7$: the
ordered-statistics search no longer finds the best candidates of both classes.
Adding a third pair of candidates seeded by the belief-matching solution (an
experiment, not in the package) lowers the SD6 count at $d=7$ from 120 to 111
against 102 for belief-matching, so the remaining gap is in the candidate
search and not in the free-energy comparison.

```python
from mdr.ft import CircuitNoise, FTMDRCircuit, TwoLevelDecoder

ft = FTMDRCircuit(5, 5, CircuitNoise.biased(5e-3, 100), detectors="combined")
dec = TwoLevelDecoder(ft, mode="corr_split", lower="gauge")
est = dec.estimate(max_shots=100_000, max_errors=1000)
print(est.rate)
```

## Noise models

| constructor | model |
|---|---|
| `CircuitNoise.uniform(p)` | SD6: $p$ on every gate, idle step, preparation and measurement |
| `CircuitNoise.si1000(p)` | SI1000: 2Q $p$, idle $p/10$, prep $2p$, meas $5p$, $2p$ on data during ancilla reset and measurement |
| `CircuitNoise.biased(p, eta)` | total $p$ per location, $Z$ fraction $\eta/(\eta+1)$, `eta = 0.5` is depolarizing on one qubit |
| `CircuitNoise.trapped_ion(p, "helios")` | one-parameter Helios model: 2Q depolarizing $p$, 1Q $0.045p$, memory $0.9p$, crosstalk $0.075p$ per measured ancilla, prep and meas $0.25p$; the device is at $p_0=10^{-3}$. See `docs/ft_mdr_protocol.md` for the channels and their verification |
| `CircuitNoise.trapped_ion(p, "h2")` | one-parameter H2 model: 1Q $0.024p$, memory $0.4p$, crosstalk $0.008p$, prep and meas $0.4p$; the device is at $p_0=1.875\times10^{-3}$ |
| `CircuitNoise.trapped_ion(p, machine, crosstalk=False)` | the same without measurement crosstalk |
| `CircuitNoise.quantinuum(machine, scale)` | the same models with $p=$ `scale` $\times\,p_0$; `two_qubit="cb"` uses the cycle-benchmarked Pauli rates of the Helios RZZ gate |
| `CircuitNoise.em3(p)` | EM3 (Gidney et al. 2021): every pair measurement is followed with probability $p$ by a random two-qubit Pauli and a random flip of its outcome, idle $p$, prep and meas $p$. `FTMDRCircuit` then builds the pair-measurement circuit of `docs/ft_mdr_protocol.md` |

## Thresholds ($r = d$)

| noise | MWPM | HCM, links | HCM | belief-matching |
|---|---|---|---|---|
| SD6 | 0.41% ± 0.01% | 0.50% ± 0.01% | 0.50% ± 0.02% | 0.55% ± 0.02% |
| SI1000 | 0.36% ± 0.01% | 0.43% ± 0.01% | 0.44% ± 0.02% | 0.46% ± 0.02% |
| biased, eta=10 | 0.44% ± 0.01% | 0.57% ± 0.01% | 0.64% ± 0.01% | 0.79% ± 0.03% |
| biased, eta=100 | 0.44% ± 0.01% | 0.59% ± 0.01% | 0.68% ± 0.01% | 0.86% ± 0.01% |
| pure Z | 0.44% ± 0.01% |  | 0.69% ± 0.01% | 0.83% ± 0.04% |
| EM3 | 0.22% ± 0.01% |  | 0.28% ± 0.01% |  |

Thresholds are in percent of $p$ (`paper/analysis/tables.py` writes this
table to `paper/build/tables/md_thresholds.md`). MWPM and both HCM variants
use $d=5,7,9$ and the finite-size scaling fit of the paper
(`paper/analysis/thresholds.py`). For belief-matching we sampled 8 to 12
values of $p$ within $\pm15\%$ of the crossing, fitted $\ln p_L$ against
$\ln p$ for $d=5$ and $d=7$ separately and quote the intersection of the two
lines, with a parametric bootstrap for the error
(`paper/analysis/bm_thresholds.py`, `paper/analysis/crossings.py`). The
crossing of $d=3$ and $d=5$ agrees within the errors except for pure $Z$.
Under EM3 the plaquettes are measured with pair measurements, so the fault
distance is $d$ instead of $d+1$.

The Helios and H2 models have no threshold. Their crosstalk per data qubit and
round grows with the number of measured ancillas, so the crossing points of
consecutive distances fall as $d$ grows. `docs/ft_mdr_protocol.md` lists the
crossings with and without crosstalk.

Reproduce one row:

```bash
python scripts/run_decoder_threshold_sweep.py --noise biased100 \
    --decoders mwpm corr_gauge --distances 3 5 7 9 \
    --values 4e-3 5e-3 6e-3 6.5e-3 7e-3 8e-3 --out data/ft_mdr/thresholds.csv
XYZ2_EXTRA_SWEEPS=data/ft_mdr/thresholds.csv python paper/analysis/thresholds.py
```

The EM3 row comes from `sh scripts/run_em3_sweeps.sh` and the Helios and H2
crossings from `sh scripts/helios/run_ion_p_sweeps.sh`.
