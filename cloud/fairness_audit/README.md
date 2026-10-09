# Fairness audit of the competitor circuits (sd6)

Automated audit run on 2026-10-09 against `main` at `e2db03a`. No file under `src/mdr/ft` was changed; the
ablations below are applied to copies of the circuits inside this folder (`common.ablate`).
Machine: 2 CPU cores. Total compute about 1 h 15 min.

| file | what it is |
|---|---|
| `common.py` | circuit builders (ours, `stim.Circuit.generated`, ablations), noise census, Monte-Carlo loop (stim sampler + `competitor_decoder`) |
| `census.py` → `census.json` | noise census, detector/observable counts, distances, XZZX-vs-CSS DEM identity (d = 3, 5, 7) |
| `distance.py` → `distance.json` | circuit distance of XZZX (sd6, biased100, purez) and CSS, d = 3–9, both bases |
| `mc.py 2a / 2b / abl` → `results_*.jsonl` | logical-error sweeps (MWPM and correlated MWPM) |
| `analyze.py` → `tables.md` | LER tables, pairwise crossings, finite-size fits |
| `structure_tables.md` | census and distance tables |
| `pytest_log.txt` | test-suite output |

Reproduce: `pip install -e ".[decoders,dev]"`, then from this folder `python census.py; python distance.py;
python mc.py 2a; python mc.py 2b; python mc.py abl; python analyze.py > tables.md`.

Error bars: LER ± is the 1σ Wilson interval (≥ 3000–4000 logical errors per point). Threshold ± is a 1σ
bootstrap over those binomial errors; it is a statistical error only and does not cover fit-model systematics,
which are about ±0.02 % judging by the spread of pairwise crossings.

## 1. Test suite

`pytest -q`: **920 passed, 6 skipped, 0 failed** (302 s). The 6 skips are
`tests/test_competitor_circuits.py:460` (×2) and `:472` (×4), "bm_pkg needs an optional package"
(`beliefmatching` is not in the `decoders` extra). 13 DeprecationWarnings from `fork()` in `tests/test_campaign.py`.
The repository's `addopts` includes `-q`, so `pytest -q` runs at `-qq` and prints no summary line; the counts come
from a second run with `-o addopts=""`.

## 2a. CSS rotated surface code: our circuit against `stim.Circuit.generated`

Stim reference: `surface_code:rotated_memory_{x,z}`, rounds = d, with `after_clifford_depolarization`,
`before_measure_flip_probability`, `after_reset_flip_probability` and `before_round_data_depolarization` all = p.

**Structure.** Both circuits have d² data qubits, d² − 1 ancillas, the same 2·(d²−1)/2 + (d−1)(d²−1) detectors
(24 / 120 / 336 for d = 3 / 5 / 7) and 1 observable. Circuit distance is d for both, in both bases: shortest
graphlike error d = 3, 5, 7, and `search_for_undetectable_logical_errors` gives d for d = 3, 5, 7.

**Noise census at d = 5, r = 5** (sites / summed probability in units of p):

| channel | ours (CSS, either basis) | Stim generated | effect on the competitor |
|---|---|---|---|
| DEPOLARIZE2 after each CX/CZ | 400 | 400 | equal |
| DEPOLARIZE1 on data | **350**: 2 per round (ancilla-reset step and ancilla-measure step, `_idle_mr`) + 100 for boundary data qubits idling in a CX layer | **125**: 1 per round (`before_round_data_depolarization`) | **hurts** (+225 p) |
| DEPOLARIZE1 on ancillas | 80: weight-2 ancillas idling in 2 of the 4 CX layers | 120: the two H gates on each X-type ancilla | **helps** (−40 p); ours uses native RX/MX and CZ, so there are no H gates |
| ancilla reset flip | 120 | 120 (+24 after the last MR, which do nothing) | equal |
| ancilla measurement flip | 120 | 120 | equal |
| data prep / readout flips | 25 / 25 | 25 / 25 | equal |
| idle noise during CX layers | yes (`_idle`) | none | hurts slightly |
| TICKs per round | 6 (R, 4×CX, M) | 7 (H, 4×CX, H, MR) | — |
| Σ DEM probability, p = 10⁻³ | 1.044 | 0.860 | ours has 21 % more |

**Logical error rates, MWPM** (full table in `tables.md`):

| d | p | basis | ours | Stim generated | ours / Stim |
|---|---|---|---|---|---|
| 3 | 0.4 % | X | 0.0244 ± 0.0004 | 0.0129 ± 0.0002 | 1.89 ± 0.04 |
| 5 | 0.4 % | X | 0.0177 ± 0.0003 | 0.0087 ± 0.0001 | 2.04 ± 0.05 |
| 7 | 0.4 % | X | 0.0114 ± 0.0002 | 0.0052 ± 0.0001 | 2.17 ± 0.05 |
| 7 | 0.4 % | Z | 0.0110 ± 0.0002 | 0.0043 ± 0.0001 | 2.54 ± 0.06 |
| 7 | 0.6 % | X | 0.0449 ± 0.0007 | 0.0239 ± 0.0004 | 1.88 ± 0.04 |
| 7 | 0.8 % | X | 0.1087 ± 0.0016 | 0.0638 ± 0.0010 | 1.70 ± 0.04 |
| 7 | 0.8 % | Z | 0.1106 ± 0.0016 | 0.0517 ± 0.0008 | 2.14 ± 0.05 |

At every (d, p, basis) of the requested grid, our CSS circuit has **1.7–2.5× the logical error rate** of Stim's.

**Thresholds, MWPM, r = d, d = 5–11** (finite-size fit over d ≥ 7):

| circuit | basis X | basis Z |
|---|---|---|
| ours (CSS) | 0.687 ± 0.005 % | 0.687 ± 0.005 % |
| Stim generated | 0.700 ± 0.005 % | 0.768 ± 0.005 % |
| ours without the reset-step data idle (`css_mr`, d ≤ 9) | 0.81 ± 0.01 % (crossings 0.816, 0.815) | — |
| … and without gate-layer idles (`css_mr_noidle`, d ≤ 9) | crossings 0.71 → 0.77, still drifting | — |

So the extra data idle per round (`_idle_mr` in the reset step, `src/mdr/ft/competitor_circuits.py:511`) costs our
CSS circuit about 0.1 percentage points of MWPM threshold. Stim's weaker basis (X, hurt by the H gates on the
X ancillas) is 0.700 %, slightly above our 0.687 %. **The competitor circuit is not favoured. Compared with Stim's
reference it is mildly handicapped** (2× the LER at fixed p, about −2 % in weaker-basis threshold). XYZ² is built with
the same convention (`FTMDRCircuit._build_gates` applies `_idle_mr` to the data in both the reset step,
`ft_mdr_circuit.py:474`, and the measurement step, `:545`), so the head-to-head stays consistent. Absolute
numbers are not comparable with literature that uses the merged-MR SD6 convention.

## 2b. XZZX against CSS under sd6

- **The detector error models are identical.** The undecomposed DEM text of `competitor_circuit("xzzx", …)` equals
  that of `"css"` character for character at d = 3, 5, 7 in both bases (`census.json`, `dem_identity`). Only
  `decompose_errors=True` picks slightly different decompositions of a few hyperedges (e.g. 1980 vs 1978 mechanisms at d = 5, X).
- Thresholds, r = d, d = 5, 7, 9, 11, p = 0.60–0.90 %:

| circuit | decoder | basis | cross 5/7 | cross 7/9 | cross 9/11 | FSS (d ≥ 7) |
|---|---|---|---|---|---|---|
| CSS | MWPM | X | 0.687 ± 0.024 | 0.689 ± 0.010 | 0.692 ± 0.015 | 0.687 ± 0.005 |
| CSS | MWPM | Z | 0.696 ± 0.017 | 0.649 ± 0.025 | 0.712 ± 0.021 | 0.687 ± 0.005 |
| XZZX | MWPM | X | 0.674 ± 0.017 | 0.710 ± 0.026 | 0.687 ± 0.010 | 0.686 ± 0.005 |
| XZZX | MWPM | Z | 0.706 ± 0.010 | 0.682 ± 0.025 | 0.719 ± 0.011 | 0.708 ± 0.005 |
| CSS | corr. MWPM | X | 0.772 ± 0.010 | 0.821 ± 0.015 | 0.824 ± 0.013 | 0.820 ± 0.006 |
| CSS | corr. MWPM | Z | 0.783 ± 0.011 | 0.817 ± 0.010 | 0.800 ± 0.031 | 0.818 ± 0.007 |
| XZZX | corr. MWPM | X | 0.806 ± 0.017 | 0.820 ± 0.012 | 0.802 ± 0.018 | 0.810 ± 0.005 |
| XZZX | corr. MWPM | Z | 0.815 ± 0.014 | 0.789 ± 0.039 | 0.810 ± 0.011 | 0.820 ± 0.007 |
| Stim CSS | MWPM | X | 0.676 ± 0.008 | 0.717 ± 0.020 | 0.706 ± 0.013 | 0.700 ± 0.005 |
| Stim CSS | MWPM | Z | 0.738 ± 0.016 | 0.813 ± 0.031 | 0.774 ± 0.017 | 0.768 ± 0.005 |

  XZZX and CSS agree within the scatter of the crossings (MWPM 0.69–0.71 %, correlated MWPM 0.81–0.82 %). The
  pointwise LERs agree within about 2σ, as the identical DEMs require.
- **Where 0.81 % vs 0.66 % comes from: the decoder.** Our MWPM threshold is 0.69–0.71 %, matching the campaign's
  "MWPM for every code: 0.69–0.70 %" and within 0.03–0.05 of the literature's 0.66 % (MWPM). Correlated matching adds
  +0.12 and gives 0.81–0.82 %, which reproduces the campaign's "best decoder" 0.81 %. It is not a circuit
  difference. Our circuit has *more* noise than Stim's reference (2a), and no noise is missing: every gate,
  reset, measurement and idle carries p. The schedule is the hook-safe one (2c), and the hooks do not cut the
  distance. The remaining 0.03–0.05 gap between our MWPM value and 0.66 % fits differences in the literature's
  conventions (e.g. explicit H gates in the XZZX compilation, threshold-fitting method). I did not re-implement
  arXiv:2505.17718's circuit, so that last attribution is not verified. Under sd6 the XZZX number to compare
  with the literature MWPM figure is 0.69–0.71 %, not 0.81 %.

## 2c. XZZX gate ordering keeps distance d

| noise | d = 3 | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| sd6, X / Z basis | 3 / 3 | 5 / 5 | 7 / 7 | 9 / 9 (graphlike) |
| biased η = 100, X / Z | 3 / 3 | 5 / 5 | 7 / 7 | 9 / 9 (graphlike) |
| purez (η → ∞), X / Z | 3 / 3 | 5 / 5 | 7 / 7 | 9 / 9 (graphlike) |

d ≤ 7 is checked with `search_for_undetectable_logical_errors` (detection-event sets ≤ 6 for d ≤ 5, ≤ 4 for d = 7)
and with `shortest_graphlike_error(ignore_ungraphlike_errors=False)`. d = 9 uses the graphlike search only.
CSS gives the same. The existing test `test_hook_unsafe_orders_lose_distance` (passing) confirms that the search is
sensitive to the order: swapping the two shapes drops the distance.

## Verdict

**The competitor circuits are fair, and they are not favoured.** Our CSS/XZZX sd6 circuits contain every
noise channel of Stim's reference circuit except the H-gate noise (which our native RX/MX/CZ compilation does
not need), plus a second data-idle step per round and idle noise in the gate layers. That gives them 21 % more total fault
probability, about 2× the logical error rate of `stim.Circuit.generated`, and a slightly lower MWPM threshold (0.687 %
vs 0.700 % in the weaker basis). They keep circuit distance d in both bases. XZZX is exactly Clifford-equivalent to
CSS under sd6 (identical DEMs), and its 0.81 % sd6 threshold comes from correlated matching. With MWPM it is
0.69–0.71 %, consistent with the literature's 0.66 %.

**Recommended fixes (documentation only, no code fix needed for fairness):**

1. `docs/noncss_comparison.md`, §1 table row "SD6": state that our sd6 has separate ancilla-reset and
   ancilla-measure steps, so the data qubits idle twice per round (`competitor_circuits.py:511` and `:553`,
   `ft_mdr_circuit.py:474` and `:545`). That is harsher than the merged-MR SD6 of Stim/Gidney: our MWPM threshold
   would be ≈ 0.81 % without the reset-step idle. Compare the XZZX literature 0.66 % (MWPM) with our MWPM 0.69–0.70 %,
   not with the best-decoder 0.81 %.
2. If you want absolute numbers to line up with the literature, an optional variant (applied to XYZ² *and* every
   competitor) is to drop `_idle_mr` in the ancilla-reset step when reset and measurement can share a step. That changes
   all codes and would need a re-run, so it is not a fairness fix.
3. Side note on XYZ², not the competitors: the optional `share_ancillas > 0` path of `FTMDRCircuit._build_gates`
   (`ft_mdr_circuit.py:510–520`) adds a mid-round MX/RX step in which the data and the other ancillas get
   crosstalk only, with no `_idle_mr`/`_idle`/`_memory`. That would favour XYZ². The campaign does not use it
   (`share_ancillas=0` by default, and `scripts/campaign.py:443` does not set it), so no reported number is affected.
   If the path is ever used, add `self._idle_mr(c, data)` and the ancilla idle there.

---

Data tables: `tables.md` (all LERs) and `structure_tables.md` (census, structure, distances).
