# The corners of the Hetényi–Wootton threshold surface in our circuits

**Question.** Hetényi and Wootton (arXiv:2306.17786, Phys. Rev. A 109, 032433) fit a plane to the threshold surface of a spin-qubit model with three error parameters. Their Table I gives the intercepts of that plane, that is, the threshold with only one kind of error switched on. What are the same three corners for XYZ², the rotated CSS and XZZX surface codes and the honeycomb Floquet code in our circuits, with our decoders, distances and fitting rules? Do the rankings agree?

**Status (2026-10-14).** Done. 11,640 tasks, no failures, 483 core-hours (upper bound 1,857). No fitted threshold lay within 25% of an end of its grid, so no series needed a second pass. Fits: `docs/data/campaign/hw/`. Paper text: `paper/overleaf_patches/hw_relation.tex`.

## 1. Their model and their numbers

Their Table I (ηT = 20, ηG = 1, d = 11 to 17, d rounds, matching), in %:

| code | gates p_G | idling p_T | reset and readout p_RR |
|---|---|---|---|
| rotated surface code | 0.82 | 3.94 | 14.5 |
| XZZX surface code | 0.37 | 15.1 | 12.9 |
| XYZ² | 0.465 | 4.2 | 16.5 |
| honeycomb Floquet code | 0.43 | 1.16 | 0.99 |

What their definitions say (from the paper text):

- **Gates (Eqs. 1–2).** Single- and two-qubit gates are followed by depolarizing noise with p_G1 = 2p_G/(1 + ηG) and p_G2 = 2p_G ηG/(1 + ηG). At ηG = 1 both equal p_G, so a single-qubit gate is as noisy as a two-qubit gate. Their XZZX circuit adds single-qubit basis changes on the data qubits, and their XYZ² circuit adds X and Y basis changes, so single-qubit errors weigh more in these two codes. Their surface-code circuits also use SWAP gates.
- **Idling (Eqs. 3–5, 12, Fig. 1).** Relaxation gives X and Y errors with p_T1/2 each, dephasing gives Z errors with p_T2, ηT = p_T2/p_T1 = p_Z/(p_X + p_Y) and p_T = p_T1 + p_T2. The idling error acts on the data qubits while the ancillas are read out and reset (their Sec. II.4 and the caption of Fig. 1). Gates are fast, so there is no idling error in the gate layers.
- **Reset and readout (Eqs. 6–8).** A readout error is a two-qubit depolarizing error on the readout pair followed by a classical flip of the outcome. A faulty reset gives the same syndrome as a faulty readout, so both are merged into one parameter p_RR.
- **The intercepts** come from a plane fitted to thresholds measured at mixtures of the three errors, not from runs at the corners themselves.
- **Their XYZ² links are ZZ** (Sec. IV.2: the concatenation with "a two-qubit repetition code (stabilized by ZZ)", "ZZ link along the XY edges"). Z-biased idling at ηT = 20 therefore falls on the Pauli that commutes with the links, the weak direction of the code. It is the ZZ-link image of the X-biased case of arXiv:2112.06036 (their Sec. 3: X bias with XX links is Z bias with ZZ links; Sec. 4.3, Fig. 7(b): at η = 10 XYZ² has about XZZX's threshold but higher logical failure rates). All our `hw_*` models use XX links, for which ηT = 20 dephasing is the strong direction.
- **Two readout windows per cycle.** Their link measurements reuse the plaquette ancilla pairs and run in a separate measurement round after the plaquettes (App. D.2). Every data qubit is "subject to two idling phase during measurements" per cycle (App. D.3, Eq. D1), against one window for their surface codes. Their p_T is per window, so their 4.2% XYZ² corner is about 8.4% per cycle (first order; inference). `hw_tr` has one window per round (§2).
- **Their decoder (App. D.3)** is a matching decoder built by hand: the plaquettes of one colour together with all links form a subgraph, the two subgraphs are decoded separately, and the edge weights come from analytic single-qubit error probabilities. Ours decode Stim detector error models with MWPM, correlated matching and belief-matching.
- **Per cycle,** their XYZ²/XZZX idling ratio is 8.4/15.1 = 0.56. The `hw_tr` pilot (§4, MWPM, d = 7/9, weaker basis) gives 10.49(55)%/18.95(160)% = 0.55(6), statistical errors only. The two agree although the link Pauli, the readout windows per cycle and the decoders differ, so the agreement does not test their model.

## 2. Our one-parameter models

Each model switches on one kind of error with one parameter p (`scripts/run_decoder_threshold_sweep.py`, `NOISE`):

| name | errors | stand-in for |
|---|---|---|
| `hw_g` | DEPOLARIZE2(p) after every entangling gate, nothing else | their gate corner |
| `hw_g1q` | `hw_g` plus DEPOLARIZE1(p) on both qubits of every entangling gate (`dress_1q`) | their gate corner with single-qubit basis changes as noisy as two-qubit gates |
| `hw_t` | PAULI_CHANNEL_1(p/42, p/42, 20p/21) on every qubit that idles in a gate layer, and on the data qubits while the ancillas are reset and measured (`p_idle_mr = None`) | their idling corner, ηT = 20 by their definition |
| `hw_rr` | every measurement outcome (ancillas and the final data readout) flipped with probability p; no reset error | their combined reset-and-readout error, one flip per measured qubit |
| `hw_tr` (diagnostic) | the `hw_t` channel only on the data qubits while the ancillas are reset and read out, half of it at each step (about p per round), no idling in the gate layers | the placement of their idling error |

Choices to keep in mind when reading the numbers:

- `hw_rr` flips outcomes only. Their readout error also carries a two-qubit depolarizing error on the readout pair, and their reset error is merged into the same event. One flip per measurement is the single event that both their reset and readout errors produce in the syndrome.
- `hw_t` idles in every gate layer as well as during readout and reset. This is harsher than their model, above all for XYZ², whose link ancillas wait in four of the six gate layers of a round. `hw_tr` places the idling error where they place it.
- `hw_g1q` puts the single-qubit error on every gate of every code. In their circuits only some codes carry extra basis changes.

## 3. Checks before the large run

- **Each model puts errors only where it says** (`tests/test_hw_noise.py`): the noise channels of XYZ² (both schedules, Logical X and Y), CSS, XZZX and the honeycomb under each model, the gate noise on exactly the qubits of the gates before it, every readout flip right before a measurement of the same qubits, and the idling channel on every data qubit during the ancilla reset.
- **CSS and XZZX under `hw_g`, `hw_g1q` and `hw_rr`.** These models are invariant under single-qubit Cliffords, so XZZX is CSS in another frame. At d = 3 and 5, r = d, both bases, the undecomposed detector error models are identical (`approx_equals`, the same number of mechanisms: 219 and 1,677 at d = 3 and 5 in the X basis under `hw_g`), and the graphlike distance is d for both codes. Stim splits a few faults into different graphlike pieces in the two frames, so the decomposed models differ by a handful of components (1,978 against 1,975 at d = 5, X basis, `hw_g`). Under `hw_rr` even the decomposed models are identical.
- **XYZ² both-bases memory under `hw_g`.** At d = 5 the graphlike distance of the matching level (the S₀ detectors) is 6 = d + 1 for X̄ and 5 = d for Ȳ (also tested at d = 7: 8 and 7).
- **The earlier spot check through the campaign worker.** CSS surface code, SD6 without idling errors (`sd6_noidle`), MWPM, r = d, d = 5, 7, 9, about 4 × 10⁶ shots per basis: the crossing of d = 7 and 9 is 0.921(10)% in the X basis and 0.933(13)% in the Z basis (d = 5/7: 0.894(13)% and 0.885(10)%), against the earlier 0.91(5)%.

### The honeycomb code under readout errors alone

Under `hw_rr` a flipped edge outcome of the honeycomb code sets off three or four detectors (at d = 8: 288 faults with three and 816 with four detectors), and no other fault gives Stim the graphlike pieces to split it into. Stim's decomposition therefore raises, and matching, correlated matching and belief-matching cannot be built. Under SD6 the same fault is split with the help of data errors (for example `D12 D16 L0 ^ D20`).

With the user's approval, `competitor_decoders.hinted_decomposition` handles this case, and only this case (it runs only when Stim's own decomposition raises). Stim decomposes the circuit with DEPOLARIZE1(10⁻⁹) added on every qubit before every TICK, and the hint faults are then dropped. The result has the faults of the circuit with their probabilities, to within about 10⁻⁸, each split into graphlike pieces. Models that Stim decomposes keep Stim's decomposition (tested).

## 4. Pilot

MWPM, r = d, d = 5, 7, 9 (honeycomb 4, 8), 10 values of p per series, 300 logical errors or 2 × 10⁴ shots per point. The gate models started from the expected thresholds. The idling and readout models scanned 0.5% to 30% and were refined around the crossing. Series without a crossing were scanned again beyond their range, which affected the honeycomb under the gate models (above the range) and under `hw_t` (below). Crossings are in `docs/data/campaign/hw/pilot_centers.csv`, in the format `campaign_competitors.pilot_centers` reads (code, basis, noise, rounds, decoder, d_pair, crossing_p, lo, hi, note).

MWPM crossings of the largest pilot pair, in % (d = 7/9, honeycomb 4/8), with the bootstrap error of the fitted crossing; "grid" marks a crossing interpolated between coarse grid values; "> 44" means no crossing up to 44%. `hw_tr` is the diagnostic placement of the idling error (§2).

| model | CSS X | CSS Z | XZZX X | XZZX Z | XYZ² X̄ | XYZ² Ȳ | honeycomb X (H) | honeycomb Z (V) |
|---|---|---|---|---|---|---|---|---|
| hw_g | 1.13 ± 0.05 | 1.03 ± 0.04 | 1.16 ± 0.07 | 1.21 ± 0.03 | 0.89 ± 0.02 | 0.94 ± 0.02 | 1.05 ± 0.02 | 1.06 ± 0.02 |
| hw_g1q | 0.44 ± 0.01 | 0.42 ± 0.03 | 0.41 ± 0.03 | 0.41 ± 0.02 | 0.31 ± 0.01 | 0.34 ± 0.01 | 0.34 ± 0.01 | 0.34 ± 0.01 |
| hw_t | 2.92 ± 0.23 | 29.59 ± 1.84 | 7.96 (grid) | 7.35 ± 0.36 | 0.87 ± 0.03 | 1.01 ± 0.03 | 0.42 ± 0.01 | 0.45 ± 0.01 |
| hw_rr | 11.11 ± 1.00 | 10.18 ± 0.80 | 9.46 ± 0.41 | 9.27 ± 0.78 | 10.00 ± 0.71 | 11.20 ± 0.46 | 1.18 ± 0.02 | 1.16 ± 0.02 |
| hw_tr | 6.64 ± 0.21 | > 44 | unresolved (19–31) | 18.95 ± 1.60 | 10.49 ± 0.55 | 13.99 ± 1.66 | 1.24 ± 0.02 | 1.36 ± 0.03 |

The pilot points are not used in the fits of §6. At 2 × 10⁴ shots their small-p points can put a spurious early sign change into the crossing search of `campaign_report`.

## 5. Campaign

`scripts/campaign_hw.py tasks` builds `data/campaign/hw/tasks_hw.jsonl`:

- XYZ²: `campaign.make_tasks` with the labels `hw_g@both-X`, `hw_g@both-Y` and the same for the other models. Decoders: mwpm, corr_gauge and bm. r = d, d = 3 to 21. Grids of 14 values over a factor 4, centred on the pilot crossing of d = 7/9 times the decoder's gain in `campaign._G_DEFAULT`.
- CSS, XZZX and honeycomb: `campaign_competitors.make_tasks`, bases X and Z, decoders mwpm, corr and bm, r = d, d = 3 to 21 (honeycomb 4 to 20). Grids are centred on the pilot crossing (7/9, honeycomb 4/8) times the decoder gain.
- 11,640 tasks, 1x budgets. The upper bound of the cost is 1,857 core-hours, or 14.5 node-hours.

How it ran (2026-10-14):

- Work pool `data/campaign/hw/pool_hw.jsonl` (88 units of 128 tasks, r = d first) and `pool_hw2.jsonl` (the 420 honeycomb `hw_rr` tasks, added after the decoder fallback).
- Preempt pool jobs (`slurm/ft_mdr/run_pool.sh`, flexible node counts) were submitted but never left the queue: about 750 preempt jobs were pending, and the user's other project held the two accrual slots. They were cancelled once every unit was claimed.
- All tasks ran on pool runners on 20 to 24 login nodes (`$SCRATCH/ftmdr_tools/login_runner.sh`), within the per-user limits of 32 CPUs and 30 GiB per node. Runners first took tasks up to 1 GB, then up to 2.5 GB. They were restarted with SIGTERM checkpoints when the code changed.
- No task failed.

## 6. Results

### Their Table I next to ours

Thresholds in %, weaker basis. MWPM: matching alone. Best: the best of MWPM, correlated matching (corr) and belief-matching (BM) in each basis, then the weaker basis. Finite-size fits over d = 11 to 21 (honeycomb 12 to 20), r = d.

| code | gates, theirs | gates, MWPM | gates, best | gates + 1q, MWPM | gates + 1q, best | idling, theirs | idling, MWPM | idling, best | readout, theirs | readout, MWPM | readout, best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rotated surface (CSS) | 0.82 | 1.15(1) | 1.40(5) | 0.433(3) | 0.52(1) | 3.94 | 3.6(4) | 3.8(4) | 14.5 | 9.8(4) | 10.5(6) |
| XZZX surface | 0.37 | 1.15(2) | 1.39(1) | 0.438(4) | 0.51(1) | 15.1 | 9.8(3) | 10(2) | 12.9 | 9.8(3) | 10.3(4) |
| XYZ² | 0.465 | 0.89(2) | 1.10(4) | 0.326(8) | 0.402(7) | 4.2 | 0.88(1) | 1.330(6) | 16.5 | 10.6(3) | 10.6(5) |
| honeycomb Floquet | 0.43 | 0.88(1) | 1.081(8) | 0.293(2) | 0.372(5) | 1.16 | 0.392(3) | 0.51(1) | 0.99 | 1.007(7) | 1.72(3) |

The crossing of d = 7 and 9 (MWPM, weaker basis; honeycomb d = 4 and 8) and every threshold per basis and decoder are in `docs/data/campaign/hw/thresholds.md`. p_L against p for every series is in `fig_<model>.pdf` and `.png`.

### Rankings

- **Gates.** Ours: CSS = XZZX (1.15%, the same detector error model) > XYZ² ≈ honeycomb (0.89(2)% and 0.88(1)%; best decoders 1.10(4)% and 1.081(8)%). Theirs: CSS > XYZ² > honeycomb > XZZX.
  - The XYZ²/CSS ratio is 0.770(17) in ours (best decoders 0.78(4)) and 0.57 in theirs.
  - **Ranking change:** at the pure gate corner the honeycomb code is level with XYZ², while they place XYZ² slightly above it (0.465 against 0.43).
- **Gates with noisy single-qubit gates (`hw_g1q`).** CSS = XZZX (0.433(3)% and 0.438(4)%) > XYZ² 0.326(8)% > honeycomb 0.293(2)%, and the same order with the best decoders.
  - The XZZX value, 0.438(4)%, is close to their 0.37%. Relative to our CSS gate corner it is 0.380(6), against their XZZX/CSS ratio of 0.45.
  - This supports the explanation that noisy single-qubit basis changes cause their low XZZX value. In our circuits this error also lowers CSS, because `hw_g1q` puts it on every gate of every code.
- **Idling.** Ours: XZZX 9.8% > CSS 3.6% > XYZ² 0.88% > honeycomb 0.39%. Theirs: XZZX 15.1 > XYZ² 4.2 > CSS 3.94 > honeycomb 1.16.
  - **Ranking change:** XYZ² falls below CSS, also with the best decoders (1.330(6)% against 3.8(4)%).
  - The reason is the placement of the idling error (§2). With the idling error only during ancilla readout and reset (`hw_tr`), the MWPM pilot crossings of d = 7/9 (`diag_hw_tr_centers.csv`) are XZZX 19.0(16)% in Z (none below 44% in X), XYZ² 10.5(6)% in X̄, CSS 6.6(2)% in X and honeycomb 1.24(2)% (d = 4/8). XYZ² is above CSS again, as in their table.
    - These are pilot crossings with MWPM only. At the idling corner MWPM handicaps XYZ²: under `hw_t` the best decoder raises its threshold 1.52 times (0.88(1)% to 1.330(6)%, corr_gauge), against 1.05 times for CSS and 1.02 times for XZZX (`summary.csv`).
    - From d = 5/7 to 7/9 the crossings move by +4.7% (XZZX, Z), +0.9% (XYZ², X̄) and +1.2% (CSS, X). With each code's shift counted as a systematic error, XZZX against XYZ² is 1.7σ (5.0σ with the bootstrap errors alone) and XYZ² against CSS is 2.4σ (6.6σ). Only the honeycomb is separated from the rest, by at least 4.5σ. The crossings support the placement explanation, but they do not give a threshold ranking.
    - Their XYZ² circuit has ZZ links and reads the plaquettes and the links in two readout rounds (their Fig. 5 and its text). Ours has XX links and one readout window per round. Under Z-biased idling (ηT = 20) the two link orientations do not see the same noise, so `hw_tr` is not their XYZ² model in our circuits.
- **Readout.** With MWPM: XYZ² 10.6(3)% > CSS 9.8(4)% = XZZX 9.8(3)% > honeycomb 1.007(7)%. Theirs: XYZ² 16.5 > CSS 14.5 > XZZX 12.9 > honeycomb 0.99.
  - **Ranking change with the best decoders:** XYZ², CSS and XZZX agree within errors (10.6(5)%, 10.5(6)%, 10.3(4)%). The honeycomb stays an order of magnitude below (1.72(3)% with BM, 1.007(7)% with MWPM, against their 0.99).

### Surprises

- **The honeycomb code under readout errors alone** could not be decoded by matching at all: its faults are hyperedges (§3). With the hinted decomposition its MWPM threshold, 1.007(7)%, is within 2% of their 0.99%, and belief-matching raises it to 1.72(3)%.
- **XYZ² at the idling corner.** Its MWPM threshold is 0.877(11)% with our placement (best decoder 1.330(6)%) against 4.2% in their model, and its relative order to CSS flips with the placement, with either decoder. The link ancillas of XYZ² wait in four of the six gate layers of a round, and its data qubits in two.
- **The honeycomb code at the gate corner** reaches XYZ²'s threshold (MWPM 0.88(1)% against 0.89(2)%; best decoders 1.081(8)% against 1.10(4)%). Under SD6 it is far below (0.17% against 0.40% with MWPM): its weakness is idling and readout, not the gates.
- **Our gate-corner values exceed theirs for every code** (with MWPM by a factor 1.40(2) for CSS and 1.91(3) for XYZ²; with the best decoders 1.71(6) and 2.36(8)). Their circuits use SWAP gates and their intercepts come from a plane fitted over mixtures of errors.

## 7. Files

- `scripts/run_decoder_threshold_sweep.py`: the models `hw_g`, `hw_g1q`, `hw_t`, `hw_rr`, `hw_tr` and `sd6_noidle`.
- `scripts/campaign_hw.py`: pilot tasks, pilot fit, spot-check tasks, the campaign's task list.
- `src/mdr/ft/competitor_decoders.py`: `hinted_decomposition`. `src/mdr/ft/circuit_noise.py`, `ft_mdr_circuit.py`: `idle_mr_xyz`, used only by `hw_tr`.
- `paper/analysis/hw_report.py`: fits, tables and figures.
- `tests/test_hw_noise.py`: the checks of §3.
- `docs/data/campaign/hw/`: `points.csv` (every point: campaign, pilot and spot check), `thresholds.csv` (every series), `summary.csv`, `thresholds.md`, `crossings.csv`, `pilot_centers.csv`, `diag_hw_tr_centers.csv` and `points_diag.csv` (the `hw_tr` pilot), and `fig_<model>.pdf`/`.png`.
- Raw counts: `data/campaign/hw/` on `$SCRATCH`, mirrored to `/global/cfs/cdirs/m4980/ftmdr_campaign/hw/`. They are not merged into `docs/data/campaign/points.csv.gz`, and the main campaign is not refitted.
- Paper text: `paper/overleaf_patches/hw_relation.tex`.
