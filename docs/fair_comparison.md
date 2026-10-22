# Fair comparison of XYZ² with other codes under circuit-level noise

The goal is to find a hardware setting in which XYZ² has a higher circuit-level threshold than the XZZX surface code (and CSS, XY, honeycomb), and to show that the comparison is fair. These rules apply to every run in `docs/data/campaign/competitors*.md` and `docs/noncss_comparison.md`.

## 1. Noise

- Every code is built with the same noise primitives: `CircuitNoise` and `FTMDRCircuit`'s idle, reset, measurement, memory-dephasing and crosstalk methods, which `competitor_circuits.py` reuses.
- Every operation a circuit uses carries noise at the model's rate. Nothing is noiseless, except in the diagnostics of §3.4.
- Rates are per operation. A code whose rounds take more steps or more checks pays for them with more noise.

## 2. Hardware and compilation

A variant that gives XYZ² a hardware capability gives it to every competitor too. Each code then uses its best compilation on that hardware, and the comparison takes each code's best threshold.

| variant | hardware | XYZ² | competitors |
|---|---|---|---|
| `sd6`, `si1000`, `biased*`, ion models | controlled-Pauli gates | gates | gates |
| `em3` | pair measurements only | pairs | pairs (honeycomb native) |
| `hyb` | gates **and** pair measurements, both at rate p | weight-2 links as single pair measurements, plaquettes with gates | best of: all gates (= `sd6`), all pairs (= `em3`); the third compilation, weight-2 boundary checks as single pair measurements (`weight2="direct"`), has fault distance (d + 1) / 2 and is not used |
| `sd6_lr2`, `sd6_lr3`, `si1000_lr2`, `hyb_lr2` | as the base model | links measured 2 or 3 times per round | surface codes have no links: compared with their best on the base hardware |
| `phen`, `phen_b10` | phenomenological (as in arXiv:2505.03691) | same model | same model |
| `hw_g`, `hw_g1q`, `hw_t`, `hw_rr`, `hw_tr` (diagnostics, §3.4) | one error channel at a time, every other operation noiseless | gates | gates |
| `sd6_noidle`, `sd6_il_noidle` (diagnostics, §3.4) | `sd6` and `sd6_il` without idle errors | gates | gates (`sd6_il_noidle` against `sd6_noidle`) |

## 3. What counts

1. **Threshold.** A finite-size-scaling fit over d ≥ 11, on a series not flagged `drift`. A series is flagged only for downward drift: the crossing of the two largest distances is below 0.75 of the crossing of the two smallest, and the last three crossings keep falling. Series whose crossings drift upward, or fall without a monotone tail, are still fitted and count. Crossings of only two distances do not count.
2. **Decoders.** Each code is measured with its best decoder:
   - XYZ²: every decoder of `TwoLevelDecoder` (14 decoders in `docs/data/campaign/thresholds.csv`);
   - surface codes: MWPM, correlated MWPM and belief-matching. BP-OSD and Tesseract were also run, but only up to d = 11, so they give two-distance crossings and no fit, and by §3.1 they do not count;
   - honeycomb: MWPM, correlated MWPM and belief-matching.

   The decoder sets are therefore unequal: XYZ² is scored with the best of 14 decoders, every competitor with the best of three (MWPM, correlated MWPM, belief-matching).
3. **Logical bases.** A memory must protect every logical operator. Competitors count with their weaker basis, and a competitor without a usable threshold (rule 1) in both of its bases is left out of that comparison. XYZ² must be run in both of its conjugate logical bases where possible, and counts, per extraction schedule, with the weaker one, then with its best schedule. The rule is not symmetric: when no schedule has usable thresholds in both bases, `competitor_report.py` keeps XYZ²'s best single-basis value, marked "only", whether or not the other basis was run. At `helios_p` r = 1, 3 and `h2_p` r = 1, 2, 3, 5 both bases were run, but only one has a usable fit, so the XYZ² value there is a single-basis threshold, not its weaker-basis threshold.
4. **Diagnostics.** These models make some operations noiseless or remove an error channel. They only locate where XYZ² gains or loses threshold and are never hardware claims:
   - `sd6_il`: noiseless link checks (`docs/gap_diagnostic.md`);
   - `sd6_noidle`, `sd6_il_noidle`: `sd6` and `sd6_il` with the idle errors set to zero (`docs/gap_diagnostic.md`);
   - `hw_g`, `hw_g1q`, `hw_t`, `hw_rr`: the corners of the Hetenyi-Wootton model (arXiv:2306.17786): gates only, gates with noisy single-qubit gates, idling only with η = 20, readout only, every other operation noiseless; `hw_tr`: the `hw_t` channel only on the data qubits during ancilla readout and reset (`docs/hetenyi_wootton_check.md`).
5. **Footprint.** Qubit counts per distance are reported next to thresholds (XYZ² uses about 4d² qubits against 2d² for a rotated surface code), together with logical error at equal qubit count where the data allows.
