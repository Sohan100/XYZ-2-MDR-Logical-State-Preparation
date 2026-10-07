# Fair comparison of XYZ² with other codes under circuit-level noise

The goal is to find a hardware setting in which XYZ² has a higher circuit-level threshold than the XZZX surface code (and CSS, XY, honeycomb), and to show that the comparison is fair. These rules apply to every run in `docs/data/campaign/competitors*.md` and `docs/noncss_comparison.md`.

## 1. Noise

- Every code is built with the same noise primitives: `CircuitNoise` and `FTMDRCircuit`'s idle, reset, measurement, memory-dephasing and crosstalk methods, which `competitor_circuits.py` reuses.
- Every operation a circuit uses carries noise at the model's rate. Nothing is noiseless, except in the diagnostic of §3.4.
- Rates are per operation. A code whose rounds take more steps or more checks pays for them with more noise.

## 2. Hardware and compilation

A variant that gives XYZ² a hardware capability gives it to every competitor too. Each code then uses its best compilation on that hardware, and the comparison takes each code's best threshold.

| variant | hardware | XYZ² | competitors |
|---|---|---|---|
| `sd6`, `si1000`, `biased*`, ion models | controlled-Pauli gates | gates | gates |
| `em3` | pair measurements only | pairs | pairs (honeycomb native) |
| `hyb` | gates **and** pair measurements, both at rate p | weight-2 links as single pair measurements, plaquettes with gates | best of: all gates (= `sd6`), all pairs (= `em3`), weight-2 boundary checks as pair measurements if that keeps distance d |
| `sd6_lr2`, `sd6_lr3`, `si1000_lr2`, `hyb_lr2` | as the base model | links measured 2 or 3 times per round | surface codes have no links: compared with their best on the base hardware |
| `phen`, `phen_b10` | phenomenological (as in arXiv:2505.03691) | same model | same model |

## 3. What counts

1. **Threshold.** A finite-size-scaling fit over d ≥ 11, with no drift. Crossings of only two distances do not count.
2. **Decoders.** Each code is measured with its best decoder:
   - XYZ²: every decoder of `TwoLevelDecoder`;
   - surface codes: MWPM, correlated MWPM, belief-matching, BP-OSD and Tesseract;
   - honeycomb: MWPM, correlated MWPM and belief-matching.
3. **Logical bases.** A memory must protect every logical operator. Competitors count with their weaker basis. XYZ² must be run in both of its conjugate logical bases where possible, and counts with the weaker one; where only one basis is available, this is stated.
4. **Diagnostic.** `sd6_il` (noiseless link checks) only locates where XYZ² loses threshold. It is never a hardware claim.
5. **Footprint.** Qubit counts per distance are reported next to thresholds (XYZ² uses about 4d² qubits against 2d² for a rotated surface code), together with logical error at equal qubit count where the data allows.
