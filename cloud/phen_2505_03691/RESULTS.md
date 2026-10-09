# Phenomenological check of arXiv:2505.03691

Unattended run, 2026-10-09, branch `cloud/phen-2505-03691`, base `main` at `e2db03a`.
Machine: 2 cores, about 2 h of wall time for all sweeps.

Files:

- `sweep.py` runs the grids (`main`, `ext`, `ext2`, `cs`) on all cores and writes resumable CSVs.
- `analyze.py` computes the crossings and their bootstrap error bars.
- `data/main.csv` and `data/cs.csv` hold the raw counts. `data/main_table.md` and `data/cs_table.md` hold the full crossing tables and every raw p_L.

## 1. What the paper does

Paper: B. Srivastava, Y. Xiao, A. Frisk Kockum, B. Criger, M. Granath, *Sequential decoding of the XYZ² hexagonal stabilizer code*, arXiv:2505.03691 (v1). arxiv.org is blocked by this container's egress proxy, so I read the HTML version (arxiv.org/html/2505.03691v1) through the web-fetch tool, which passes the page through a summarizing model. The captions and sentences quoted below came back verbatim. I could not check them against the PDF.

**Noise**

- Pauli noise on data qubits, with bias η = p_z/(p_x + p_y) (Eq. 1). Depolarizing noise is η = 1/2 and pure Z is η = ∞. η < 1/2 is not considered. The paper does not write p_x, p_y, p_z in terms of p; p_x = p_y is implied by "depolarizing ↔ η = 1/2".
- Code capacity: data errors only, perfect measurement, one round.
- Phenomenological: "we assume a syndrome-measurement error rate of q = p, where p is the data-qubit error rate, both for plaquette and link stabilizers". There are d rounds of syndrome measurement (Fig. 6 caption). The paper does not say:
  - whether the last round is perfect;
  - whether the weight-3 boundary checks get measurement errors;
  - how the memory is initialised;
  - which logical failures count toward P_f.

**Decoder (sequential)**

- First, each of the d² links is matched in the time direction over the d rounds.
- A matched pair of link events is kept as a data-error signature if p² > q^r, and removed as a measurement-error string if q^r > p².
- The links are then fixed by a set choice of vertex, and conditional error probabilities go to the upper YZZY surface code.
- The upper code is decoded by:
  - PyMatching ("matching");
  - belief-matching on the YZZY code;
  - MPS maximum likelihood, used for code capacity only.

**Statistics:** 5 × 10⁴ syndromes per point. The distances are not listed ("different distances" in every caption).

**Reported XYZ² thresholds**

| model | η | matching | belief-matching | MPS | source |
|---|---|---|---|---|---|
| code capacity | 0.5 | 18.34 ± 0.08 % | 18.52 ± 0.08 % | 18.92 ± 0.05 % | Fig. 8 |
| code capacity | 10 | 18.36 ± 0.11 % | 24.08 ± 0.05 % | 28.61 ± 0.05 % | Fig. 9 |
| phenomenological | 0.5 | 3.42 ± 0.01 % | 3.43 ± 0.01 % | – | Fig. 10 |
| phenomenological | 10 | 3.49 ± 0.02 % | 4.30 ± 0.02 % | – | Fig. 11 |

Appendix A, code-capacity matching: 18.51 ± 0.08 % (η = 0.5, Fig. 12) and 18.64 ± 0.09 % (η = 10, Fig. 13), against 18.27 % and 17.46 % for the decoder of their Ref. [47]. Fig. 8 and Fig. 12 give different values (18.34 vs 18.51) for what looks like the same setting.

**XZZX and other codes:** the paper reports no XZZX or surface-code threshold, under either noise model. Its XZZX claims are about code capacity only:

- XYZ² has "code-capacity thresholds close to those of the XZZX code under biased noise".
- Near threshold, XYZ² has lower logical failure rates than XZZX for Z-biased noise, because its pure-Z distance is 2d² instead of d (Sec. II B).
- In the conclusion, XYZ² is "comparable to the XZZX code" under code capacity.

The paper makes **no** XYZ²-vs-XZZX claim for phenomenological noise.

## 2. Differences between the paper and our `CircuitNoise.phenomenological`

Our model is `CircuitNoise.phenomenological(p, eta)`: `p_meas = p` and `data_xyz = (p_x, p_y, p_z)`, with p_z = pη/(η+1) and p_x = p_y. `FTMDRCircuit._build_phen` uses it as follows:

- the frame product state is prepared noiselessly;
- each round is `PAULI_CHANNEL_1` on every data qubit, then one noiseless MPP per check with outcome flip `p`;
- after d rounds, the data are read out noiselessly in the frame.

The XZZX phen circuits (`CompetitorCircuit._build_phen`) do the same.

| # | item | paper | ours | effect |
|---|---|---|---|---|
| 1 | rate convention, bias | p = total data rate, q = p, η = p_z/(p_x+p_y) | same; p_x = p_y | none |
| 2 | checks with measurement errors | plaquettes and links (boundaries not mentioned) | every check, including weight-3 boundaries | probably none or tiny |
| 3 | start | presumably a memory from a code state (not stated) | MDR: start from a product frame state, so round 0 has only the frame-deterministic S₀ detectors | tested with `*_cs` (extra noiseless first round): no change beyond the error bars, see §3.3 |
| 4 | last round | not stated | d noisy rounds, then a noiseless data readout (an effectively perfect round d+1) | small at d = rounds |
| 5 | logical failure | not stated (MPS/ML suggests "any logical") | one observable per run: XYZ² Logical X or Logical Y; XZZX "X" or "Z" basis. We take the weaker basis | an "any-logical" threshold is ≈ the weaker basis; our two bases differ by 0.3–0.6 pp |
| 6 | decoder | sequential: hard decisions on time-matched links (p² vs q^r), then matching/BM on the YZZY code | global DEM decoders: `mwpm` (matching on the S₀ graph only), `bm`/`bp_full` (BP on the full DEM incl. links, then matching), plus `seq_match` (link matching → erasures, our closest analogue of the paper) and `corr_links` | main source of differences, see §4 |
| 7 | shots, distances | 5 × 10⁴, distances not given | 8k–40k per point, d = 3–9 (MWPM to 11) | our error bars are larger; finite-size drift is visible |
| 8 | schedule | – | DEPTH6 and BOTH_BASES give identical phen circuits (checked); BOTH_BASES used | none |

## 3. Our crossings

rounds = d throughout. Each cell is the crossing of consecutive distances, in % of p. The error bar is half the 16–84 % range of a parametric bootstrap. "> x" / "< x" means no sign change inside the scanned window. "unresolved" means the curves overlap within noise over the whole window. The full table, with FSS fits and raw p_L, is in `data/main_table.md`.

### 3.1 XYZ² (FTMDRCircuit, frame start)

| noise | basis | decoder | 3/5 | 5/7 | 7/9 | 9/11 |
|---|---|---|---|---|---|---|
| phen (η=0.5) | X | mwpm | 2.85 ± 0.06 | 3.03 ± 0.05 | 3.16 ± 0.08 | 3.19 ± 0.08 |
| | X | seq_match | 3.04 ± 0.10 | 3.58 ± 0.11 | | |
| | X | corr_links | 3.55 ± 0.09 | 3.91 ± 0.09 | | |
| | X | bm | 3.74 ± 0.07 | 3.94 ± 0.07 | > 4.20 (d=9 ≤ d=7 at 3.6–4.2, within ~1σ) | |
| | X | bp_full | 3.73 ± 0.17 | 3.77 ± 0.09 | | |
| | Y | mwpm | 3.61 ± 0.04 | 3.45 ± 0.05 | 3.32 ± 0.06 | 3.33 ± 0.07 |
| | Y | seq_match | 4.17 ± 0.09 | 3.72 ± 0.07 | | |
| | Y | bm | > 5.00 | 4.35 ± 0.12 | | |
| | Y | bp_full | 4.76 ± 0.08 | 4.22 ± 0.10 | | |
| phen_b10 (η=10) | X | mwpm | 3.05 ± 0.05 | 3.06 ± 0.05 | 3.11 ± 0.07 | 3.14 ± 0.04 |
| | X | seq_match | 3.52 ± 0.09 | 3.57 ± 0.11 | | |
| | X | corr_links | 4.17 ± 0.08 | 4.05 ± 0.08 | | |
| | X | bm | 5.42 ± 0.06 | 5.55 ± 0.09 | 5.48 ± 0.20 | |
| | X | bp_full | 5.43 ± 0.09 | 5.66 ± 0.30 | | |
| | Y | mwpm | 3.50 ± 0.04 | 3.28 ± 0.04 | 3.25 ± 0.05 | 3.27 ± 0.12 |
| | Y | seq_match | 3.85 ± 0.09 | 3.80 ± 0.07 | | |
| | Y | bm | 6.04 ± 0.08 | 6.01 ± 0.08 | | |
| | Y | bp_full | > 6.20 | 6.08 ± 0.28 | | |

### 3.2 XZZX (rotated, CompetitorCircuit)

| noise | basis | decoder | 3/5 | 5/7 | 7/9 | 9/11 |
|---|---|---|---|---|---|---|
| phen | X | mwpm | 3.27 ± 0.12 | 3.72 ± 0.05 | 3.80 ± 0.05 | 3.79 ± 0.10 |
| | Z | mwpm | 3.30 ± 0.13 | 3.70 ± 0.06 | 3.73 ± 0.06 | 3.91 ± 0.07 |
| | X | bm | 4.35 ± 0.08 | 4.54 ± 0.11 | 4.54 ± 0.09 | |
| | Z | bm | 4.40 ± 0.10 | 4.46 ± 0.06 | | |
| phen_b10 | X | mwpm | 6.04 ± 0.13 | 5.79 ± 0.09 | 5.60 ± 0.09 | 5.48 ± 0.13 |
| | Z | mwpm | < 4.60 | 5.07 ± 0.12 | 5.23 ± 0.09 | 5.34 ± 0.15 |
| | X | bm | > 6.20 | 5.75 ± 0.11 | unresolved (d=7, 9 overlap at 5.4–6.2) | |
| | Z | bm | 4.63 ± 0.28 | < 4.60 (≈ equal at 4.6) | 5.37 ± 0.33 | |

Under η = 10, the two XZZX bases drift towards each other as d grows: X goes down from 6.0 to 5.5 and Z goes up from < 4.6 to 5.3. Both point to an asymptotic value near 5.4 %.

### 3.3 Paper-style memory from a code state (`phen_cs`, `phen_b10_cs`; d = 3, 5, 7)

| noise | code / basis | mwpm 5/7 | bm 5/7 |
|---|---|---|---|
| η=0.5 | XYZ² X | 3.11 ± 0.05 | 4.06 ± 0.16 |
| | XYZ² Y | 3.38 ± 0.05 | 4.20 ± 0.12 |
| | XZZX X / Z | 3.75 ± 0.06 / 3.69 ± 0.07 | > 4.40 / 4.32 ± 0.13 |
| η=10 | XYZ² X | 3.13 ± 0.05 | 5.28 ± 0.15 |
| | XYZ² Y | 3.32 ± 0.05 | 5.44 ± 0.09 |
| | XZZX X / Z | 5.79 ± 0.16 / 4.99 ± 0.12 | 5.79 ± 0.22 / 5.17 ± 0.54 |

These agree with the frame-start runs within about 2σ. The way the circuit starts does not change any conclusion.

### 3.4 Summary per code (weaker basis, largest resolved pair)

| noise | XYZ² mwpm | XYZ² seq_match | XYZ² best (bm / bp_full) | XZZX mwpm | XZZX bm | paper XYZ² (matching / BM) |
|---|---|---|---|---|---|---|
| η = 0.5 | 3.19 ± 0.08 (X, 9/11) | 3.58 ± 0.11 (X, 5/7) | 3.94 ± 0.07 (X bm, 5/7); 7/9 > 4.2 | 3.79–3.91 (9/11) | 4.46–4.54 | 3.42 ± 0.01 / 3.43 ± 0.01 |
| η = 10 | 3.14 ± 0.04 (X, 9/11) | 3.57 ± 0.11 (X, 5/7) | 5.48 ± 0.20 (X bm, 7/9) | 5.34 ± 0.15 (Z, 9/11) | 5.37 ± 0.33 (Z, 7/9) | 3.49 ± 0.02 / 4.30 ± 0.02 |

## 4. Verdict

**Reproduction of the paper's numbers**

- **Matching.** The paper's decoder (hard link decisions, then matching on the YZZY code) is closest to our `seq_match`. That decoder gives 3.5–3.6 % at η = 10, which matches the paper's 3.49 %. At η = 0.5 it gives 3.0–3.6 %, drifting upwards with d, against the paper's 3.42 %. So the matching numbers are reproduced to within the finite-size drift at d ≤ 7.
  - Our plain `mwpm` (S₀ graph, no link information) is lower: about 3.2 % in the weaker X basis and 3.3 % in Y, at both biases.
  - That bias barely moves the matching threshold (3.42 → 3.49 % in the paper; 3.19 → 3.14 % for our X-basis MWPM) is reproduced qualitatively.
- **Belief-matching.** Not reproduced as a number, for a known reason. Our `bm`/`bp_full` runs BP on the *full* detector error model, including link detectors and data–measurement correlations, before matching. The paper runs BM only on the upper YZZY code, after hard link decisions. Our versions are stronger:
  - η = 0.5: 3.8–3.9 % against 3.43 %;
  - η = 10: 5.4–5.5 % against 4.30 %.
  - The pattern is the same as in the paper: BM gains a lot over matching under bias and little at η = 0.5. Our `corr_links` (4.0–4.2 % at η = 10) lands close to the paper's BM value.
- Our pilot numbers are confirmed, and narrowed:
  - XYZ² phen: 3.75–3.94 % for the best decoder, against the pilot's 3.6–4.0 %.
  - XYZ² phen_b10 with BM: 5.42–5.55 %, against the pilot's 5.4–5.7 %.
  - XZZX phen_b10 with MWPM: 5.3–5.5 % at d = 9–11, against the pilot's 5.35 %.
  - One correction to the pilot: under phen, the surface code with MWPM gives 3.7–3.9 %, and with BM 4.4–4.5 %.

**Does XYZ² beat XZZX?**

- The paper does not claim this under phenomenological noise. Its XZZX statements are code-capacity only.
- In our phenomenological implementation, with each code on its best decoder and counting the weaker basis:
  - **η = 0.5:** XZZX wins clearly, 4.5 % (BM) against 3.9–4.1 % for XYZ² (BM). It also wins with MWPM, 3.8 % against 3.2 %.
  - **η = 10:** a tie within error bars. XYZ² BM gives 5.48 ± 0.20 %; XZZX gives 5.34 ± 0.15 % (MWPM, d = 9/11) and 5.37 ± 0.33 % (BM, d = 7/9). XYZ² only edges ahead in its stronger Y basis (6.0 %). With matching only, XZZX wins at η = 10 by a wide margin (5.3 % against 3.1–3.6 %).
- Only η = 0.5 and η = 10 were run. A threshold advantage for XYZ² is not established at either bias. At η = 10, XYZ² with full-DEM belief-matching reaches parity with XZZX. Larger biases (η = 100 or ∞), where the paper's pure-Z distance argument (2d² vs d) would matter most, were not tested in this run.

**Caveats**

- Small distances: d ≤ 9 for BM, d ≤ 11 for MWPM.
- The XZZX crossings under bias drift by about 0.5 pp between d = 3/5 and d = 9/11.
- The XYZ² X-basis BM 7/9 crossing at η = 0.5 is only bounded (> 4.2 %, within about 1σ).
- 8k–40k shots per point, against the paper's 50k.
- The paper's figures could not be read as images. Its numbers come from the text and captions of the HTML version.
