# Documentation

Start with [`xyz2_usefulness.md`](xyz2_usefulness.md). It is the final answer to the question this repository asks: when, if ever, is the XYZ² code more useful than the surface codes and the honeycomb Floquet code?

For the paper rewrite, [`paper/REWRITE_BRIEF.md`](../paper/REWRITE_BRIEF.md) collects every new result, what it means for the paper's claims, and the LaTeX patches in one place.

## Protocol and decoders

| document | contents |
|---|---|
| [`ft_mdr_protocol.md`](ft_mdr_protocol.md) | The fault-tolerant MDR preparation of the logical plus state: frame initialization, depth-6 schedule, fault distance d + 1 |
| [`ft_mdr_decoders.md`](ft_mdr_decoders.md) | The two-level decoders (matching, correlated matching, belief-matching, hierarchical modes), the circuit noise models and the threshold table |
| [`ft_mdr_section.tex`](ft_mdr_section.tex) | Early draft of the protocol section, superseded by the manuscript; kept for history |

## Comparison with other codes

| document | contents |
|---|---|
| [`fair_comparison.md`](fair_comparison.md) | The rules every comparison follows: noise per operation, hardware variants, decoders, logical bases, what counts as a threshold |
| [`noncss_comparison.md`](noncss_comparison.md) | XYZ² against the CSS, XZZX and XY surface codes and the honeycomb code under every noise model, with the literature values |
| [`hetenyi_wootton_check.md`](hetenyi_wootton_check.md) | The four corners of the spin-qubit model of Hetényi and Wootton (arXiv:2306.17786) in our circuits, and why the rankings differ |
| [`gap_diagnostic.md`](gap_diagnostic.md) | Where the threshold gap between XYZ² and XZZX comes from (link noise, idling) |
| [`equal_qubit_comparison.md`](equal_qubit_comparison.md) | Logical error per round below threshold at equal qubit count, Λ, and qubits needed for 10⁻⁶ and 10⁻⁹ per round |
| [`xyz2_usefulness.md`](xyz2_usefulness.md) | Consensus of eight review rounds (10 reviewers each) on where XYZ² is useful, with the open disputes and the agreement record |

## Running the campaign

[`nersc_campaign.md`](nersc_campaign.md) is the runbook for the threshold campaign on Perlmutter (work pool, checkpointing, merging and fitting).

## Data (`docs/data/`)

| path | contents |
|---|---|
| `campaign/thresholds.csv`, `thresholds.md` | Every XYZ² threshold: decoder, schedule, basis, rounds r = 1 to 21 and r = d |
| `campaign/competitors*.csv`, `competitors.md` | Head-to-head tables against the other codes |
| `campaign/coverage*` | Which series reached which distances |
| `campaign/points.csv.gz` | Merged raw counts of the campaign |
| `campaign/hw/` | Hetényi–Wootton corners: points, thresholds, figures |
| `campaign/gap/` | Gap diagnostic: points, thresholds, gap fractions, pgfplots figure |
| `campaign/eq/` | Equal-qubit comparison: points, error per round, Λ, qubit targets, figure |
| `campaign/usefulness/` | The consensus: final state (`consensus_final.json`), earlier rounds (`rounds/`) and every pilot the reviewers ran (`pilots/`) |
| `thresholds/`, `rounds_threshold/`, `cfe/`, `helios/`, `res_*.csv` | Earlier sweeps behind the paper figures |

The per-chunk raw counts are not in git (`data/campaign/` is ignored). They are mirrored at `/global/cfs/cdirs/m4980/ftmdr_campaign/` on NERSC.
