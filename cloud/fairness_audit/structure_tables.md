| channel | role | css_X | stim_X | css_Z | stim_Z | xzzx_X | xzzx_Z |
|---|---|---|---|---|---|---|---|
| DEPOLARIZE1 | anc | 80 / 80 | 120 / 120 | 80 / 80 | 120 / 120 | 80 / 80 | 80 / 80 |
| DEPOLARIZE1 | data | 350 / 350 | 125 / 125 | 350 / 350 | 125 / 125 | 350 / 350 | 350 / 350 |
| DEPOLARIZE2 | pair | 400 / 400 | 400 / 400 | 400 / 400 | 400 / 400 | 400 / 400 | 400 / 400 |
| X_ERROR@reset | anc | 0 | 144 / 144 | 0 | 144 / 144 | 0 | 0 |
| X_ERROR@reset | data | 0 | 0 | 25 / 25 | 25 / 25 | 12 / 12 | 13 / 13 |
| X_ERROR | anc | 0 | 120 / 120 | 0 | 120 / 120 | 0 | 0 |
| X_ERROR | data | 0 | 0 | 25 / 25 | 25 / 25 | 12 / 12 | 13 / 13 |
| Z_ERROR@reset | anc | 120 / 120 | 0 | 120 / 120 | 0 | 120 / 120 | 120 / 120 |
| Z_ERROR@reset | data | 25 / 25 | 25 / 25 | 0 | 0 | 13 / 13 | 12 / 12 |
| Z_ERROR | anc | 120 / 120 | 0 | 120 / 120 | 0 | 120 / 120 | 120 / 120 |
| Z_ERROR | data | 25 / 25 | 25 / 25 | 0 | 0 | 13 / 13 | 12 / 12 |

| circuit | d | basis | qubits used | detectors | observables | TICKs | DEM mechanisms | ΣP(DEM) at p=1e-3 | graphlike dist | undetectable-error search |
|---|---|---|---|---|---|---|---|---|---|---|
| css | 3 | X | 17 | 24 | 1 | 19 | 300 | 0.227 | 3 | 3 |
| xzzx | 3 | X | 17 | 24 | 1 | 19 | 299 | 0.227 | 3 | 3 |
| stim | 3 | X | 17 | 24 | 1 | 21 | 291 | 0.171 | 3 | 3 |
| css | 3 | Z | 17 | 24 | 1 | 19 | 291 | 0.227 | 3 | 3 |
| xzzx | 3 | Z | 17 | 24 | 1 | 19 | 294 | 0.227 | 3 | 3 |
| stim | 3 | Z | 17 | 24 | 1 | 21 | 286 | 0.171 | 3 | 3 |
| css | 5 | X | 49 | 120 | 1 | 31 | 1980 | 1.044 | 5 | 5 |
| xzzx | 5 | X | 49 | 120 | 1 | 31 | 1978 | 1.044 | 5 | 5 |
| stim | 5 | X | 49 | 120 | 1 | 35 | 1958 | 0.860 | 5 | 5 |
| css | 5 | Z | 49 | 120 | 1 | 31 | 1967 | 1.043 | 5 | 5 |
| xzzx | 5 | Z | 49 | 120 | 1 | 31 | 1973 | 1.043 | 5 | 5 |
| stim | 5 | Z | 49 | 120 | 1 | 35 | 1953 | 0.860 | 5 | 5 |
| css | 7 | X | 97 | 336 | 1 | 43 | 6136 | 2.832 | 7 | — |
| xzzx | 7 | X | 97 | 336 | 1 | 43 | 6133 | 2.832 | 7 | — |
| stim | 7 | X | 97 | 336 | 1 | 49 | 6605 | 2.430 | 7 | — |
| css | 7 | Z | 97 | 336 | 1 | 43 | 6119 | 2.832 | 7 | — |
| xzzx | 7 | Z | 97 | 336 | 1 | 43 | 6128 | 2.832 | 7 | — |
| stim | 7 | Z | 97 | 336 | 1 | 49 | 6602 | 2.430 | 7 | — |

XZZX vs CSS undecomposed DEM text identical: True [(3, 'X'), (3, 'Z'), (5, 'X'), (5, 'Z'), (7, 'X'), (7, 'Z')]

| code | noise | d | basis | shortest graphlike error | undetectable-logical search (limits) |
|---|---|---|---|---|---|
| xzzx | sd6 | 3 | X | 3 | 3 (≤6) |
| xzzx | sd6 | 3 | Z | 3 | 3 (≤6) |
| xzzx | sd6 | 5 | X | 5 | 5 (≤6) |
| xzzx | sd6 | 5 | Z | 5 | 5 (≤6) |
| xzzx | sd6 | 7 | X | 7 | 7 (≤4) |
| xzzx | sd6 | 7 | Z | 7 | 7 (≤4) |
| xzzx | sd6 | 9 | X | 9 | — |
| xzzx | sd6 | 9 | Z | 9 | — |
| xzzx | biased100 | 3 | X | 3 | 3 (≤6) |
| xzzx | biased100 | 3 | Z | 3 | 3 (≤6) |
| xzzx | biased100 | 5 | X | 5 | 5 (≤6) |
| xzzx | biased100 | 5 | Z | 5 | 5 (≤6) |
| xzzx | biased100 | 7 | X | 7 | 7 (≤4) |
| xzzx | biased100 | 7 | Z | 7 | 7 (≤4) |
| xzzx | biased100 | 9 | X | 9 | — |
| xzzx | biased100 | 9 | Z | 9 | — |
| xzzx | purez | 3 | X | 3 | 3 (≤6) |
| xzzx | purez | 3 | Z | 3 | 3 (≤6) |
| xzzx | purez | 5 | X | 5 | 5 (≤6) |
| xzzx | purez | 5 | Z | 5 | 5 (≤6) |
| xzzx | purez | 7 | X | 7 | 7 (≤4) |
| xzzx | purez | 7 | Z | 7 | 7 (≤4) |
| xzzx | purez | 9 | X | 9 | — |
| xzzx | purez | 9 | Z | 9 | — |
| css | sd6 | 3 | X | 3 | 3 (≤6) |
| css | sd6 | 3 | Z | 3 | 3 (≤6) |
| css | sd6 | 5 | X | 5 | 5 (≤6) |
| css | sd6 | 5 | Z | 5 | 5 (≤6) |
| css | sd6 | 7 | X | 7 | 7 (≤4) |
| css | sd6 | 7 | Z | 7 | 7 (≤4) |
| css | sd6 | 9 | X | 9 | — |
| css | sd6 | 9 | Z | 9 | — |
