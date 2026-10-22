# Pilot runs of the usefulness consensus

Scripts and outputs that the consensus agents ran (rounds 1 to 8) to check claims in `docs/xyz2_usefulness.md`. Most are short, statistical-error-only pilots; the claims say which numbers come from here. Paths in the claims of the form `<scratchpad>/x/y` refer to `x/y` in this folder.

Known issues found by the last review round (not fixed):

- docs/data/campaign/usefulness_pilots/ul/b10_xyz2_X.txt and ul/b10_xyz2_Y.txt: the two files report identical results for different bases (shots 25,400, errors 55, eps_round 1.446e-4) and have no progress lines. One is probably a copy or mislabelled. Both feed C31's 2% pools.
- docs/data/campaign/usefulness_pilots/r6om/mom.py line 12: the XYZ^2-BM key test `('xyz2' in key) and ("'bm'" in key ...)` misses the sets keyed 'biased10@both-X' and 'biased10@both-Y' (r6om/homog.out lines 41-42), which are XYZ^2 BM runs. With them included the result is 37 sets, chi2 74.4 on 64, W 2226, delta 0.068. The script prints 0.074 over 35 sets.
- docs/data/campaign/usefulness_pilots/claim_C9.txt and claim_C10.txt: these are older snapshots. claim_C10.txt line 94 records runs (1.696e-3, 1.512e-4, 1.264e-4) that the current C10 treats as unrecorded, and claim_C9.txt still says 'Reviewer runs give 9.8-14.3x'. Refresh them so the archive matches the final claims.
