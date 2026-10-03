"""Error budget of the Helios model at lambda = 1, decoded with HCM.

Each variant removes one component of CircuitNoise.quantinuum("helios") or keeps
only one component; "xtalk_per_step" applies the crosstalk once per reset and
measurement step instead of once per measured ancilla. Run from the repository
root; rows (d, r, variant, shots, errors, rate, stderr) are appended to
docs/data/helios/helios_budget.csv.
"""
import csv, sys, time
from dataclasses import replace
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder

base = CircuitNoise.quantinuum("helios", 1.0)
VARIANTS = {
    "full": base,
    "no_gates": replace(base, p1=0.0, p2=0.0),
    "no_memory": replace(base, p_mem_z=0.0),
    "no_crosstalk": replace(base, p_xtalk=0.0),
    "no_spam": replace(base, p_prep=0.0, p_meas=0.0),
    "xtalk_per_step": replace(base, xtalk_per_mcmr=False),
    "only_crosstalk": replace(base, p1=0.0, p2=0.0, p_mem_z=0.0, p_prep=0.0, p_meas=0.0),
    "only_memory": replace(base, p1=0.0, p2=0.0, p_xtalk=0.0, p_prep=0.0, p_meas=0.0),
    "only_gates": replace(base, p_mem_z=0.0, p_xtalk=0.0, p_prep=0.0, p_meas=0.0),
    "only_spam": replace(base, p1=0.0, p2=0.0, p_mem_z=0.0, p_xtalk=0.0),
}
out = "docs/data/helios/helios_budget.csv"
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    for d in (3, 5, 7):
        for r in (1, d):
            for name, nz in VARIANTS.items():
                t0 = time.time()
                ft = FTMDRCircuit(d, r, nz, detectors="combined")
                dec = TwoLevelDecoder(ft, mode="corr_split", lower="gauge")
                est = dec.estimate(max_shots=4_000_000, max_errors=300, batch=20000, time_limit=60)
                w.writerow([d, r, name, est.shots, est.errors, est.rate, est.stderr])
                fh.flush()
                print(f"d={d} r={r} {name:15s} pL={est.rate:.3e} ({est.errors}/{est.shots}) {time.time()-t0:.0f}s", flush=True)
