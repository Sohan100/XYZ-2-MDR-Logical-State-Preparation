"""MWPM runs of the Quantinuum models (CircuitNoise.quantinuum) with sinter.

Samples Helios and H2 with one round and with r = d rounds over a range of
p/p0, the two-qubit error rate in units of the device point p0 (1e-3 for
Helios, 1.875e-3 for H2; `lam` in the sinter metadata), the Helios model with
cycle-benchmarked gate noise at p = p0, and Helios with r = 2..7 rounds at
p = p0. helios_plots.py converts `lam` to p. Run from the repository root;
results go to docs/data/helios/helios_mwpm_sinter.csv (resumable).
"""
import copy, sys
import sinter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from mdr.ft import CircuitNoise, FTMDRCircuit

def model(machine, lam):
    if machine == "helios_cb":
        return CircuitNoise.quantinuum("helios", lam, two_qubit="cb")
    return CircuitNoise.quantinuum(machine, lam)

def tasks():
    plan = []
    for machine in ("helios", "h2"):
        plan += [(machine, d, "1", lam) for d in (3, 5, 7) for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
        plan += [(machine, d, "d", lam) for d in (3, 5, 7, 9) for lam in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    plan += [("helios_cb", d, rm, 1.0) for d in (3, 5, 7) for rm in ("1", "d")]
    plan += [("helios", d, str(r), 1.0) for d in (3, 5, 7) for r in (2, 3, 4, 5, 6, 7)]
    bases = {}
    for machine, d, rm, lam in plan:
        if d not in bases:
            bases[d] = FTMDRCircuit(d, 1, CircuitNoise.quantinuum("helios", 1.0))
        ft = copy.copy(bases[d])
        ft.rounds = d if rm == "d" else int(rm)
        ft.noise = model(machine, lam)
        yield sinter.Task(circuit=ft.build(), json_metadata=dict(machine=machine, d=d, r=ft.rounds, rmode=rm, lam=lam))

if __name__ == "__main__":
    t = list(tasks())
    print(len(t), "tasks", flush=True)
    sinter.collect(num_workers=1, tasks=t, decoders=["pymatching"], max_shots=4_000_000, max_errors=1000,
                   save_resume_filepath="docs/data/helios/helios_mwpm_sinter.csv", print_progress=False)
    print("done", flush=True)
