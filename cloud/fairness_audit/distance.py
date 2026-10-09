"""Task 2c: circuit (fault) distance of the XZZX (and CSS) memories in both bases."""
import json
import sys
import time

from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.competitor_circuits import competitor_circuit

NOISES = {"sd6": lambda p: CircuitNoise.uniform(p),
          "biased100": lambda p: CircuitNoise.biased(p, 100),
          "purez": lambda p: CircuitNoise.biased(p, 1e12)}
rows = []
for code in ("xzzx", "css"):
    for noise in NOISES:
        for d in (3, 5, 7, 9):
            for basis in ("X", "Z"):
                if code == "css" and noise != "sd6":
                    continue
                t0 = time.time()
                c = competitor_circuit(code, d, d, NOISES[noise](1e-3), basis)
                row = {"code": code, "noise": noise, "d": d, "basis": basis,
                       "graphlike": len(c.shortest_graphlike_error(
                           ignore_ungraphlike_errors=False))}
                if d <= 7:
                    lim = 6 if d <= 5 else 4
                    row["search"] = len(c.search_for_undetectable_logical_errors(
                        dont_explore_detection_event_sets_with_size_above=lim,
                        dont_explore_edges_with_degree_above=lim,
                        dont_explore_edges_increasing_symptom_degree=False,
                        canonicalize_circuit_errors=True))
                    row["search_limits"] = lim
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "distance.json", "w"), indent=1)
