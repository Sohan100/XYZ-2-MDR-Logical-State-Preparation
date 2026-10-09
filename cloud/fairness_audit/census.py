"""Task 2a/2b/2c (structure): noise census, detector counts, distances, DEM identity."""
import json
import sys

from common import build, census, ops_census

P = 1e-3
out = {"census": {}, "structure": [], "dem_identity": []}
for d in (3, 5, 7):
    for basis in ("X", "Z"):
        for kind in ("css", "xzzx", "stim"):
            c = build(kind, d, d, P, basis)
            row = {"kind": kind, "d": d, "basis": basis, "qubits": c.num_qubits,
                   "detectors": c.num_detectors, "observables": c.num_observables,
                   "ticks": c.num_ticks, "ops": ops_census(c)}
            dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
            row["dem_errors"] = dem.num_errors
            row["graphlike_distance"] = len(c.shortest_graphlike_error())
            maxw = 7 if d <= 5 else 4
            if d <= 5:
                row["fault_distance"] = len(c.search_for_undetectable_logical_errors(
                    dont_explore_detection_event_sets_with_size_above=maxw,
                    dont_explore_edges_with_degree_above=maxw,
                    dont_explore_edges_increasing_symptom_degree=False,
                    canonicalize_circuit_errors=True))
            row["dem_total_prob"] = sum(i.args_copy()[0] for i in dem.flattened() if i.type == "error")
            out["structure"].append(row)
            if d == 5:
                out["census"][f"{kind}_{basis}"] = {f"{a}|{b}": v for (a, b), v in census(c).items()}
            print(json.dumps({k: v for k, v in row.items() if k != "ops"}), flush=True)
        # XZZX vs CSS: identical detector error models under sd6?
        dc = build("css", d, d, P, basis).detector_error_model(approximate_disjoint_errors=True)
        dx = build("xzzx", d, d, P, basis).detector_error_model(approximate_disjoint_errors=True)
        out["dem_identity"].append({"d": d, "basis": basis, "identical": str(dc) == str(dx),
                                    "approx_equal": dc.approx_equals(dx, atol=1e-12)})
        print(out["dem_identity"][-1], flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "census.json", "w"), indent=1)
