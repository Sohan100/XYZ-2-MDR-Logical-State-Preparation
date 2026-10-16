import sys
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
import stim, networkx as nx
from networkx.algorithms import isomorphism as iso
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.ft_mdr_circuit import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit

def graph(c):
    g = nx.Graph()
    for inst in c.flattened():
        if inst.name in ("CX","CY","CZ","XCX","XCY","XCZ","YCX","YCY","YCZ","CNOT","ZCX","ZCY","ZCZ"):
            t = [x.value for x in inst.targets_copy()]
            for a,b in zip(t[::2], t[1::2]):
                g.add_edge(a,b)
    return g

def stats(c, name):
    g = graph(c)
    used = set()
    for inst in c.flattened():
        for t in inst.targets_copy():
            if t.is_qubit_target: used.add(t.value)
    degs = sorted(dict(g.degree()).values())
    gates = set(i.name for i in c.flattened() if len(i.targets_copy())>=2 and i.name in ("CX","CY","CZ","XCZ","YCZ"))
    print(name, "qubits", len(used), "graph nodes", g.number_of_nodes(), "edges", g.number_of_edges(), "maxdeg", max(degs), "gate types", sorted(gates))
    return g

n = CircuitNoise.uniform(1e-3)
gx = {}
for d in (3,5):
    c = FTMDRCircuit(d, 1, n, detectors="combined").build()
    gx[d] = stats(c, f"xyz2 d={d}")
gs = {}
for code in ("xzzx","css"):
    for d in (3,):
        c = competitor_circuit(code, d, 1, n, basis="Z")
        gs[(code,d)] = stats(c, f"{code} d={d}")
for (code,d),g in gs.items():
    for D,G in gx.items():
        m = iso.GraphMatcher(G, g)
        print(code, d, "into xyz2", D, "monomorphic:", m.subgraph_is_monomorphic())
# hybrid qubit count
c = FTMDRCircuit(5, 1, CircuitNoise.hybrid(1e-3), detectors="combined").build()
print("hyb d=5 num_qubits", c.num_qubits)
