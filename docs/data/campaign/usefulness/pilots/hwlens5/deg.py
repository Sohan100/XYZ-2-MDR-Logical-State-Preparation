import sys, collections
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
import networkx as nx
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.ft_mdr_circuit import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
TWO = {"CX","CY","CZ","XCZ","YCZ","XCX","XCY","YCX","YCY","ZCX","ZCY","ZCZ","CNOT"}
def g_of(c):
    g = nx.Graph()
    for inst in c.flattened():
        if inst.name in TWO:
            t = [x.value for x in inst.targets_copy()]
            for a,b in zip(t[::2], t[1::2]): g.add_edge(a,b)
    return g
n = CircuitNoise.uniform(1e-3)
for d in (3,5,7):
    c = FTMDRCircuit(d, 1, n, detectors="combined").build()
    g = g_of(c)
    print("xyz2", d, "nodes", g.number_of_nodes(), "edges", g.number_of_edges(), "deg hist", sorted(collections.Counter(dict(g.degree()).values()).items()))
for code in ("xzzx","css"):
    for d in (3,5,7):
        c = competitor_circuit(code, d, 1, n, basis="Z")
        g = g_of(c)
        print(code, d, "nodes", g.number_of_nodes(), "edges", g.number_of_edges(), "deg hist", sorted(collections.Counter(dict(g.degree()).values()).items()))
