import sys
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.ft_mdr_circuit import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
def used(c, mpp_only=False):
    s=set()
    for inst in c.flattened():
        if mpp_only and inst.name not in ("MPP","M","MX","MY","R","RX","RY","MR","MRX","MRY"): continue
        for t in inst.targets_copy():
            if t.is_qubit_target or t.is_x_target or t.is_y_target or t.is_z_target: s.add(t.value)
    return s
for nm, n in (("em3", CircuitNoise.em3(1e-3)), ("hyb", CircuitNoise.hybrid(1e-3))):
    c = FTMDRCircuit(5, 2, n, detectors="combined").build()
    print(nm, "xyz2 d5 num_qubits", c.num_qubits, "touched by MPP/M/R", len(used(c, True)), "any", len(used(c)))
    c2 = competitor_circuit("css", 5, 2, n, basis="Z")
    print(nm, "css d5 num_qubits", c2.num_qubits, "touched by MPP/M/R", len(used(c2, True)), "any", len(used(c2)))
