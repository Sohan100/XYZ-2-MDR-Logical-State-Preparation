import sys, collections
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0, R+'/src'); sys.path.insert(0, R+'/scripts')
from mdr.ft import FTMDRCircuit
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.competitor_circuits import competitor_circuit
nz=CircuitNoise.trapped_ion(1e-3,'helios',crosstalk=True)
pm=nz.p_mem_z
def count(c):
    k=collections.Counter(); g2=0
    for inst in c.flattened():
        if inst.name=='Z_ERROR' and abs(inst.gate_args_copy()[0]-pm)<1e-12:
            for t in inst.targets_copy(): k[t.value]+=1
        if inst.name in ('CX','CY','CZ','CNOT','SQRT_ZZ','ZCX','ZCY','ZCZ'): g2+=len(inst.targets_copy())//2
    return collections.Counter(k.values()), g2
ft=FTMDRCircuit(5,1,nz,final='frame',detectors='combined',logical='X')
c=ft.circuit if hasattr(ft,'circuit') else ft.build()
print('xyz2 d5', count(c), c.num_qubits)
for code in ('css','xzzx'):
    c=competitor_circuit(code,7,1,nz,basis='X'); print(code, count(c), c.num_qubits)
