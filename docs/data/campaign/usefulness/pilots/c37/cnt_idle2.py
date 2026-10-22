import sys
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft import FTMDRCircuit
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
# mixg-style: readout-window channel and gate-layer channel with distinct args
pT=0.04; q=0.02*pT
G=(q/42,q/42,20*q/21); M=(pT/84,pT/84,10*pT/21)
def count(c):
    n={'G':0,'M':0}
    for inst in c.flattened():
        if inst.name=='PAULI_CHANNEL_1':
            a=tuple(inst.gate_args_copy())
            k='G' if abs(a[2]-G[2])<1e-12 else ('M' if abs(a[2]-M[2])<1e-12 else None)
            if k: n[k]+=len(inst.targets_copy())
    return n
nz=CircuitNoise(p1=0,p2=0,p_meas=0,idle_mr_xyz=M,idle_xyz=G,name='mixg')
def bx(r):
    ft=FTMDRCircuit(5,r,nz,final='frame',detectors='combined',schedule=SCHEDULES['both'],logical='X')
    return ft.circuit if hasattr(ft,'circuit') else ft.build()
a=[count(bx(r)) for r in (15,16)]
b=[count(competitor_circuit('xzzx',7,r,nz,basis='X')) for r in (21,22)]
print('xyz2 d5 per round', {k:a[1][k]-a[0][k] for k in a[0]}, 'xzzx d7 per round', {k:b[1][k]-b[0][k] for k in b[0]})
