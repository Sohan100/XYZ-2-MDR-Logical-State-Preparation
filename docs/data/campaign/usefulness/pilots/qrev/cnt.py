import sys, collections
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft import FTMDRCircuit
from mdr.ft.circuit_noise import CircuitNoise
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
nz = CircuitNoise(p_idle=1e-3)
def per_round(build):
    out=[]
    for r in (2,3):
        c=build(r)
        cnt=collections.Counter()
        for inst in c.flattened():
            n=inst.name
            if n in ('CX','CY','CZ','XCZ','YCZ','CNOT'): cnt['2q']+=len(inst.targets_copy())//2
            elif n=='TICK': cnt['tick']+=1
            elif n=='DEPOLARIZE1': cnt['idle']+=len(inst.targets_copy())
        out.append(cnt)
    return {k: out[1][k]-out[0][k] for k in out[1]}, out[1], build(3).num_qubits
for d in (3,5,7):
    print('xyz2', d, per_round(lambda r: FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X').build() if not hasattr(FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X'),'circuit') else FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X').circuit))
for code in ('xzzx','css'):
    for d in (3,5,7):
        print(code, d, per_round(lambda r: competitor_circuit(code, d, r, nz, basis='X')))
