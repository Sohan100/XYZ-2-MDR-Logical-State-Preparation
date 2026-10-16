import sys, collections
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft import FTMDRCircuit
from mdr.ft.circuit_noise import CircuitNoise
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
def get(ft): return ft.circuit if hasattr(ft,'circuit') else ft.build()
nz=CircuitNoise(p1=0.001,p2=0.001,p_meas=0.01,idle_mr_xyz=(0.0005,0.0005,0.02),idle_xyz=(0.0001,0.0001,0.002))
for code,d in (('xyz2',5),('xzzx',7)):
    vals=[]
    for r in (2,3):
        c = get(FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X')) if code=='xyz2' else competitor_circuit(code,d,r,nz,basis='X')
        k=collections.Counter()
        for inst in c.flattened():
            if inst.name=='PAULI_CHANNEL_1': k[tuple(round(x,6) for x in inst.gate_args_copy())]+=len(inst.targets_copy())
        vals.append(k)
    print(code,d,{kk: vals[1][kk]-vals[0][kk] for kk in vals[1]})
