import sys, collections
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft import FTMDRCircuit
from mdr.ft.circuit_noise import CircuitNoise
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
def get(ft): return ft.circuit if hasattr(ft,'circuit') else ft.build()
def cnt(c):
    k=collections.Counter()
    for inst in c.flattened():
        if inst.name in ('DEPOLARIZE1','PAULI_CHANNEL_1'): k[inst.name]+=len(inst.targets_copy())
    return k
for label,nz in [('gate-layer only',CircuitNoise(p_idle=1e-3,p_idle_mr=0.0)),('mr only',CircuitNoise(p_idle=0.0,p_idle_mr=1e-3)),
                 ('mixg f>0 idle_xyz only', CircuitNoise(idle_xyz=(1e-3,1e-3,1e-3)))]:
    for code,d in (('xyz2',5),('xzzx',7)):
        vals=[]
        for r in (2,3):
            c = get(FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X')) if code=='xyz2' else competitor_circuit(code,d,r,nz,basis='X')
            vals.append(cnt(c))
        print(label, code, d, {k: vals[1][k]-vals[0][k] for k in vals[1]})
