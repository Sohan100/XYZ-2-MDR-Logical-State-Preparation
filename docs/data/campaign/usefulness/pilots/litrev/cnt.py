import sys, collections
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft import FTMDRCircuit
from run_decoder_threshold_sweep import NOISE, SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
nz = NOISE['hw_tr'](0.06)
def tally(c, r):
    tot = collections.Counter(); kinds=collections.Counter()
    for inst in c.flattened():
        if inst.name == 'PAULI_CHANNEL_1':
            a = inst.gate_args_copy()
            for t in inst.targets_copy(): tot[t.value] += sum(a)
        kinds[inst.name]+=1
    vals = sorted(tot.values())
    return len(tot), vals[0]/r, vals[-1]/r, sum(vals)/len(vals)/r, {k:v for k,v in kinds.items() if 'PAULI' in k or 'DEPOL' in k or 'X_ERROR' in k}
for d in (3,5):
    r = 3*d
    ft = FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X')
    c = ft.circuit if hasattr(ft,'circuit') else ft.build()
    print('xyz2', d, c.num_qubits, tally(c, r))
    for code in ('xzzx','css'):
        cc = competitor_circuit(code, d+2 if d==5 else d, r, nz, basis='X')
        print(code, cc.num_qubits, tally(cc, r))
