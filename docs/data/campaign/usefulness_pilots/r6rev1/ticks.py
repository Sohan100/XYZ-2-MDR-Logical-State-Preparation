import sys
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft import FTMDRCircuit
from mdr.ft.circuit_noise import CircuitNoise
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
def get(ft): return ft.circuit if hasattr(ft,'circuit') else ft.build()
nz=CircuitNoise(p1=0.001,p2=0.001,p_meas=0.001)
for code,d in (('xyz2',5),('xzzx',7)):
    t=[]
    for r in (2,3):
        c = get(FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X')) if code=='xyz2' else competitor_circuit(code,d,r,nz,basis='X')
        f=c.flattened(); t.append((sum(1 for i in f if i.name=='TICK'), sum(len(i.targets_copy())//2 for i in f if i.name in ('CX','CY','CZ','CNOT','XCZ','YCZ','ZCX','ZCY')), c.num_qubits))
    print(code,d,t,'per round ticks',t[1][0]-t[0][0],'2q',t[1][1]-t[0][1])
