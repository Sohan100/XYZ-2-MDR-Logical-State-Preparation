import sys
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft import FTMDRCircuit
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft.competitor_circuits import competitor_circuit
def count(c):
    n=0
    for inst in c.flattened():
        if inst.name=='PAULI_CHANNEL_1':
            n+=len(inst.targets_copy())
    return n
for which in ('gate','mr'):
    if which=='gate':
        nz=CircuitNoise(p1=0,p2=0,p_meas=0,idle_mr_xyz=None,idle_xyz=(1e-4,1e-4,2e-3),name='g')
    else:
        nz=CircuitNoise(p1=0,p2=0,p_meas=0,idle_mr_xyz=(1e-4,1e-4,2e-3),idle_xyz=None,name='m')
    res={}
    for r in (15,16):
        ft=FTMDRCircuit(5,r,nz,final='frame',detectors='combined',schedule=SCHEDULES['both'],logical='X')
        c=ft.circuit if hasattr(ft,'circuit') else ft.build()
        res[('xyz2',r)]=count(c)
    for r in (21,22):
        cc=competitor_circuit('xzzx',7,r,nz,basis='X')
        res[('xzzx',r)]=count(cc)
    print(which,'xyz2 per round',res[('xyz2',16)]-res[('xyz2',15)],'xzzx per round',res[('xzzx',22)]-res[('xzzx',21)])
