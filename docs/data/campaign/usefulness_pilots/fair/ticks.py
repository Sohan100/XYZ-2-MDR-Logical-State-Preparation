import sys
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
nz=NOISE['sd6'](1e-3)
def ticks(c): return sum(1 for i in c.flattened() if i.name=='TICK')
for s in ('both','depth6'):
    a=[ticks(FTMDRCircuit(3,r,nz,final='frame',detectors='combined',schedule=SCHEDULES[s],logical='X').build()) for r in (4,5)]
    print('xyz2',s,a,'per round',a[1]-a[0])
for code in ('css','xzzx'):
    a=[ticks(competitor_circuit(code,3,r,nz,basis='X')) for r in (4,5)]
    print(code,a,'per round',a[1]-a[0])
