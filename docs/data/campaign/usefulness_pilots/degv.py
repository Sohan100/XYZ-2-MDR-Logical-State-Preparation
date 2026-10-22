import sys, collections
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.ft_mdr_circuit import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
def g_of(c):
    E=set()
    for inst in c.flattened():
        if len(inst.name)<=4 and inst.name.endswith(('X','Y','Z','NOT')) and inst.name[0] in 'CXYZ' and len(inst.name)>=2 and inst.name not in ('MX','MY','MZ','RX','RY','RZ','MRX','MRY','MRZ','MPP','SQRT_X'):
            t=[x.value for x in inst.targets_copy()]
            for a,b in zip(t[::2],t[1::2]): E.add(frozenset((a,b)))
    deg=collections.Counter()
    for e in E:
        for q in e: deg[q]+=1
    return len(E), sorted(collections.Counter(deg.values()).items())
n=CircuitNoise.uniform(1e-3)
for d in (3,5):
    print('xyz2',d,g_of(FTMDRCircuit(d,1,n,detectors="combined").build()))
for code in ('css','xzzx'):
    for d in (3,5,7): print(code,d,g_of(competitor_circuit(code,d,1,n,basis="Z")))
try:
    FTMDRCircuit(4,1,n,detectors="combined").build(); print('d=4 ok?!')
except Exception as e: print('d=4:',repr(e))
