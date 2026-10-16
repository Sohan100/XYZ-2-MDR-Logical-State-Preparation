import sys, collections
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
import stim
def count(c):
    cnt=collections.Counter(); deg=collections.defaultdict(set)
    for inst in c.flattened():
        n=inst.name
        if n in ('CX','CY','CZ','XCX','XCZ','YCX','ZCX','ZCY','ZCZ','CNOT','XCY','YCY','YCZ','SQRT_XX','SQRT_ZZ'):
            t=[q.value for q in inst.targets_copy()]
            cnt[n]+=len(t)//2
            for a,b in zip(t[::2],t[1::2]): deg[a].add(b); deg[b].add(a)
        if n=='MPP': cnt['MPP']+=1
    return cnt, max(len(v) for v in deg.values()) if deg else 0
for model in ['sd6','hyb','em3']:
  for d in (3,5):
    for r in (1,2):
        ft=FTMDRCircuit(d, r, NOISE[model](1e-3), final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='X')
        c=ft.build() if hasattr(ft,'build') else ft.circuit
        cn,dg=count(c)
        print(model,'xyz2',d,r,c.num_qubits,dict(cn),dg)
    for code in ('css','xzzx'):
      for r in (1,2):
        c=competitor_circuit(code, d if d!=5 else 5, r, NOISE[model](1e-3), basis='Z')
        cn,dg=count(c); print(model,code,d,r,c.num_qubits,dict(cn),dg)
