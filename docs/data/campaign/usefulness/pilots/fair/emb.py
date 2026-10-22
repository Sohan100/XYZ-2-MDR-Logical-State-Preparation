import sys, time
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
import networkx as nx, stim
from run_decoder_threshold_sweep import NOISE, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
def graph(c):
    G=nx.Graph(); 
    for inst in c.flattened():
        if inst.name in ('CX','CY','CZ','CNOT','XCZ','YCZ','ZCX','ZCY','ZCZ'):
            t=[x.value for x in inst.targets_copy()]
            for a,b in zip(t[::2],t[1::2]): G.add_edge(a,b)
    return G
nz=NOISE['sd6'](1e-3)
out={}
for d in (3,5):
    ft=FTMDRCircuit(d,3,nz,final='frame',detectors='combined',schedule=SCHEDULES['both'],logical='X')
    G=graph(ft.build() if hasattr(ft,'build') else ft.circuit)
    out[d]=G; degs=sorted(set(dict(G.degree()).values()))
    print('xyz2',d,'nodes',G.number_of_nodes(),'edges',G.number_of_edges(),'degrees',degs, 'ticks/round?')
for d in (3,):
    c=competitor_circuit('xzzx',d,3,nz,basis='X'); H=graph(c)
    print('xzzx',d,'nodes',H.number_of_nodes(),'edges',H.number_of_edges(),'degrees',sorted(set(dict(H.degree()).values())))
    for dx in (3,5):
        t0=time.time()
        gm=nx.algorithms.isomorphism.GraphMatcher(out[dx],H)
        print('XZZX d=3 into XYZ2 d=%d monomorphic:'%dx, gm.subgraph_is_monomorphic(), '%.1fs'%(time.time()-t0), flush=True)
