import sys,re
src=open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/usefulness_pilots/r6rev4/rect.py').read()
head=src.split("code,basis,dec,dx,dy")[0]
exec(head)
orig=cc.RotatedSurfaceLayout
noise=NOISE['si1000'](0.001)
def build(dx,dy,basis):
    cc.RotatedSurfaceLayout=lambda code_,d_: RectLayout(code_,dx,dy)
    c=cc.CompetitorCircuit('css',max(dx,dy),1,noise,basis=basis).build()
    cc.RotatedSurfaceLayout=orig
    return c
for b in 'XZ':
    c=build(4,4,b); s=cc.CompetitorCircuit('css',4,1,noise,basis=b).build()
    print('4x4',b,str(c)==str(s), c.num_qubits)
c=build(6,3,'X')
print('6x3 X nq',c.num_qubits,'graphlike',len(c.shortest_graphlike_error()),
  'search',len(c.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=4,dont_explore_edges_with_degree_above=4,dont_explore_edges_increasing_symptom_degree=False,canonicalize_circuit_errors=True)))
