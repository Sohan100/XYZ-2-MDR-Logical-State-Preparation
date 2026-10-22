import sys, runpy
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE
import mdr.ft.competitor_circuits as cc
orig=cc.RotatedSurfaceLayout
src=open(R+'/docs/data/campaign/usefulness_pilots/r6rev4/rect.py').read()
cls_src=src[src.index('class RectLayout'):src.index('code,basis,dec,dx')]
ns={'cc':cc}; exec(cls_src, ns); Rect=ns['RectLayout']
for b in 'XZ':
    a=cc.CompetitorCircuit('css',4,1,NOISE['si1000'](1e-3),basis=b).build()
    cc.RotatedSurfaceLayout=lambda code_,d_: Rect(code_,4,4)
    c=cc.CompetitorCircuit('css',4,1,NOISE['si1000'](1e-3),basis=b).build()
    cc.RotatedSurfaceLayout=orig
    print(b, 'identical' if str(a)==str(c) else 'differ', a.num_qubits, c.num_qubits, len(a.shortest_graphlike_error()) if False else '')
    if str(a)!=str(c):
        da=a.detector_error_model(decompose_errors=True); dc=c.detector_error_model(decompose_errors=True)
        print(' dem equal:', da==dc, ' gl distances', len(da.shortest_graphlike_error()), len(dc.shortest_graphlike_error()))
cc.RotatedSurfaceLayout=lambda code_,d_: Rect(code_,6,3)
c=cc.CompetitorCircuit('css',6,1,NOISE['si1000'](1e-3),basis='X').build()
print('6x3 X nq',c.num_qubits,'graphlike distance',len(c.shortest_graphlike_error()))
