import sys, time, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE
import mdr.ft.competitor_circuits as cc
from mdr.ft.competitor_decoders import competitor_decoder
class RectLayout(cc.RotatedSurfaceLayout):
    def __init__(self, code, dx, dy):
        self.code=code; self.dx=dx; self.dy=dy; self.d=max(dx,dy)
        self.n=dx*dy
        self.coords=[(q%dx, q//dx) for q in range(self.n)]
        index={xy:q for q,xy in enumerate(self.coords)}
        self.checks=[]
        for y in range(-1,dy):
            for x in range(-1,dx):
                t="X" if (x+y)%2==0 else "Z"
                inside={c:(x+ddx,y+ddy) for c,(ddx,ddy) in cc._CORNERS.items() if (x+ddx,y+ddy) in index}
                if len(inside)==4: kind="bulk"
                elif len(inside)==2 and ((y in (-1,dy-1) and t=="X") or (x in (-1,dx-1) and t=="Z")): kind="boundary"
                else: continue
                terms,layer=[],{}
                for k,c in enumerate(cc.SCHEDULE[t]):
                    if c in inside:
                        q=index[inside[c]]; terms.append((cc.code_pauli(code,*inside[c],t),q)); layer[q]=k
                cat=[index[inside[c]] for c in cc.CAT_STAR[t] if c in inside]
                self.checks.append({"css_type":t,"cell":(x,y),"center":(x+0.5,y+0.5),"kind":kind,"terms":terms,"layer":layer,"cat":cat,"color":(x%2)+2*(y%2)})
        assert len(self.checks)==dx*dy-1, len(self.checks)
    def logical(self, basis):
        dx,dy=self.dx,self.dy
        pts=[(0,y) for y in range(dy)] if basis=="X" else [(x,0) for x in range(dx)]
        return [(cc.code_pauli(self.code,x,y,basis), y*dx+x) for x,y in pts]
code,basis,dec,dx,dy,p,budget,model,r=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),sys.argv[8],int(sys.argv[9])
noise=NOISE[model](p)
cc.RotatedSurfaceLayout=lambda code_,d_: RectLayout(code_,dx,dy)
t0=time.time()
C=cc.CompetitorCircuit(code,max(dx,dy),r,noise,basis=basis)
c=C.build()
dem=c.detector_error_model(decompose_errors=False)
# graphlike-free distance estimate
try:
    sh=c.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=4,dont_explore_edges_with_degree_above=4,dont_explore_edges_increasing_symptom_degree=False,canonicalize_circuit_errors=True)
    fd=len(sh)
except Exception as e:
    fd=f'?({e.__class__.__name__})'
D=competitor_decoder(c,dec)
smp=c.compile_detector_sampler()
shots=err=0; n=256
while time.time()-t0<budget and err<1500:
    tb=time.time(); dets,obs=smp.sample(n,separate_observables=True)
    err+=int(np.sum(np.any(D.decode_batch(dets)!=obs,axis=1))); shots+=n
    n=min(n*2,262144) if time.time()-tb<3 else n
pl=err/shots; se=(pl*(1-pl)/shots)**.5
print(f"{model} {code} rect {dx}x{dy} {basis} {dec} r={r} nq={c.num_qubits} fd={fd} p={p} shots={shots} err={err} pL={pl:.3e}+-{se:.1e} t={time.time()-t0:.0f}s",flush=True)
