import sys, time, math, resource
import numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE, DECODERS, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
code,basis,dec,d,p,budget=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])
iters=int(sys.argv[7]) if len(sys.argv)>7 else None
model='phen_b10'; noise=NOISE[model](p); r=3*d; t0=time.time()
if code=='xyz2':
    ft=FTMDRCircuit(d,r,noise,final='frame',detectors='combined',schedule=SCHEDULES['both'],logical=basis)
    kw=dict(DECODERS[dec]); 
    if iters: kw['bp_iters']=iters
    D=TwoLevelDecoder(ft,**kw); c=D.circuit
else:
    c=competitor_circuit(code,d,r,noise,basis=basis)
    D=competitor_decoder(c,dec,**({'bp_iters':iters} if iters else {}))
smp=c.compile_detector_sampler(seed=12345+d+int(p*1e4)); shots=err=0; n=64
while time.time()-t0<budget:
    dets,obs=smp.sample(n,separate_observables=True)
    err+=int(np.sum(np.any(D.decode_batch(dets)!=obs,axis=1))); shots+=n; n=min(n*2,1024)
pl=err/shots; eps=0.5*(1-(1-2*pl)**(1/r))
print(f"{model} {code} {basis} {dec} iters={iters} d={d} r={r} nq={c.num_qubits} p={p} shots={shots} err={err} pL={pl:.3e} eps={eps:.3e} t={time.time()-t0:.0f}s rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}GB",flush=True)
