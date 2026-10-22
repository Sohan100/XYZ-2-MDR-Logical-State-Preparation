import sys, time, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from run_decoder_threshold_sweep import NOISE
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
print("imported",flush=True)
d=int(sys.argv[1]); r=3*d; budget=float(sys.argv[2])
circ=competitor_circuit('xzzx',d,r,NOISE['hw_rr'](0.04),basis='X'); D=competitor_decoder(circ,'mwpm')
print("built",time.time(),flush=True)
smp=circ.compile_detector_sampler(seed=12345+d)
t0=time.time(); s=e=0
while time.time()-t0<budget:
    dets,obs=smp.sample(20000,separate_observables=True)
    e+=int(np.sum(np.any(D.decode_batch(dets)!=obs,axis=1))); s+=20000
p=e/s
print(d, circ.num_qubits, s, e, p, (p*(1-p)/s)**.5, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
