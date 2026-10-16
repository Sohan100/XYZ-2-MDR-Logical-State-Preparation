import sys, time
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from run_decoder_threshold_sweep import NOISE
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
t0=time.time()
for code, basis in [('xzzx','X'),('xzzx','Z'),('css','Z'),('css','X')]:
    for p in [0.04, 0.08, 0.14, 0.2, 0.3]:
      out=[]
      for d in [5,9]:
        nz = NOISE['hw_tr'](p)
        circ = competitor_circuit(code, d, d, nz, basis=basis); D = competitor_decoder(circ, 'mwpm')
        smp = circ.compile_detector_sampler(seed=2)
        n=6000
        dets, obs = smp.sample(n, separate_observables=True)
        e = int(np.sum(np.any(D.decode_batch(dets) != obs, axis=1)))
        out.append((d, round(e/n,4)))
      print(code, basis, p, out, f"{time.time()-t0:.0f}s", flush=True)
