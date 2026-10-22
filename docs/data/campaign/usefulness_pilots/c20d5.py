import sys, time, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from run_decoder_threshold_sweep import NOISE
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
code, basis, d, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
r = 3*d
nz = NOISE['hw_rr'](0.04)
circ = competitor_circuit(code, d, r, nz, basis=basis); D = competitor_decoder(circ, 'mwpm')
smp = circ.compile_detector_sampler(seed=12345 + d)
t0 = time.time(); shots = errs = 0
while time.time() - t0 < budget:
    dets, obs = smp.sample(20000, separate_observables=True)
    errs += int(np.sum(np.any(D.decode_batch(dets) != obs, axis=1))); shots += 20000
pl = errs/shots
print(f"hw_rr {code} {basis} d={d} nq={circ.num_qubits} r={r} mwpm shots={shots} errors={errs} pL={pl:.4e}({np.sqrt(pl*(1-pl)/shots):.1e}) t={time.time()-t0:.0f}s rssGB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}")
