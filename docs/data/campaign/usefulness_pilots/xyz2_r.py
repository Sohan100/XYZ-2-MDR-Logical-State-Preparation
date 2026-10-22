import sys, time, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from run_decoder_threshold_sweep import DECODERS as DEC, SCHEDULES, NOISE
d = int(sys.argv[1]); basis = sys.argv[2]; budget = float(sys.argv[3]); r = int(sys.argv[4])
nz = NOISE['hw_rr'](0.04)
ft = FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical=basis)
D = TwoLevelDecoder(ft, **DEC['mwpm']); circ = D.circuit
smp = circ.compile_detector_sampler(seed=4242 + r)
t0 = time.time(); shots = errs = 0
while time.time() - t0 < budget:
    dets, obs = smp.sample(20000, separate_observables=True)
    errs += int(np.sum(np.any(D.decode_batch(dets) != obs, axis=1))); shots += 20000
p = errs / shots
print(f"xyz2 {basis} d={d} nq={circ.num_qubits} r={r} mwpm shots={shots} errors={errs} pL={p:.4e}({np.sqrt(p*(1-p)/shots):.1e}) rssGB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}", flush=True)
