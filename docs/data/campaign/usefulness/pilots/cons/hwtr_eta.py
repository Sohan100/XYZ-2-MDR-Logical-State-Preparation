"""hw_tr equal-qubit probe: python hwtr.py code basis d p dec budget"""
import sys, time, os, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from run_decoder_threshold_sweep import DECODERS as DEC, SCHEDULES, NOISE
code, basis, d, p, dec, budget, eta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], float(sys.argv[6]), float(sys.argv[7])
r = 3 * d
from mdr.ft.circuit_noise import CircuitNoise
h = p / 2; nz = CircuitNoise(idle_mr_xyz=(h / (2 * (eta + 1)), h / (2 * (eta + 1)), h * eta / (eta + 1)), name='hw_tr_eta')
t0 = time.time()
if code == 'xyz2':
    ft = FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical=basis)
    D = TwoLevelDecoder(ft, **DEC[dec]); circ = D.circuit
else:
    from mdr.ft.competitor_circuits import competitor_circuit
    from mdr.ft.competitor_decoders import competitor_decoder
    circ = competitor_circuit(code, d, r, nz, basis=basis); D = competitor_decoder(circ, dec)
smp = circ.compile_detector_sampler(seed=(int(time.time() * 1000) + os.getpid() * 7919) % 2**31)
shots = errs = 0; size = 100
while True:
    tb = time.time()
    dets, obs = smp.sample(size, separate_observables=True)
    errs += int(np.sum(np.any(D.decode_batch(dets) != obs, axis=1))); shots += size
    dt = time.time() - tb; left = budget - (time.time() - t0)
    if left < 2: break
    size = int(max(50, min(size * 2, size / max(dt, 1e-3) * min(left * 0.8, 15))))
pl = errs / shots
eps = 0.5 * (1 - (1 - 2 * min(pl, 0.4999)) ** (1 / r))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"{code} {basis} d={d} nq={circ.num_qubits} p={p} hw_tr eta={eta} {dec} r={r} shots={shots} errors={errs} pL={pl:.3e} eps_round={eps:.3e} t={time.time()-t0:.0f}s rssGB={rss:.2f}", flush=True)
