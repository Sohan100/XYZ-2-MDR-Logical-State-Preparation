"""python mixg.py code basis d pT pG pR fgate dec budget : hw_tr mixture plus gate-layer idling at fraction fgate of pT"""
import sys, time, os, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from run_decoder_threshold_sweep import DECODERS as DEC, SCHEDULES
code, basis, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
pT, pG, pR, fg = map(float, sys.argv[4:8]); dec, budget = sys.argv[8], float(sys.argv[9])
r = 3 * d
q = fg * pT
nz = CircuitNoise(p1=pG, p2=pG, p_meas=pR, idle_mr_xyz=(pT / 84, pT / 84, 10 * pT / 21),
                  idle_xyz=((q / 42, q / 42, 20 * q / 21) if fg > 0 else None), name="mixg")
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
print(f"{code} {basis} d={d} nq={circ.num_qubits} pT={pT} pG={pG} pR={pR} fgate={fg} {dec} r={r} shots={shots} errors={errs} pL={pl:.3e} eps_round={eps:.3e} t={time.time()-t0:.0f}s rssGB={rss:.2f}", flush=True)
