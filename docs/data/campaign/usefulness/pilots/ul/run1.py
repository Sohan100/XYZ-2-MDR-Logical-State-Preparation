"""Quick equal-qubit probe: python run1.py code basis d p noise dec budget_s [sched] [rounds]"""
import sys, time, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import numpy as np
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from run_decoder_threshold_sweep import DECODERS as DEC, SCHEDULES, NOISE
code, basis, d, p, noise, dec, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6], float(sys.argv[7])
sched = sys.argv[8] if len(sys.argv) > 8 else 'both'
r = int(sys.argv[9]) if len(sys.argv) > 9 else 3 * d
if noise.startswith('phenb'):
    from dataclasses import replace
    rest = noise[5:]
    if 'q' in rest:
        eta_s, q_s = rest.split('q'); nz = CircuitNoise.phenomenological(p, float(eta_s)); nz = replace(nz, p_meas=p*float(q_s))
    else:
        nz = CircuitNoise.phenomenological(p, float(rest))
else:
    nz = NOISE[noise](p)
t0 = time.time()
if code == 'xyz2':
    ft = FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES[sched], logical=basis)
    D = TwoLevelDecoder(ft, **DEC[dec]); circ = D.circuit
else:
    from mdr.ft.competitor_circuits import competitor_circuit
    from mdr.ft.competitor_decoders import competitor_decoder
    circ = competitor_circuit(code, d, r, nz, basis=basis); D = competitor_decoder(circ, dec)
nq = circ.num_qubits
import os
smp = circ.compile_detector_sampler(seed=(int(time.time()*1000)+os.getpid()*7919) % 2**31)
tb0=time.time()
shots = errs = 0; size = 200
size = 100
while True:
    tb = time.time()
    dets, obs = smp.sample(size, separate_observables=True)
    errs += int(np.sum(np.any(D.decode_batch(dets) != obs, axis=1))); shots += size
    dt = time.time() - tb
    left = budget - (time.time() - t0)
    rate = size / max(dt, 1e-3)
    if left < 2: break
    size = int(max(50, min(size * 2, rate * min(left * 0.8, 15))))
    print(f"  progress shots={shots} errors={errs} t={time.time()-t0:.0f}", flush=True)
pl = errs / shots
eps = 1 - (1 - min(pl, 0.4999) * 2) ** (1 / r) if pl > 0 else 0.0
eps = eps / 2
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"{code} {basis} d={d} q={nq} p={p} {noise} {dec} {sched} r={r} shots={shots} errors={errs} pL={pl:.3e} eps_round={eps:.3e} t={time.time()-t0:.0f}s rssGB={rss:.2f}", flush=True)
