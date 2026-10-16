import sys, time, math, resource
import numpy as np
ROOT='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0, ROOT+'/src'); sys.path.insert(0, ROOT+'/scripts')
from run_decoder_threshold_sweep import NOISE, DECODERS, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
code, basis, dec, d, p, tlim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
noisename = sys.argv[7] if len(sys.argv) > 7 else 'phen_b10'
seed = int(sys.argv[8]) if len(sys.argv) > 8 else None
r = 3*d
noise = NOISE[noisename](p)
t0=time.time()
print('start',flush=True)
if code == 'xyz2':
    ft = FTMDRCircuit(d, r, noise, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical=basis)
    D = TwoLevelDecoder(ft, **DECODERS[dec]); circ = D.circuit
else:
    from mdr.ft.competitor_circuits import competitor_circuit
    from mdr.ft.competitor_decoders import competitor_decoder
    circ = competitor_circuit(code, d, r, noise, basis=basis); D = competitor_decoder(circ, dec)
sampler = circ.compile_detector_sampler(seed=seed)
nq = circ.num_qubits
print(f'built t={time.time()-t0:.0f}s',flush=True)
shots=errs=0; size=64; tb0=time.time()
while time.time()-t0 < tlim:
    tb=time.time()
    dets, obs = sampler.sample(size, separate_observables=True)
    k = int(np.sum(np.any(D.decode_batch(dets) != obs, axis=1)))
    shots += size; errs += k
    dt=max(time.time()-tb,1e-3)
    size = int(max(8, min(size*2, size*2.0/dt, 4000)))
    print(f'partial shots={shots} errors={errs} t={time.time()-t0:.0f}', flush=True)
P = errs/shots
eps = 0.5*(1-(1-2*P)**(1/r)) if P < 0.5 else float('nan')
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
print(f"{noisename} p={p} {code} {basis} {dec} d={d} r={r} qubits={nq} shots={shots} errors={errs} P={P:.4e} eps={eps:.4e} rel={1/math.sqrt(max(errs,1)):.3f} t={time.time()-t0:.0f}s rssGB={rss:.2f}", flush=True)
