import sys, time, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE, DECODERS, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
code, basis, dec, d, p, budget, model, sched, r = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8], int(sys.argv[9])
noise = NOISE[model](p)
t0=time.time()
if code=='xyz2':
    ft = FTMDRCircuit(d, r, noise, final='frame', detectors='combined', schedule=SCHEDULES[sched], logical=basis)
    D = TwoLevelDecoder(ft, **DECODERS[dec]); c = D.circuit
else:
    c = competitor_circuit(code, d, r, noise, basis=basis); D = competitor_decoder(c, dec)
smp = c.compile_detector_sampler()
shots=err=0; n=256
while time.time()-t0 < budget and err < 1000:
    tb=time.time(); dets, obs = smp.sample(n, separate_observables=True)
    err += int(np.sum(np.any(D.decode_batch(dets)!=obs, axis=1))); shots += n
    n = n*2 if time.time()-tb < 3 else n
pl=err/shots; se=(pl*(1-pl)/shots)**.5
print(f"{model} {code} {sched} {basis} {dec} d={d} r={r} nq={c.num_qubits} p={p} shots={shots} err={err} pL={pl:.3e}+-{se:.1e} t={time.time()-t0:.0f}s", flush=True)
