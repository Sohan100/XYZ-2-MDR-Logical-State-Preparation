import sys, time, os, json, resource
import numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0, R+'/src'); sys.path.insert(0, R+'/scripts')
import campaign as C
from run_decoder_threshold_sweep import DECODERS as DEC, NOISE, SCHEDULES
label, decname, d, p, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
rounds = int(os.environ["ROUNDS"]) if os.environ.get("ROUNDS") else 3*d
t0=time.time()
base, sched, logical = C.split_noise(label)
noise = NOISE[base](p)
comp = C.competitor(decname)
if comp is not None:
    from mdr.ft.competitor_circuits import competitor_circuit
    from mdr.ft.competitor_decoders import competitor_decoder
    code, basis, name = comp
    circuit = competitor_circuit(code, d, rounds, noise, basis=basis)
    dec = competitor_decoder(circuit, name)
else:
    from mdr.ft import FTMDRCircuit
    from mdr.ft.two_level_decoder import TwoLevelDecoder
    ft = FTMDRCircuit(d, rounds, noise, final="frame", detectors="combined", schedule=SCHEDULES[sched], logical=logical)
    dec = TwoLevelDecoder(ft, **DEC[decname]); circuit = dec.circuit
nq = circuit.num_qubits
sampler = circuit.compile_detector_sampler(seed=int(sys.argv[6]) if len(sys.argv)>6 else None)
tb=time.time(); shots=errs=0; size=50
while time.time()-tb < budget:
    dets, obs = sampler.sample(size, separate_observables=True)
    k = int(np.sum(np.any(dec.decode_batch(dets) != obs, axis=1)))
    shots += size; errs += k
    size = min(size*2, int(os.environ.get("MAXB","2000")))
pl = errs/shots
eps = 1-(1-2*pl)**(1/rounds) if pl<0.5 else float('nan')
eps = eps/2
print(json.dumps(dict(label=label, dec=decname, d=d, p=p, r=rounds, nq=nq, shots=shots, errors=errs, pL=pl, eps=eps,
      build=round(tb-t0,1), sec=round(time.time()-tb,1), maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)), flush=True)
