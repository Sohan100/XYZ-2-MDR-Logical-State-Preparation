import sys, time, numpy as np
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import DECODERS, SCHEDULES
from mdr.ft import FTMDRCircuit, CircuitNoise
from mdr.ft.two_level_decoder import TwoLevelDecoder
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
code, basis, dec, d, p, budget, eta, qfac, rr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), sys.argv[9]
pz = p*eta/(eta+1); px=py=(p-pz)/2
noise = CircuitNoise(p_meas=p*qfac, native="phen", data_xyz=(px,py,pz), name="phenx")
r = 3*d if rr=='3d' else int(rr)
t0=time.time()
if code=='xyz2':
    ft = FTMDRCircuit(d, r, noise, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical=basis)
    D = TwoLevelDecoder(ft, **DECODERS[dec]); c = D.circuit
else:
    c = competitor_circuit(code, d, r, noise, basis=basis); D = competitor_decoder(c, dec)
smp = c.compile_detector_sampler()
shots=err=0; n=16
while time.time()-t0 < budget and err < 400:
    tb=time.time(); dets, obs = smp.sample(n, separate_observables=True)
    err += int(np.sum(np.any(D.decode_batch(dets)!=obs, axis=1))); shots += n
    n = n*2 if time.time()-tb < 4 else n
pl=err/shots
eps=(1-(1-2*pl)**(1/r))/2 if pl<0.5 else float('nan')
print(f"eta={eta} q={qfac}p {code} {basis} {dec} d={d} r={r} nq={c.num_qubits} p={p} shots={shots} err={err} pL={pl:.3e} eps={eps:.3e} t={time.time()-t0:.0f}s", flush=True)
