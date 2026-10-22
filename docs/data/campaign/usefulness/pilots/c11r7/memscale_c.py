import sys, time, numpy as np
from dataclasses import replace
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE, DECODERS, SCHEDULES
from mdr.ft import FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
scale=float(sys.argv[1]); budget=float(sys.argv[2]); seed=int(sys.argv[3])
nz=NOISE['helios_p'](1e-3); nz=replace(nz, p_mem_z=scale*nz.p_mem_z)
t0=time.time()
ft=FTMDRCircuit(5,1,nz,final='frame',detectors='combined',schedule=SCHEDULES['depth6'],logical='X')
D=TwoLevelDecoder(ft,**DECODERS['corr_gauge']); c=D.circuit
smp=c.compile_detector_sampler(seed=seed)
shots=err=0; n=4096
while time.time()-t0<budget:
    tb=time.time(); dets,obs=smp.sample(n,separate_observables=True)
    err+=int(np.sum(np.any(D.decode_batch(dets)!=obs,axis=1))); shots+=n
    n=min(n*2,1048576) if time.time()-tb<3 else n
pl=err/shots; se=(pl*(1-pl)/shots)**.5
print(f"helios_p xyz2 d=5 r=1 corr_gauge p_mem_z x{scale} nq={c.num_qubits} shots={shots} err={err} pL={pl:.3e}+-{se:.1e} t={time.time()-t0:.0f}s",flush=True)
