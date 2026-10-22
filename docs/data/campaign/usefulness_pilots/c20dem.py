import sys, resource
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
from mdr.ft import FTMDRCircuit
from run_decoder_threshold_sweep import SCHEDULES, NOISE
from mdr.ft.competitor_circuits import competitor_circuit
import stim
nz = NOISE['hw_rr'](0.04)
def obsflip(circ):
    dem = circ.detector_error_model(decompose_errors=False)
    n=0; tot=0; dets=[]
    for inst in dem.flattened():
        if inst.type != 'error': continue
        tot += 1
        tg = inst.targets_copy()
        if any(t.is_logical_observable_id() for t in tg):
            n += 1; dets.append(sorted(t.val for t in tg if t.is_relative_detector_id()))
    return n, tot, dets, circ.num_detectors
for r in (5, 15):
    ft = FTMDRCircuit(5, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='Y')
    print('xyz2 Y d5 r', r, obsflip(ft.build())[:2], obsflip(ft.build())[3], obsflip(ft.build())[2])
for r in (7, 21):
    c = competitor_circuit('xzzx', 7, r, nz, basis='X')
    n,tot,dets,nd = obsflip(c)
    print('xzzx X d7 r', r, n, tot, nd, dets)
print('rssGB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
