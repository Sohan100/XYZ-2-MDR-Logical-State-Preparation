import sys
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src')
sys.path.insert(0, '/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts')
import stim
from mdr.ft import FTMDRCircuit
from run_decoder_threshold_sweep import SCHEDULES, NOISE
from mdr.ft.competitor_circuits import competitor_circuit
nz = NOISE['hw_rr'](0.04)
def chk(circ, label):
    dem = circ.detector_error_model(decompose_errors=False)
    n = 0; L = 0; dets = []
    for inst in dem.flattened():
        if inst.type != 'error': continue
        n += 1
        tg = inst.targets_copy()
        if any(t.is_logical_observable_id() for t in tg):
            L += 1; dets.append(sorted(t.val for t in tg if t.is_relative_detector_id()))
    print(label, 'mechanisms', n, 'obs-flipping', L, 'ndet', circ.num_detectors, dets)
for r in (5, 15):
    ft = FTMDRCircuit(5, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='Y')
    chk(ft.circuit if hasattr(ft,'circuit') else ft.build(), f'xyz2 Y d5 r{r}')
for r in (7, 21):
    chk(competitor_circuit('xzzx', 7, r, nz, basis='X'), f'xzzx X d7 r{r}')
