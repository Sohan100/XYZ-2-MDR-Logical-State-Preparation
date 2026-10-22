import sys
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft import FTMDRCircuit, CircuitNoise
from mdr.ft.competitor_circuits import competitor_circuit
p=0.03; eta=10.0; h=p/2
pz = p*eta/(eta+1); px=py=(p-pz)/2
phen0 = CircuitNoise(p_meas=0.0, native="phen", data_xyz=(px,py,pz), name="phenx")
hwtr = CircuitNoise(idle_mr_xyz=(h/(2*(eta+1)), h/(2*(eta+1)), h*eta/(eta+1)), name='hw_tr_eta')
which=sys.argv[1]
nz = phen0 if which=='phen0' else hwtr
code=sys.argv[2]
if code=='xyz2':
    ft = FTMDRCircuit(3, 3, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='Y')
    c = ft.circuit if hasattr(ft,'circuit') and not callable(ft.circuit) else ft.build()
else:
    c = competitor_circuit('xzzx', 3, 3, nz, basis='X')
lines=[l for l in str(c.flattened()).splitlines() if not l.startswith('QUBIT_COORDS')]
# print names only, compressed
out=[]
for l in lines:
    nm=l.split('(')[0].split(' ')[0]
    out.append(nm)
print(' '.join(out[:60]))
print('...', ' '.join(out[-25:]))
