import sys, collections
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import SCHEDULES
from mdr.ft import FTMDRCircuit, CircuitNoise
from mdr.ft.competitor_circuits import competitor_circuit
p=0.03; eta=10.0
pz = p*eta/(eta+1); px=py=(p-pz)/2
phen0 = CircuitNoise(p_meas=0.0, native="phen", data_xyz=(px,py,pz), name="phenx")
h=p/2; hwtr = CircuitNoise(idle_mr_xyz=(h/(2*(eta+1)), h/(2*(eta+1)), h*eta/(eta+1)), name='hw_tr_eta')
def census(c, r):
    tot=collections.Counter(); cnt=collections.Counter()
    for inst in c.flattened():
        n=inst.name; a=inst.gate_args_copy()
        if a and sum(a)>0:
            k=len(inst.targets_copy())
            if n in ('MPP',): k=1
            cnt[n]+=k; tot[n]+=sum(a)*len(inst.targets_copy())
    dem=c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    s=sum(e.args_copy()[0] for e in dem.flattened() if e.type=='error')
    return dict(cnt), {k:round(v/r,4) for k,v in tot.items()}, round(s/r,4), c.num_detectors
for d in (3,5):
    r=3*d
    for name,nz in (('phen0',phen0),('hwtr',hwtr)):
        ft = FTMDRCircuit(d, r, nz, final='frame', detectors='combined', schedule=SCHEDULES['both'], logical='Y')
        c = ft.circuit if hasattr(ft,'circuit') and not callable(ft.circuit) else ft.build()
        print('xyz2',d,name,c.num_qubits,census(c,r))
    for name,nz in (('phen0',phen0),('hwtr',hwtr)):
        c = competitor_circuit('xzzx', d+2 if d==5 else d, r, nz, basis='X')
        print('xzzx',d,name,c.num_qubits,census(c,r))
