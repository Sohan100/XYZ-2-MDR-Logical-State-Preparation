import sys
R='/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation'
sys.path.insert(0,R+'/src'); sys.path.insert(0,R+'/scripts')
from run_decoder_threshold_sweep import NOISE
from mdr.ft.competitor_circuits import competitor_circuit
for m in ('sd6','em3'):
    for d in (4,6,8,10):
        c=competitor_circuit('honeycomb', d, 2, NOISE[m](1e-3), basis='Z')
        print(m,'honeycomb',d,c.num_qubits)
