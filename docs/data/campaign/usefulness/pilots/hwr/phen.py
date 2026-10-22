import sys, time
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/scripts")
from mdr.ft.circuit_noise import CircuitNoise
from mdr.ft.ft_mdr_circuit import FTMDRCircuit
from mdr.ft.extraction_schedule import BOTH_BASES_SCHEDULE
from mdr.ft.two_level_decoder import TwoLevelDecoder
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
code, basis, p, tl = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
n = CircuitNoise.phenomenological(p, 10)
t0=time.time()
if code == "xyz2":
    d=5; r=3*d
    ft = FTMDRCircuit(d, r, n, detectors="combined", schedule=BOTH_BASES_SCHEDULE, logical=basis)
    dec = TwoLevelDecoder(ft, mode="bp_full", bp_method="product_sum", bp_iters=10)
else:
    d=7; r=3*d
    c = competitor_circuit(code, d, r, n, basis=basis)
    dec = competitor_decoder(c, "bm")
setup=time.time()-t0
est = dec.estimate(max_shots=10**7, max_errors=10**6, batch=500, seed=12345, time_limit=tl)
pl = est.errors/est.shots
eps = 1-(1-pl)**(1/r)
print(code, basis, "d", d, "r", r, "shots", est.shots, "errors", est.errors, "P_L %.3e eps/round %.3e setup %.0fs" % (pl, eps, setup), flush=True)
