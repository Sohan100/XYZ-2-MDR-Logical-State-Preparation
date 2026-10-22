import sys, time
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
from mdr.ft import BOTH_BASES_SCHEDULE, CircuitNoise, FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
p=float(sys.argv[1]); tl=float(sys.argv[2]); L=sys.argv[3]; mode=sys.argv[4]
noise = CircuitNoise.phenomenological(p, 10)
d=5; r=15
ft = FTMDRCircuit(d, r, noise, final="frame", detectors="combined", schedule=BOTH_BASES_SCHEDULE, logical=L)
kw = dict(mode="bp_full", bp_method="product_sum", bp_iters=10) if mode=="bm" else dict(mode="bp_full")
dec = TwoLevelDecoder(ft, **kw)
est = dec.estimate(max_shots=10**7, max_errors=10**6, batch=1000, time_limit=tl, seed=12345)
pr = 0.5*(1-(1-2*est.rate)**(1/r))
import math
print("xyz2", L, mode, est.errors, est.shots, "%.3e +- %.1e"%(pr, pr/math.sqrt(max(est.errors,1))))
