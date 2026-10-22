import sys, time
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
from mdr.ft import BOTH_BASES_SCHEDULE, CircuitNoise, FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder
from mdr.ft.competitor_circuits import CompetitorCircuit
from mdr.ft.competitor_decoders import competitor_decoder
p = float(sys.argv[1]); tl = float(sys.argv[2])
noise = CircuitNoise.phenomenological(p, 10)
def per_round(pl, r): return 0.5*(1-(1-2*pl)**(1.0/r))
d=5; r=3*d
for L in ("X","Y"):
    ft = FTMDRCircuit(d, r, noise, final="frame", detectors="combined", schedule=BOTH_BASES_SCHEDULE, logical=L)
    for name, kw in (("bm", dict(mode="bp_full", bp_method="product_sum", bp_iters=10)),):
        t=time.time(); dec = TwoLevelDecoder(ft, **kw)
        est = dec.estimate(max_shots=10**6, max_errors=400, batch=500, time_limit=tl)
        print("xyz2", L, name, est.errors, est.shots, "%.3e"%per_round(est.rate, r), "%.0fs"%(time.time()-t), flush=True)
D=7; R=3*D
for code in ("xzzx","xy"):
    for b in ("X","Z"):
        c = CompetitorCircuit(code, D, R, noise, basis=b).build()
        for name in (("corr","bm") if code=="xzzx" else ("bm",)):
            t=time.time(); dec = competitor_decoder(c, name)
            est = dec.estimate(max_shots=10**6, max_errors=400, batch=500, time_limit=tl)
            print(code, b, name, est.errors, est.shots, "%.3e"%per_round(est.rate, R), "%.0fs"%(time.time()-t), flush=True)
