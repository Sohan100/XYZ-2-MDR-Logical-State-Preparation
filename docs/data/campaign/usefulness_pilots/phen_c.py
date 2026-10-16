import sys, math
sys.path.insert(0, "/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/src")
from mdr.ft import CircuitNoise
from mdr.ft.competitor_circuits import CompetitorCircuit
from mdr.ft.competitor_decoders import competitor_decoder
p=float(sys.argv[1]); tl=float(sys.argv[2]); code=sys.argv[3]; b=sys.argv[4]; name=sys.argv[5]
noise = CircuitNoise.phenomenological(p, 10)
D=7; R=21
c = CompetitorCircuit(code, D, R, noise, basis=b).build()
dec = competitor_decoder(c, name)
est = dec.estimate(max_shots=10**7, max_errors=10**6, batch=1000, time_limit=tl, seed=777)
pr = 0.5*(1-(1-2*est.rate)**(1/R))
print(code, b, name, est.errors, est.shots, "%.3e +- %.1e"%(pr, pr/math.sqrt(max(est.errors,1))))
