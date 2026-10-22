import math
from scipy.optimize import brentq
def z(name,e1,s1,e2,s2,bm=False):
    p1=e1/s1; p2=e2/s2; R=p1/p2; l=math.log(R)
    sa=math.sqrt(1/e1+1/e2); sb=math.sqrt(2.89/e1+2.89/e2)
    out=f"{name}: R {R:.3f}({R*sa:.3f}) za {abs(l)/sa:.1f} zb {abs(l)/sb:.1f}"
    if bm:
        ds=math.sqrt(max(0,(l/3)**2-1/e1-1/e2)); out+=f" delta* {ds:.3f}"
    print(out)
# SI1000 r=1 window (r=1 so eps=pL)
z('SI1000 35 MWPM depth6 (ovc si1000) vs CSS-Z D4 mwpm',313454,267945550,556163,323505550)
z('SI1000 35 MWPM both-X vs CSS-Z D4 mwpm',321962,268945550,556163,323505550)
z('SI1000 35 corr_gauge (bq) vs CSS-Z corr (bq)',1194,1048320,1798,1048320)
z('SI1000 35 corr_gauge (bq) vs CSS-Z corr (ovc)',1194,1048320,313329,181885550)
z('SI1000 35 BM (ovc) vs CSS-Z D4 mwpm',411,351150,556163,323505550,bm=True)
z('SI1000 35 BM (ovc) vs CSS-Z corr (ovc)',411,351150,313329,181885550,bm=True)
# C11 fixed-footprint losses
z('SD6 99/97 corr_gauge vs CSS-X d7 corr',1048,50331392,258,100663040)
z('SD6 99/97 corr_gauge vs XZZX-X d7 corr',1048,50331392,251,92274432)
z('SI1000 99/97 corr_gauge vs CSS-X d7 corr',1242,10485504,1189,25165568)
z('SI1000 99/97 corr_gauge vs CSS-Z d7 corr',1242,10485504,1107,25165568)
z('Helios corr_gauge vs CSS-X d7 mwpm',1223,14679808,1000,33554176)
z('Helios mwpm vs CSS-X d7 mwpm',1100,12582656,1000,33554176)
z('H2 corr_gauge vs CSS-X d5 mwpm',1953,2096896,1663,8388352)
z('H2 mwpm vs CSS-X d5 mwpm',1002,1048320,1663,8388352)
