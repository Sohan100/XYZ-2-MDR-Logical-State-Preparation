import sys; sys.path.insert(0,'/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad')
from pz import *
# per-run pL: use r=1 in eps -> eps=pL? eps(err,shots,1) = (1-(1-2p))/2 = p
print('mwpm Y',z([(70997,4470689)],1,[(27593,3629009)],1))
print('cg Y',z([(33044,2076184)],1,[(14883,1962124)],1))
print('bm Y',z([(887,59048)],1,[(14883,1962124)],1))
print('mwpm X',z([(48956,4718132)],1,[(27593,3629009)],1))
print('bm X',z([(574,57771)],1,[(14883,1962124)],1))
import math
p1=33044/2076184; p2=70997/4470689; print('cg vs mwpm sigma', (p1-p2)/math.hypot(p1*math.sqrt(1-p1)/math.sqrt(33044)*1, p2/math.sqrt(70997)))
print('bm/cg',(887/59048)/p1, math.sqrt(1/887+1/33044)*(887/59048)/p1)
# layout
pz5=96690/7040000
print('lay Y',z([(70997,4470689)],1,[(96690,7040000)],1))
print('lay X',(48956/4718132)/pz5)
print(math.log(16.5/14.5)/math.hypot(0.12,0.08))
