import sys; sys.path.insert(0,'/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/r7rev2')
from z import zc
# hw_rr per run pL: use r=1 so eps=pL... eps with r=1 = pL
zc([(887,59048)],1,[(14883,1962124)],1,label='hw_rr BM Y')
zc([(33044,2076184)],1,[(14883,1962124)],1,cbm=False,label='hw_rr cg Y')
zc([(70997,4470689)],1,[(27593,3629009)],1,cbm=False,label='hw_rr mwpm Y')
zc([(48956,4718132)],1,[(27593,3629009)],1,cbm=False,label='hw_rr mwpm X')
zc([(574,57771)],1,[(14883,1962124)],1,label='hw_rr BM X')
# per round eps for BM
zc([(887,59048)],15,[(14883,1962124)],21,label='hw_rr BM Y per round')
