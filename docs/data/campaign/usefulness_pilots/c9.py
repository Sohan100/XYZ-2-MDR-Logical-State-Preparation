import sys; sys.path.insert(0,'/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad')
from pz import *
S=34800
xz=[(759,262128)]
print('C9 i4 LY bm',z([(112,S),(110,S),(99,S),(95,S)],15,xz,21))
print('C9 i4 LX bm',z([(100,S),(88,S),(72,S),(92,S)],15,xz,21))
print('C9 i4 LY cg',z([(516,131056)],15,xz,21,bm=False))
print('C9 i4 LX cg',z([(438,131056)],15,xz,21))
print('C9 i4 Y bpfull',z([(73,25584)],15,xz,21))
print('C9 i4 Y bpcorr',z([(33,10224)],15,xz,21))
# hw_tr
xzX=[(11782,1850656)]
print('r3d',z([(62,22344)],15,xzX,21))
print('r6d LY',z([(52,13687),(47,13611)],30,[(14848,1132209)],42))
print('r6d LX vs X',z([(46,13542),(47,13601)],30,[(14848,1132209)],42))
print('r6d LX vs Z',z([(46,13542),(47,13601)],30,[(9381,1107657)],42))
# shared offset k=1
import math
for runs,c in (([(52,13687),(47,13611)],(14848,1132209)),([(46,13542),(47,13601)],(14848,1132209))):
  ex,E,_,_,_=pool(runs,30); ec=eps(c[0],c[1],42); l=abs(math.log(ex/ec))
  print('k1 dstar',math.sqrt((l/3)**2-1/E-1/c[0]), 'z at .37',l/math.sqrt(1/E+1/c[0]+0.37**2))
# hw_tr eta10
print('eta10 Y',z([(48,34284),(51,33493)],15,[(7370,2912826)],21))
print('x3 Y',z([(36,33978),(37,34130)],15,[(6166,2791919)],21))
print('x3 X',z([(40,33432),(22,33653)],15,[(6166,2791919)],21))
