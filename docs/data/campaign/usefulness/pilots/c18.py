import sys; sys.path.insert(0,'/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad')
from pz import *
print('18/16',z([(592,8176)],9,[(579,8176)],12))
print('50/49',z([(416,11248)],15,[(468,16368)],21))
print('98/100',z([(114,6896),(129,7152)],21,[(518,49136)],30))
# trend ln(R98/R18)
import math
r18=eps(592,8176,9)/eps(579,8176,12); ex,E,_,_,_=pool([(114,6896),(129,7152)],21); r98=ex/eps(518,49136,30)
l=math.log(r98/r18); print('trend',l)
sa=math.sqrt(1/592+1/579+1/243+1/518); print('a',l/sa)
sb=math.sqrt(2.89/592+2.89/579+1/243+2.89/518); print('b',l/sb)
def zc(d):
  v=1/592+d*d+1/579+1/(1/(1/114+d*d)+1/(1/129+d*d))+1/518
  return l/math.sqrt(v)
from scipy.optimize import brentq
print('c',zc(0.065),zc(0.074),zc(0.37),'dstar',brentq(lambda d:zc(d)-3,0,2))
