import math
s2=math.sqrt(2)
L={'b':{'x':{3:(5.453917,0.168818,-0.0372),2:(12.263311,0.437627,-0.8979),1:(31.36137,3.598526,None)},
        'z':{3:(7.841633,0.173195,0.12934),2:(13.556778,0.562336,-0.32402),1:(25.20184,1.758445,-1.917)}},
   'p':{'x':{3:(5.673773,0.13402,0.06343),2:(12.202421,0.492157,-0.63156),1:(41.53579,4.767959,None)},
        'z':{3:(9.116592,0.290428,-0.47506),2:(14.705592,0.71796,-0.71099),1:(31.52674,2.362931,-1.7243)}}}
def slope(a,b,lr):
  s=math.log(b[0]/a[0])/lr; e=math.hypot(a[1]/a[0],b[1]/b[0])/lr
  es=math.sqrt(e**2+ (math.hypot((a[2] or 0)/a[0],(b[2] or 0)/b[0])/lr)**2)
  return s,e,es
for m in 'bp':
  sx=slope(L[m]['x'][3],L[m]['x'][2],math.log(1.5)); sz=slope(L[m]['z'][3],L[m]['z'][2],math.log(1.5))
  d=sx[0]-sz[0]; e=math.hypot(sx[1],sz[1]); es=math.hypot(sx[2],sz[2])
  print(m,'3-2',['%.3f'%v for v in sx],['%.3f'%v for v in sz],'diff %.3f +- %.3f z %.2f  syst z %.2f'%(d,e,d/e,d/es))
  mar=sx[0]-s2*sz[0]; em=math.hypot(sx[1],s2*sz[1]); print('  margin %.3f +- %.3f z %.2f'%(mar,em,mar/em))
  # drop d=5: Lambda+change
  ax=L[m]['x'][3][0]+L[m]['x'][3][2]; bx=L[m]['x'][2][0]+L[m]['x'][2][2]
  az=L[m]['z'][3][0]+L[m]['z'][3][2]; bz=L[m]['z'][2][0]+L[m]['z'][2][2]
  s1=math.log(bx/ax)/math.log(1.5); s2_=math.log(bz/az)/math.log(1.5); print('  drop d5 %.2f vs %.2f diff %.2f'%(s1,s2_,s1-s2_))
  sx2=slope(L[m]['x'][2],L[m]['x'][1],math.log(2)); sz2=slope(L[m]['z'][2],L[m]['z'][1],math.log(2))
  print('  2-1',['%.3f'%v for v in sx2],['%.3f'%v for v in sz2],'z %.2f syst %.2f'%((sx2[0]-sz2[0])/math.hypot(sx2[1],sz2[1]),(sx2[0]-sz2[0])/math.hypot(sx2[2],sz2[2])))
  # 1e-3 lambda difference
  a=L[m]['x'][1]; b=L[m]['z'][1]; print('  1e-3 diff z %.2f'%((a[0]-b[0])/math.hypot(a[1],b[1])))
  # fractions
  for p in (2,1):
    print('  frac',p, math.log(L[m]['x'][p][0])/(s2*math.log(L[m]['z'][p][0])))
  # parity extrap: ln Lx(p) = ln Lx2 + sx*ln(2e-3/p); need ln Lx = s2 ln Lz
  for lab,(sxx,szz) in (('32',(sx[0],sz[0])),('21',(sx2[0],sz2[0]))):
    lx=math.log(L[m]['x'][2][0]); lz=math.log(L[m]['z'][2][0])
    # lx + sxx*t = s2*(lz + szz*t) -> t = (s2 lz - lx)/(sxx - s2 szz)
    t=(s2*lz-lx)/(sxx-s2*szz); print('  parity',lab,'%.2e'%(2e-3*math.exp(-t)))
