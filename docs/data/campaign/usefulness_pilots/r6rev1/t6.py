from lab import *
u=C.usable(th)
g=u[(u.noise=='si1000')&(u.rounds=='1')]
print(g[['decoder','pth','err','stat','syst','nu']].sort_values('decoder').to_string())
best=[]
for cb in ['css-X','css-Z','xzzx-X','xzzx-Z','xy-X','xy-Z']:
    h=u[(u.noise=='si1000')&(u.rounds=='1')&u.decoder.str.startswith(cb+':')].sort_values('pth').iloc[-1]
    best.append(h); print(cb,h.decoder,h.pth*100,h.err*100,h.stat*100,h.syst*100)
B=pd.DataFrame(best)
w=1/B.err**2; p=(B.pth*w).sum()/w.sum(); e=w.sum()**-.5
x=0.009946901; xe=0.0001429
print('errpool',p*100,e*100,(x-p)/np.hypot(xe,e))
w=1/B.stat**2; p=(B.pth*w).sum()/w.sum(); e=w.sum()**-.5
for s in [0.0002,0.00015]:
    print('statpool',p*100,e*100,s,(x-p)/np.hypot(xe,np.hypot(e,s)))
xx=u[(u.noise=='si1000')&(u.rounds=='1')&(u.decoder=='corr_gauge')]
print(xx[['pth','err','stat','syst']])
