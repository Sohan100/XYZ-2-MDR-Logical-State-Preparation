import csv,math,sys
P=list(csv.DictReader(open('points_eps.csv')))
pairs=[r for r in csv.DictReader(open('pairs.csv')) if r['mode']=='chosen' and r['model'] in ('purez','biased100') and r['other']=='xzzx']
for pr in pairs:
    m=pr['model'];p=pr['p']
    dx=pr['d_xyz2']; do=pr['d_other']
    xs=[r for r in P if r['model']==m and float(r['p'])==float(p) and r['code']=='xyz2' and r['dec']=='bm' and r['rkey']=='3d' and r['d']==dx]
    zs=[r for r in P if r['model']==m and float(r['p'])==float(p) and r['code']=='xzzx' and r['dec']=='corr' and r['basis']=='X' and r['rkey']=='3d' and r['d']==do]
    if not xs or not zs: print(m,p,pr['n_xyz2'],'missing',len(xs),len(zs)); continue
    z=zs[0]; 
    w=max(xs,key=lambda r: float(r['eps'] or 0))
    out=[]
    for x in xs:
        ex,nx=float(x['eps'] or 0),int(x['errors']); ez,nz=float(z['eps'] or 0),int(z['errors'])
        R=ex/ez if ez else float('nan')
        s='%s %s:%.3e(%d) R=%.2f [%.2f,%.2f]'%(x['basis'],x['decoder'],ex,nx,R,float(x['eps_lo'] or 0)/float(z['eps_hi'] or 1) if nx else 0,float(x['eps_hi'] or 0)/(float(z['eps_lo'] or 0) or 1))
        if nx>0 and nz>0:
            L=abs(math.log(R)); a=L/math.sqrt(1/nx+1/nz); b=L/math.sqrt(2.89/nx+1/nz); c=[L/math.sqrt(1/nx+1/nz+d*d) for d in (.15,.25)]
            s+=' z %.1f/%.1f/%.1f/%.1f'%(a,b,c[0],c[1])
        out.append(s)
    print(m,p,pr['n_xyz2'],pr['n_other'],'XZZX X %s(%s)'%(z['eps'],z['errors']),'weaker',w['basis'],'|',' ; '.join(out))
