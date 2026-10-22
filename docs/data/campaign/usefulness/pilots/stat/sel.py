import csv,sys,math,statistics
from collections import defaultdict
rows=list(csv.DictReader(open('/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv')))
V=defaultdict(dict)
for r in rows:
    if ':' in r['decoder'] or r['pth']=='' or r['err']=='': continue
    parts=r['noise'].split('@'); base=parts[0]
    sb = parts[1] if len(parts)>1 else 'depth6-X'
    s,b=sb.split('-')
    V[(base,r['rounds'],s,r['decoder'])][b]=(float(r['pth']),float(r['err']),r['method'],r['flag'])
out=defaultdict(list)
for (base,rd,s,dec),bb in V.items():
    if 'X' in bb and 'Y' in bb:
        w=min(bb.values(),key=lambda t:t[0])
        out[(base,rd)].append((w[0],w[1],s,dec,w[2],w[3]))
for key in [('sd6','d'),('si1000','d'),('si1000','1'),('phen_b10','d'),('phen','1'),('phen','d'),('biased100','d'),('purez','d'),('em3','d'),('h2_p_noxt','d')]:
    L=sorted(out[key],reverse=True)
    if not L: print(key,'none'); continue
    top=L[0]; n=len(L)
    # how many within 1 sigma (of top's err) of the top
    near=sum(1 for t in L if top[0]-t[0] < math.hypot(top[1],t[1]))
    print(key,'n=%d'%n,'top=%.3f(%.3f) %s %s %s'%(100*top[0],100*top[1],top[2],top[3],top[4]),'2nd=%.3f(%.3f) %s %s'%(100*L[1][0],100*L[1][1],L[1][2],L[1][3]),'3rd=%.3f %s %s'%(100*L[2][0],L[2][2],L[2][3]),'n within 1sig of top:',near)
