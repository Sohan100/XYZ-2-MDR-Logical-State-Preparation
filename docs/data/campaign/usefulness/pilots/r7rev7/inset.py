import re,sys,math
txt=open(sys.argv[1]).read()
# find inset clip blocks with polylines (2 J or dashed)
blocks=re.findall(r'Q q 69\.6 172\.3776 160\.704 119\.7504 re W n ([^Q]*?)S\n', txt)
# axis calibration: find y ticks for 10^-2 and 10^-1 (major ticks at x 230.304 -> 226.804)
yt=[float(m) for m in re.findall(r'230\.304 ([\d.]+) m\n226\.804', txt)]
xt=[float(m) for m in re.findall(r'\n([\d.]+) 172\.3776 m\n[\d.]+ 175\.8776 l', txt)]
print('yticks',yt,'xticks',xt)
y0,y1=yt[0],yt[1]  # 1e-2,1e-1 assumed
dec=y1-y0
x0,x100=xt[0],xt[1]
sc=(x100-x0)/100
lines=[]
for b in blocks:
    pts=re.findall(r'([\d.]+) ([\d.]+) [ml]\n',b)
    if len(pts)!=5: continue
    if 'J' not in b and 'd\n' not in b and '] 0 d' not in b: continue
    dashed='] 0 d' in b
    col=re.search(r'([\d. ]+(?:RG|G))',b).group(1)
    P=[((float(x)-x0)/sc, 10**(-2+(float(y)-y0)/dec)) for x,y in pts]
    lines.append((dashed,col,P))
for l in lines:
    print('XZZX' if l[0] else 'XYZ2', l[1][:20], [(round(n),'%.4g'%p) for n,p in l[2]])
xyz=[l for l in lines if not l[0]]; xz=[l for l in lines if l[0]]
def interp(P,N):
    for (n1,p1),(n2,p2) in zip(P,P[1:]):
        if n1<=N<=n2:
            f=(N-n1)/(n2-n1); return math.exp(math.log(p1)+f*(math.log(p2)-math.log(p1)))
for a,b in zip(xyz,xz):
    r=[ (round(n), p/interp(a[2],n)) for n,p in b[2] if interp(a[2],n)]
    print(a[1][:12], [(n,round(x,3)) for n,x in r])
