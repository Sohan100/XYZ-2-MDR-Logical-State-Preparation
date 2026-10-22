exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/c10chk3.py').read().split("g=X[X.base=='phen_b10']")[0])
for noise,x in [('biased10','cfe_tn'),('biased100','tnml'),('purez','bm')]:
    a=row(noise,x); print(noise,x,round(a.pth*100,3),round(a.err*100,3))
    for _,b in bestc(noise).iterrows(): print('   ',b.decoder, round(b.pth*100,3), round(b.err*100,3), z(a,b))
