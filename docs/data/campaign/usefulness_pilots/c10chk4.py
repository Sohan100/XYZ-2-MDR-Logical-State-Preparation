exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/c10chk3.py').read().split("g=X[X.base=='phen_b10']")[0])
for sl in ['depth6-Y','both-Y','both-X']:
    g=xs('phen',sl).sort_values('pth'); a=g.iloc[-1]; print(sl,a.decoder,a.pth,a.err,a.nu,a.stat)
    for _,b in bestc('phen').iterrows():
        if 'honeycomb' not in b.decoder: print('   vs',b.decoder, z(a,b))
