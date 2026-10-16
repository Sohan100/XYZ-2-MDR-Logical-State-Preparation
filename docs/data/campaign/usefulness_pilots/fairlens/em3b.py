exec(open('/tmp/claude-108819/-pscratch-sd-s-sohan100/b9bb5492-172d-4149-9bf3-b4e9300d223c/scratchpad/fairlens/ztab.py').read().split("t=pd.read_csv(R+'docs/data/campaign/competitors_table.csv')")[0])
for rk in ['d','1','2','3','5','10']:
    xv=xyzval('em3',rk)
    for c in ['css','xzzx','xy']:
        cv=compval('em3',rk,c)
        print(rk,c, round(100*xv[0],3),round(100*xv[1],3), xv[5],xv[3]+xv[4], round(100*cv[0],3), round(100*cv[1],3), round(xv[0]/cv[0],3), round((xv[0]-cv[0])/np.hypot(xv[1],cv[1]),1))
