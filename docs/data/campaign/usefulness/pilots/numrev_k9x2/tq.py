import csv,sys,re
f=sys.argv[1]; noise=sys.argv[2]; rounds=sys.argv[3]; pat=sys.argv[4] if len(sys.argv)>4 else '.'
for x in csv.DictReader(open(f)):
    if re.fullmatch(noise,x['noise']) and (rounds=='*' or x['rounds']==rounds) and re.search(pat,x['decoder']):
        def p(k):
            try: return '%.4f'%(float(x[k])*100)
            except: return '-'
        print(x['noise'],x['rounds'],x['decoder'],x['method'],'pth',p('pth'),'err',p('err'),'nu',x['nu'][:5],'chi2',x['chi2'][:5],'n',x['npts'],'d',x['dmin'],x['dmax'],'cf',p('cross_first'),'cl',p('cross_last'),'drift',x['drift'][:5],'flag',x['flag'])
