from lab import *
pd.set_option('display.width',250)
# admit competitor non-drift d=9/11 bposd/tesseract pair crossings
pc=cth[(cth.method=='pair')&(cth.dmin==9)&(cth.dmax==11)&(cth.flag.fillna('')!='drift')&cth.decoder.str.contains(':(?:bposd|tesseract)$')&cth.pth.notna()].copy()
pc['method']='fss'
cth2=pd.concat([cth,pc])
comp2=C.code_values(cth2)
xf=C.xyz2_values(xth)
L1=labels(xf,comp); L2=labels(xf,comp2)
m=L1.merge(L2,on=['noise','rounds'],suffixes=('','_p'))
ch=m[(m.topv_p>m.topv+1e-9)]
print(ch[['noise','rounds','top','topv','ratio','top_p','topv_p','ratio_p','zmin','zmin_p','argmin_p','label','label_p']].to_string())
print(m[m.label!=m.label_p][['noise','rounds','label','label_p','zmin','zmin_p']])
