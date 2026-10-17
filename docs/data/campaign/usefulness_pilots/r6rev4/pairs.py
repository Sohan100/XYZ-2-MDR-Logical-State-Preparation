from lab import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
cth=comp_th.copy()
dec=cth.decoder.str.split(':').str[1]
adm=(cth.method=='pair')&(cth.dmin==9)&(cth.dmax==11)&(cth.flag.fillna('')!='drift')&dec.isin(['bposd','tesseract'])
c2=cth.copy(); c2.loc[adm,'method']='fss'
comp2=C.code_values(c2)
base=C.table(C.xyz2_values(xyz_th),comp)
new=C.table(C.xyz2_values(xyz_th),comp2)
m=base[['noise','rounds','best_other','ratio']].merge(new[['noise','rounds','best_other','ratio','xyz2']],on=['noise','rounds'],suffixes=('','_p'))
m=m[(m.ratio.notna())&((m.ratio-m.ratio_p).abs()>1e-9)]
print(m.to_string())
f2=rows(C.xyz2_values(xyz_th),comp2)
f1=rows(C.xyz2_values(xyz_th),comp)
mm=f1.merge(f2,on=['noise','r'],suffixes=('','_p'))
print(mm[mm.label!=mm.label_p][['noise','r','label','zmin','label_p','zmin_p','zmincode_p']])
