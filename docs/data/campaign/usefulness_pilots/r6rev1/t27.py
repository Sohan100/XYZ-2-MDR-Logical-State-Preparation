from lab import *
pinned=(np.isclose(th.nu,0.3,atol=1e-3)|np.isclose(th.nu,6.0,atol=1e-3))&(th.method=='fss')
th2=th[~pinned]
c2=th2[th2.decoder.str.contains(':')]; x2=th2[~th2.decoder.str.contains(':')]
T1=C.table(C.xyz2_values(xth),C.code_values(cth)); T2=C.table(C.xyz2_values(x2),C.code_values(c2))
m=T1[['noise','rounds','xyz2','ratio']].merge(T2[['noise','rounds','xyz2','ratio']],on=['noise','rounds'],suffixes=('','_2'))
m['dr']=m.ratio_2/m.ratio-1; m['dx']=m.xyz2_2/m.xyz2-1
pd.set_option('display.width',200)
print(m[(m.dr.abs()>0.02)|(m.dx.abs()>0.02)|(m.ratio.notna()!=m.ratio_2.notna())].to_string())
