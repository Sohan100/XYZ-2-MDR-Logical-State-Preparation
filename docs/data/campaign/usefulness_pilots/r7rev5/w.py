import math
sets={'phenb10 2X':[(55,25400),(45,23371),(38,23016),(38,21488),(53,21488),(41,22512),(66,22569),(62,22000),(46,17450),(29,17450)],
'd7 3Y':[(32,7152),(26,7152),(21,7152),(33,6774),(10,6025)],
'hwtr3X':[(40,33432),(22,33653)],
'm1X':[(80,3401),(69,3187),(45,3201),(64,3269)]}
for k,runs in sets.items():
    E=sum(a for a,b in runs); S=sum(b for a,b in runs)
    mu=[E*b/S for a,b in runs]
    chi=sum((a-m)**2/m for (a,b),m in zip(runs,mu)); dof=len(runs)-1
    W=sum(m*(1-m/E) for m in mu)
    print(k,E,round(chi,2),dof,round(W,1),round(math.sqrt(max(0,chi-dof)/W),3))
