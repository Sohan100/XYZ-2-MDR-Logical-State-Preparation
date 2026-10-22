import re, zlib, sys
data=open(sys.argv[1],'rb').read()
want=set(int(x) for x in sys.argv[2:])
cnt=0
for m in re.finditer(rb'stream\r?\n', data):
    s=m.end(); e=data.find(b'endstream', s)
    try: dec=zlib.decompress(data[s:e])
    except Exception: continue
    cnt+=1
    if cnt in want:
        open(f's{cnt}.txt','wb').write(dec)
