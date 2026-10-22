import re, zlib, sys
data=open(sys.argv[1],'rb').read()
objs=[(m.start(), m.group(1)) for m in re.finditer(rb'(\d+) 0 obj', data)]
print(len(objs))
cnt=0
for m in re.finditer(rb'stream\r?\n', data):
    s=m.end(); e=data.find(b'endstream', s)
    raw=data[s:e]
    try: dec=zlib.decompress(raw)
    except Exception: continue
    cnt+=1
    # count path ops
    nl=dec.count(b' l\n')+dec.count(b' l ')
    if nl>500:
        # find obj number
        hdr=data[max(0,m.start()-300):m.start()]
        print(cnt, len(dec), nl, hdr[-200:].replace(b'\n',b' ')[:200])
