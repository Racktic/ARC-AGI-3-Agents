import json
FILES={'ar25':'ar25-bc829abb-1035-42b0-bdb9-18b24ba55a50.json',
       'bp35':'bp35-2317d132-4f08-4076-aca2-fde14c949001.json',
       'cd82':'cd82-58810dec-bb07-4e21-a6ca-4ac5d5d88f0c.json'}
CH={0:'.',1:'1',2:'o',3:'`',4:'4',5:' ',6:'6',7:'7',8:'8',9:'9',10:'#',11:'T',12:'c',13:'d',14:'=',15:'%'}
def render(g):
    out=[]
    for r in range(0,64,2):
        row=g[r]
        out.append(''.join(CH.get(row[c],'?') for c in range(0,64,2)))
    return '\n'.join(out)
for tag,f in FILES.items():
    recs=[json.loads(l)['data'] for l in open(f) if l.strip()]
    fr=recs[0]['frame']
    print('======== %s initial frame, grids/frame=%d ========'%(tag,len(fr)))
    print(render(fr[0]))
    print()
