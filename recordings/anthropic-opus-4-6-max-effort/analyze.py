import json, hashlib, collections, sys

FILES = {
    'ar25': 'ar25-bc829abb-1035-42b0-bdb9-18b24ba55a50.json',
    'bp35': 'bp35-2317d132-4f08-4076-aca2-fde14c949001.json',
    'cd82': 'cd82-58810dec-bb07-4e21-a6ca-4ac5d5d88f0c.json',
}

def load(f):
    return [json.loads(l)['data'] for l in open(f) if l.strip()]

def fhash(frame):
    return hashlib.md5(json.dumps(frame).encode()).hexdigest()[:8]

def frame_dims(frame):
    g = frame[0]
    return len(g), len(g[0])

def analyze(tag, f):
    recs = load(f)
    out = []
    out.append(f"\n{'='*70}\n{tag.upper()}  ({f})\n{'='*70}")
    out.append(f"frames: {len(recs)}  final_state: {recs[-1]['state']}  levels: {recs[-1]['levels_completed']}/{recs[-1]['win_levels']}")
    h, w = frame_dims(recs[0]['frame'])
    out.append(f"grid: {h}x{w}, n_grids/frame: {len(recs[0]['frame'])}")

    prev_hash = None
    noop_count = 0
    eff_count = 0
    rows = []
    for i, r in enumerate(recs):
        a = r['action_input']
        aid = a['id']
        data = a.get('data') or {}
        xy = f"({data.get('x')},{data.get('y')})" if 'x' in data else ''
        hsh = fhash(r['frame'])
        changed = (prev_hash is not None and hsh != prev_hash)
        noop = (prev_hash is not None and hsh == prev_hash and aid != 'RESET')
        if prev_hash is not None and aid != 'RESET':
            if noop: noop_count += 1
            else: eff_count += 1
        flag = ''
        if noop: flag = 'NOOP'
        elif aid == 'RESET': flag = 'reset'
        rows.append((i, aid, xy, hsh, flag, r['state']))
        prev_hash = hsh

    # action effectiveness by type (excluding RESET)
    eff_by_act = collections.defaultdict(lambda: [0,0])  # [eff, noop]
    prev_hash = None
    for r in recs:
        aid = r['action_input']['id']
        hsh = fhash(r['frame'])
        if prev_hash is not None and aid != 'RESET':
            if hsh == prev_hash: eff_by_act[aid][1]+=1
            else: eff_by_act[aid][0]+=1
        prev_hash = hsh
    out.append(f"\nNON-RESET actions: {eff_count+noop_count}  effective(frame changed): {eff_count}  NO-OP(no change): {noop_count}  ({100*noop_count//max(1,eff_count+noop_count)}% wasted)")
    out.append("per-action effectiveness  [eff / noop]:")
    for aid in sorted(eff_by_act):
        e,n = eff_by_act[aid]
        out.append(f"   {aid}: {e} eff / {n} noop")

    # RESET positions
    resets = [i for i,r in enumerate(recs) if r['action_input']['id']=='RESET']
    out.append(f"\nRESET at frames: {resets}")

    # distinct frames visited (state-space coverage)
    hashes = [fhash(r['frame']) for r in recs]
    distinct = len(set(hashes))
    out.append(f"distinct frames visited: {distinct}/{len(recs)}  (revisits={len(recs)-distinct})")
    # most revisited frames
    hc = collections.Counter(hashes)
    out.append(f"most-revisited frame hashes: {hc.most_common(5)}")

    # detect cycles: longest run returning to same hash
    return out, recs, rows

if __name__ == '__main__':
    for tag, f in FILES.items():
        out, recs, rows = analyze(tag, f)
        print('\n'.join(out))
