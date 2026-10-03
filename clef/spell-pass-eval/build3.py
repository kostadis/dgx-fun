# Arm C: arm B's card + wider tape window + near-spelling canon entries + module inventories + previous-session mentions.
# The glossary is deliberately NOT used: it contains rows created by these very rulings.
import json,glob,re,os,difflib,yaml
C='/home/kostadis/out-of-the-abyss/out-of-the-abyss'
reg=yaml.safe_load(open(f'{C}/docs/entity_registry.yaml'))['entities']
names={}  # surface -> description
for e in reg:
    d=f"registry {e.get('type','?')}: {e['name']}" + (f" (aliases: {', '.join(e['aliases'])})" if e.get('aliases') else '') + (f" — {e['note']}" if e.get('note') else '')
    for s in [e['name']]+(e.get('aliases') or []): names.setdefault(s,d)
for inv in glob.glob(f'{C}/docs/background/*inventory.md'):
    src=os.path.basename(inv).replace('-inventory.md','').replace('_',' ')
    for line in open(inv):
        for nm in re.findall(r'\*\*([^*]{2,60})\*\*',line):
            names.setdefault(nm,f"module ({src}): {re.sub(r'[*]','',line.strip())[:200]}")
party=open(f'{C}/config/party.yaml').read()

def cues(path):
    txt=open(path,errors='ignore').read();out=[]
    for blk in re.split(r'\n\s*\n',txt):
        ls=[l for l in blk.strip().split('\n') if l and '-->' not in l and not l.strip().isdigit() and l.strip()!='WEBVTT']
        if ls: out.append(' '.join(ls))
    return out
cache={}
def get(p):
    if p not in cache: cache[p]=cues(p)
    return cache[p]
def tapes(sd): return [p for p in glob.glob(f'{sd}/*.vtt') if '.cleaned' not in p and '.speakers' not in p and 'RAW' not in p]
sess=sorted(os.path.dirname(p) for p in glob.glob(f'{C}/summaries/*/'))
sess=[s.rstrip('/') for s in sorted(glob.glob(f'{C}/summaries/*/'))]
keys=list(names)
low={k.lower():k for k in keys}
B=json.load(open('cards2.json'));out=[]
for r in B:
    sd=f"{C}/summaries/{r['session']}"
    tok=re.search(r'Transcript token: (.*)',r['state']).group(1).strip()
    fix=re.search(r'Proposed replacement: (.*)',r['state']).group(1).strip()
    wide='(token not found)'
    if tok:
        for p in sorted(tapes(sd),key=lambda p:(os.path.basename(p)!='transcript.vtt',p)):
            cs=get(p);h=[i for i,c in enumerate(cs) if tok.lower() in c.lower()]
            if h: i=h[0];wide='\n'.join(cs[max(0,i-10):i+11]);break
    near=[]
    for q in {tok,fix}:
        if not q: continue
        for m in difflib.get_close_matches(q.lower(),list(low),n=4,cutoff=0.7): near.append(names[low[m]])
        for w in q.split():
            if len(w)>3:
                for m in difflib.get_close_matches(w.lower(),list(low),n=2,cutoff=0.8): near.append(names[low[m]])
    near=list(dict.fromkeys(near))[:8]
    inparty=[w for w in {tok,fix} if w and w in party]
    i=sess.index(sd) if sd in sess else -1; prev=[]
    if i>0:
        for p in tapes(sess[i-1]):
            cs=get(p)
            for q in {tok,fix}:
                if q and len(q)>2:
                    c=sum(q.lower() in x.lower() for x in cs)
                    if c: prev.append(f"'{q}' said {c} time(s) in the previous session")
            break
    extra=(f"\n\nWider raw transcript (about 10 cues either side):\n{wide}"
           f"\n\nCanon and module names with a similar spelling:\n" + ('\n'.join('- '+n for n in near) or '(none)') +
           f"\n\nIn the party file: {', '.join(inparty) or 'neither'}"
           f"\nPrevious session: {'; '.join(dict.fromkeys(prev)) or 'neither spelling was said'}")
    out.append({**r,'state':r['state']+extra})
json.dump(out,open('cards3.json','w'),indent=1)
import statistics;print(len(out),'median chars',statistics.median(len(r['state']) for r in out))
