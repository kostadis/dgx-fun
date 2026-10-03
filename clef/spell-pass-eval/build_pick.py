# "Which canon name is this?" test set.  Gold = the GM-approved canonical (approve verdicts),
# or LEAVE for cards the GM ruled table chatter.  Candidate pool = canonical NAMES only
# (glossary bold column, registry names+aliases, party file, module inventories) -- the glossary's
# wrong-form column is never used, because it contains these very rulings.
import json,glob,re,os,difflib,yaml,sys,functools
sys.path.insert(0,os.path.expanduser('~/.claude/skills/vtt-spell-pass'))
from dmetaphone import codes as dm
K=int(os.environ.get('K','8'))
@functools.lru_cache(None)
def dmc(w): return dm(re.sub(r'[^A-Za-z]','',w)) or set()
C='/home/kostadis/out-of-the-abyss/out-of-the-abyss'
desc={}
for e in yaml.safe_load(open(f'{C}/docs/entity_registry.yaml'))['entities']:
    d=f"{e.get('type','?')}" + (f": {e['note']}" if e.get('note') else '')
    for s in [e['name']]+(e.get('aliases') or []): desc.setdefault(s,d[:160])
for line in open(f'{C}/notes/vtt_transcription_corrections.md'):
    m=re.match(r'\|.*\|\s*\*\*(.+?)\*\*\s*\|',line)
    if m: desc.setdefault(m.group(1).strip(),'campaign glossary name')
for inv in glob.glob(f'{C}/docs/background/*inventory.md'):
    for line in open(inv):
        for nm in re.findall(r'\*\*([^*]{2,60})\*\*',line): desc.setdefault(nm.strip(),'published module: '+re.sub(r'\*','',line.strip())[:140])
pool=list(desc); low={p.lower():p for p in pool}
def cands(tok,k=None):
    k=k or K
    t=tok.lower(); sc={}
    for p in pool:
        pl=p.lower(); s=difflib.SequenceMatcher(None,t,pl).ratio()
        for w in t.split():
            if len(w)>2:
                for pw in pl.split(): s=max(s,0.95*difflib.SequenceMatcher(None,w,pw).ratio())
        # phonetic: exact Double Metaphone code overlap (as cluster_unknowns.py does), whole string or word-wise
        ph=0
        for a in [t.replace(' ','')]+t.split():
            for b in [pl.replace(' ','')]+pl.split():
                if len(a)>2 and len(b)>2 and dmc(a)&dmc(b): ph=max(ph,difflib.SequenceMatcher(None,a,b).ratio())
        s=max(s, 0.5+0.5*ph if ph else 0)
        sc[p]=s
    return sorted(sc,key=lambda p:-sc[p])[:k]
c3={(r['session'],r['id']):r for r in json.load(open('cards3.json'))}
rows=[];recall=[0,0]
for f in sorted(glob.glob(f'{C}/summaries/*/spell_review/decisions.json')):
    dec=json.load(open(f)); mp=json.load(open(f.replace('decisions','review_map'))); sess=f.split('/')[-3]
    for k,v in dec['decisions'].items():
        m=mp[k]; tok=m.get('token'); tok=(tok[0] if isinstance(tok,list) else tok) or ''
        fix=m.get('canonical'); fix=(fix[0] if isinstance(fix,list) else fix) or ''
        note=dec['notes'].get(k,'')
        if v=='approve' and isinstance(fix,str) and fix and fix!=tok: gold=fix
        elif v=='chatter' or re.search(r'chatter|table talk|real.?world',note,re.I): gold='LEAVE'
        else: continue
        if not tok.strip(): continue
        if gold!='LEAVE' and gold not in desc: continue
        s=c3[(sess,k)]['state']
        wide=re.search(r'Wider raw transcript.*?:\n(.*?)\n\nCanon and module',s,re.S); sib=re.search(r'Second, independent transcription at the same span:\n(.*?)\n\nProposed',s,re.S)
        n=re.search(r'Occurrences on tape: (\d+)',s).group(1)
        prev=sum(1 for _ in [0]) 
        state=(f"Transcript token under review: {tok}\nTimes it occurs on this session's tape: {n}\n\n"
               f"Raw transcript around its first occurrence:\n{wide.group(1) if wide else '(not found)'}\n\n"
               f"Second, independent transcription at the same span:\n{sib.group(1) if sib else '(none)'}")
        opts=cands(tok)
        if gold!='LEAVE':
            recall[1]+=1
            if gold in opts: recall[0]+=1
        rows.append(dict(session=sess,id=k,verdict=v,token=tok,gold=gold,gold_in_opts=(gold in opts or gold=='LEAVE'),
                         options={o:desc[o] for o in opts},state=state))
json.dump(rows,open('pick.json','w'),indent=1)
print(len(rows),'items;',sum(r['gold']=='LEAVE' for r in rows),'LEAVE; candidate recall (gold in top-8):',recall)
miss=[(r['token'],r['gold']) for r in rows if not r['gold_in_opts']][:15];print('misses e.g.',miss)
