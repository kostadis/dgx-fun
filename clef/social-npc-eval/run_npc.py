import json,sys,re,time,urllib.request
from pathlib import Path
sys.path.insert(0,'/home/kostadis/src/mytools/flexai-combat')
import flexai_combat as fc
from scenarios import PAIRS,PARTY
T=fc.load_tables(Path("/mnt/g/My Drive/DriveThru/Infinium Game Studios/FlexAI Digital Resource Companion (unisystem_5E_Pathfinder_P2E_OSR)"))
OUT={"attack_main":"Attacks its target with its standard attack (melee for most creatures, ranged for some)",
     "attack_secondary":"Attacks with its other mode: ranged if its main attack is melee, melee if its main attack is ranged",
     "maneuver":"Moves: closes on its preferred target, evades the enemies around it, or takes advantage of the terrain",
     "use_defend":"Uses an item (potion, wand, staff) or, lacking one, takes a defensive stance",
     "ability":"Uses a special ability or spell against its target",
     "flee":"Tries to flee the fight entirely"}
TGT={"frontline":"The front-most enemies","rearguard":"The rear-most enemies","closest":"The enemy physically closest to it",
     "farthest":"The enemy physically farthest from it","strongest":"The healthiest enemy, furthest from death",
     "weakest":"The enemy closest to death","ranged_enemy":"An enemy whose main attack is ranged","melee_enemy":"An enemy whose main attack is melee"}
def norm(d): s=sum(d.values()) or 1; return {k:v/s for k,v in d.items()}
def flex(role,size,stance):
    c=fc.get_cell(T,role,size,stance,'B'); o={k:0 for k in OUT}; t={k:0 for k in TGT}
    for (oc,_),rg in c['outcomes'].items():
        if rg: o[oc]+=rg[1]-rg[0]+1
    for k,rg in c['targeting'].items():
        if rg: t[k]+=rg[1]-rg[0]+1
    return norm(o),norm(t)
def post(u,b): return json.load(urllib.request.urlopen(urllib.request.Request(u,json.dumps(b).encode(),{"content-type":"application/json"}),timeout=300))
def state(s): return f"D&D 5e combat, the creature's turn.\nCreature: {s}\n{PARTY}"
QS={"outcome":{"type":"choice","instructions":"What does this creature do on its turn?","criteria":OUT},
    "target":{"type":"choice","instructions":"Which enemy does this creature target this turn?","criteria":TGT}}
def clef(s,m):
    name,_,port=m.partition('@')
    a=post(f"http://192.168.1.121:{port or '8002'}/v1/systemone",{"model":name,"state":state(s),"questions":QS})['answers']
    return a['outcome']['probabilities'],a['target']['probabilities']
def qwen(s):
    p=(state(s)+"\n\nGive a probability distribution for what this creature does this turn and whom it targets.\nOutcomes:\n"+
       '\n'.join(f'- "{k}": {v}' for k,v in OUT.items())+"\nTargets:\n"+'\n'.join(f'- "{k}": {v}' for k,v in TGT.items())+
       '\n\nReply with JSON only: {"outcome": {<every outcome key>: probability}, "target": {<every target key>: probability}}')
    t=post("http://192.168.1.147:8001/v1/chat/completions",{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":400,"chat_template_kwargs":{"enable_thinking":False}})['choices'][0]['message']['content']
    j=json.loads(re.search(r'\{.*\}',t,re.S).group(0))
    return norm({k:float(j['outcome'].get(k,0)) for k in OUT}),norm({k:float(j['target'].get(k,0)) for k in TGT})
EXTRA=sys.argv[1:]
res=[]
for P in PAIRS:
    row={'id':P['id'],'metric':P['metric'],'sign':P['sign']}
    for side in ('a','b'):
        s,(r,z,st)=P[side]
        row[side]={'flexai':flex(r,z,st),**{w.split('@')[0]:clef(s,w) for w in EXTRA}} if EXTRA else {'flexai':flex(r,z,st),'clef':clef(s,'clef'),'clef-flash':clef(s,'clef-flash'),'qwen':qwen(s)}
    res.append(row)
json.dump(res,open('npc_results_extra.json' if EXTRA else 'npc_results.json','w'),indent=1)
W=['flexai']+[w.split('@')[0] for w in EXTRA] if EXTRA else ['flexai','clef','clef-flash','qwen']; wins={w:0 for w in W}
print(f"{'pair':22s} {'metric':22s} "+' '.join(f'{w:>17s}' for w in W))
for r in res:
    kind,key=r['metric']; i=0 if kind=='outcome' else 1; cells=[]
    for w in W:
        a,b=r['a'][w][i][key],r['b'][w][i][key]; ok=(b-a)*r['sign']>0.02; wins[w]+=ok
        cells.append(f"{a:.2f}->{b:.2f} {'ok ' if ok else 'NO '}")
    print(f"{r['id']:22s} {kind+':'+key+(' up' if r['sign']>0 else ' down'):22s} "+' '.join(f'{c:>17s}' for c in cells))
print('moved the right way:',{w:f"{v}/{len(res)}" for w,v in wins.items()})
