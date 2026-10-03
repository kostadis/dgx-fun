import json,sys,re,math,time,urllib.request,collections
sys.path.insert(0,'/home/kostadis/src/mytools/flexai-social')
import flexai_social as fs
LAB={"helps":"Agrees to do what was asked: helps, joins or assists",
     "answers":"Gives the information asked for, willingly or grudgingly, without bargaining",
     "bargains":"Agrees only for a price, makes a counter-offer, or sets conditions",
     "questions_motives":"Challenges the party, demands justification, is suspicious, or tests them",
     "deceives":"Lies, misleads, or gives false or deliberately unrelated information",
     "ignores":"Ignores the party or brushes them off",
     "leaves":"Ends the conversation and walks away",
     "hostile":"Turns openly hostile: threatens violence or attacks"}
MAP={"turns_hostile":"hostile","leaves":"leaves","ignores_you":"ignores","helps":"helps","answers_grudgingly":"answers","answers":"answers",
     "answers_willingly":"answers","volunteers_info":"answers","grant_plot_clue":"answers","reveals_plot_clue":"answers",
     "challenges_you":"questions_motives","questions_motives":"questions_motives","red_herring":"deceives","lies":"deceives"}
T=fs.load_tables(fs.default_data_dir())
def norm(d): s=sum(d.values()) or 1; return {k:v/s for k,v in d.items()}
def flex(role,ctx):
    c=T[role]['normal'][ctx]['B']; d={k:0 for k in LAB}
    for tab in ('success_results','failure_results'):
        for k,rg in c[tab].items():
            if rg: d[MAP[k]]+=rg[1]-rg[0]+1
    return norm(d)
def post(u,b): return json.load(urllib.request.urlopen(urllib.request.Request(u,json.dumps(b).encode(),{"content-type":"application/json"}),timeout=300))
def state(r): return f"Out of the Abyss, a D&D 5e campaign. The party is talking to {r['npc']}.\n\nWhat has happened so far:\n{r['pre_context']}"
def ins(r): return f"How does {r['npc']} respond to the party at this moment?"
def clef(r,m): return post("http://192.168.1.121:8002/v1/systemone",{"model":m,"state":state(r),"questions":{"resp":{"type":"choice","instructions":ins(r),"criteria":LAB}}})['answers']['resp']['probabilities']
def qwen(r,think=False):
    p=state(r)+"\n\n"+ins(r)+" Give a probability for each option:\n"+'\n'.join(f'- "{k}": {v}' for k,v in LAB.items())+'\n\nReply with JSON only: {<every option key>: probability}'
    t=post("http://192.168.1.147:8001/v1/chat/completions",{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":6000 if think else 300,"chat_template_kwargs":{"enable_thinking":think}})['choices'][0]['message']['content'] or ''
    try: j=json.loads(re.search(r'\{[^{}]*\}',t,re.S).group(0)); return norm({k:max(0.0,float(j.get(k,0))) for k in LAB})
    except Exception: return None
if __name__=='__main__':
    rows=json.load(open(sys.argv[1])); out=[]
    for r in rows:
        x={**{k:r[k] for k in('id','npc','label')},
           'flexai':flex(r['npc_role'],r['context_setting']),'clef':clef(r,'clef'),'clef-flash':clef(r,'clef-flash'),'qwen':qwen(r)}
        out.append(x)
    json.dump(out,open('social_results.json','w'),indent=1)
    prior=collections.Counter(r['label'] for r in rows); n=len(rows)
    print(f"n={n} labels={dict(prior)}  majority-class top-1 = {max(prior.values())/n:.0%}")
    for w in ('flexai','clef','clef-flash','qwen'):
        ok=[x for x in out if x[w]]; top=sum(max(x[w],key=x[w].get)==x['label'] for x in ok)
        pt=[x[w][x['label']] for x in ok]; ll=-sum(math.log(max(p,1e-3)) for p in pt)/len(pt)
        top2=sum(x['label'] in sorted(x[w],key=x[w].get)[-2:] for x in ok)
        print(f"{w:10s} top-1 {top}/{len(ok)} ({top/len(ok):.0%})  top-2 {top2/len(ok):.0%}  mean P(truth) {sum(pt)/len(pt):.2f}  log-loss {ll:.2f}")
