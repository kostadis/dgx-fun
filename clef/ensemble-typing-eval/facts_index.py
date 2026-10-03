import sys,json,glob,collections
sys.path.insert(0,'/home/kostadis/src/CampaignGenerator')
from pipelines.ensemble.facts_to_state import _norm_subject, slugify
from campaignlib.registry import load_registry
C='/home/kostadis/out-of-the-abyss/out-of-the-abyss'
def index(corpus):
    al={_norm_subject(k):v for k,v in load_registry(f'{C}/docs/entity_registry.yaml').alias_to_canonical().items()}
    idx=collections.defaultdict(list)
    for p in sorted(glob.glob(f'{C}/docs/ensemble/{corpus}/*/merged.json')):
        for f in json.load(open(p)):
            raw=(f.get('subject') or '').strip(); disp=al.get(_norm_subject(raw),raw)
            f=dict(f,chapter=p.split('/')[-2]); idx[(f.get('type',''),slugify(disp))].append(f)
    return idx
if __name__=='__main__':
    G=json.load(open(f'{C}/docs/ensemble/.type_merge_decisions.json'))['groups']
    mem=[m for g in G for r in g['resolution'] for m in r['members']]
    for corpus in ('per_chapter','per_chapter_rerun'):
        idx=index(corpus); found=sum(tuple(m[:-3].split('_',1)) in idx for m in mem)
        print(corpus,'facts',sum(map(len,idx.values())),'members found',found,'/',len(mem))
