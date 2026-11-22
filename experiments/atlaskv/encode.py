"""Encode frozen facts and build the paper's three-level UMAP/GMM hierarchy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def main():
    import numpy as np
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.mixture import GaussianMixture
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();a.data=a.data.resolve()
    rows=json.loads((a.data/'facts.json').read_text())
    assets=json.loads(a.assets.read_text())['assets']
    started=time.time()
    encoder=SentenceTransformer(assets['sentence_encoder']['path'],device='cpu')
    arrays={};truncation={}
    for name,field in [('keys','key_string'),('values','description')]:
        tokenized=encoder.tokenizer([r[field] for r in rows],truncation=False,padding=False,
                                     return_attention_mask=False,return_token_type_ids=False)
        lengths=[len(ids) for ids in tokenized['input_ids']]
        truncation[name]={'encoder_max_tokens':encoder.max_seq_length,'max_input_tokens':max(lengths),
            'over_limit_fact_indices':[i for i,n in enumerate(lengths) if n>encoder.max_seq_length]}
        del tokenized
        path=a.data/(name+'.npy')
        if path.exists():
            value=np.load(path)
        else:
            value=encoder.encode([r[field] for r in rows],batch_size=128,
                show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=False).astype('float32')
            np.save(path,value)
        assert value.shape==(len(rows),384) and np.isfinite(value).all()
        arrays[name]=value
    print('base embeddings complete',len(rows),flush=True)
    reduced=umap.UMAP(n_components=64,n_neighbors=30,min_dist=.1,
        metric='cosine',random_state=42).fit_transform(arrays['keys'])
    levels=[]
    current,original=reduced,arrays['keys']
    for name,power in [('inter',2/3),('root',1/3)]:
        requested=math.ceil(len(rows)**power)
        print('clustering',name,len(current),'into',requested,flush=True)
        mixture=GaussianMixture(n_components=requested,n_init=2,max_iter=50,
            random_state=42,init_params='k-means++',verbose=1).fit(current)
        labels=mixture.predict(current)
        # An empty hard-assignment component has no valid pooled key. Remove it
        # and preserve exact membership rather than publishing a NaN centroid.
        live=sorted(set(labels.tolist()))
        children={str(i):np.flatnonzero(labels==c).tolist() for i,c in enumerate(live)}
        parents={str(child):i for i,ids in enumerate(children.values()) for child in ids}
        assert set(map(int,parents))==set(range(len(current)))
        pooled=np.stack([original[ids].mean(axis=0) for ids in children.values()]).astype('float32')
        assert np.isfinite(pooled).all()
        np.save(a.data/(name+'.npy'),pooled)
        (a.data/(name+'-children.json')).write_text(json.dumps(children)+'\n')
        (a.data/(name+'-parents.json')).write_text(json.dumps(parents)+'\n')
        levels.append({'name':name,'requested_clusters':requested,'nonempty_clusters':len(live),
                       'converged':bool(mixture.converged_),'iterations':int(mixture.n_iter_)})
        current=mixture.means_[live]
        original=pooled
    manifest={'facts':len(rows),'facts_sha256':hashlib.sha256((a.data/'facts.json').read_bytes()).hexdigest(),
        'sentence_encoder':assets['sentence_encoder'],'dimensions':384,'device':'cpu',
        'umap':{'dimensions':64,'neighbors':30,'min_dist':.1,'metric':'cosine','seed':42},
        'gmm':{'n_init':2,'max_iter':50,'init':'k-means++','seed':42},'levels':levels,
        'encoder_truncation_audit':truncation,
        'empty_component_policy':'Discard empty hard-assignment components and renumber membership maps.',
        'started_unix':started,'finished_unix':time.time()}
    (a.data/'embedding-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('embeddings and hierarchy ready',flush=True)


if __name__=='__main__':main()
