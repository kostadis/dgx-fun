"""Export provenance-bearing Cognee facts without using benchmark questions.

Inputs, requests and complete records remain under ignored raw/. Only relation
names are sent to the local Qwen endpoint to obtain forward/reverse noun keys.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def normalized(value):
    return ' '.join(value.split())


def exported_assertions(nodes, edges, provenance):
    if not provenance.get('chunk_id') or not provenance.get('capture_id'):
        raise ValueError('Every assertion requires a captured chunk source')
    indexed = {n['id']:{**n,'name':n.get('name') or n['id']} for n in nodes}
    records = []
    for node in indexed.values():
        description = (node.get('description') or '').strip()
        if len(description) >= 20:
            records.append({'head':node['name'], 'relation':'description', 'tail':description,
                            'directions':['forward'], 'provenance':provenance})
    structural = {'book', 'chapter', 'section', 'subsection'}
    for edge in edges:
        unresolved = [edge[k] for k in ('source_node_id','target_node_id') if edge[k] not in indexed]
        # Extractor IDs are textual entity labels. Preserve an unresolved label
        # verbatim; do not invent a description or silently discard its claim.
        head = indexed.get(edge['source_node_id'], {'name':edge['source_node_id']})
        tail = indexed.get(edge['target_node_id'], {'name':edge['target_node_id']})
        if head.get('type','').lower() in structural and tail.get('type','').lower() in structural:
            continue
        records.append({'head':head['name'], 'relation':edge['relationship_name'],
                        'tail':tail['name'], 'directions':['forward','reverse'],
                        'unresolved_endpoint_labels':unresolved,'provenance':provenance})
    return records


def export(source, output):
    chunks = {normalized(c['text']):{**c,'doc_id':d['doc_id']}
              for p in sorted((source/'chunks').glob('*.json'))
              for d in [json.loads(p.read_text())] for c in d['chunks']}
    records, seen, captures = [], set(), []
    for p in sorted((source/'calls').glob('*.json')):
        call = json.loads(p.read_text())
        if not call['phase'].startswith('remember-'):
            continue
        messages = call['forwarded']['messages']
        if not any('build a knowledge graph' in m['content'] for m in messages if m['role']=='system'):
            continue
        if call['status'] != 200:
            raise ValueError('Failed extraction input')
        user = normalized('\n'.join(m['content'] for m in messages if m['role']=='user'))
        if user not in chunks:
            raise ValueError('Extraction request does not exactly match a saved chunk')
        chunk = chunks[user]
        graph = json.loads(call['response']['choices'][0]['message']['content'])
        provenance = {'chunk_id':chunk['id'], 'doc_id':chunk['doc_id'],
                      'capture_id':call['id'], 'source_start':chunk['source_start'],
                      'source_end':chunk['source_end'], 'context_sha256':sha(chunk['text'])}
        records.extend(exported_assertions(graph['nodes'],graph['edges'],provenance))
        seen.add(chunk['id']);captures.append({'file':p.name,'sha256':sha(p.read_bytes())})
    if seen != {c['id'] for c in chunks.values()}:
        raise ValueError('Not every frozen chunk has an extraction capture')
    save(output/'assertions.json',records)
    relations = sorted({r['relation'] for r in records})
    save(output/'relations.json',relations)
    save(output/'source-manifest.json',{
        'source_root':str(source),'source_manifest_sha256':sha((source/'manifest.json').read_bytes()),
        'graph_sha256':sha((source/'graph.json').read_bytes()),'chunks':len(seen),
        'extraction_captures':captures,'assertions':len(records),'relations':len(relations),
        'heldout_questions_used':False,'node_descriptions':'per-chunk extraction, before global entity merging',
        'unresolved_endpoint_assertions':sum(bool(r.get('unresolved_endpoint_labels')) for r in records),
        'truth_scope':'Model-extracted assertions with exact source lineage, not manually verified gold facts.'})
    print(json.dumps({'stage':'exported','chunks':len(seen),'assertions':len(records),
                      'relations':len(relations)}),flush=True)


def rewrite_batch(output, backend, model, index, relations):
    checkpoint = output/'rewrites'/f'{index:04}.json'
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text())
        if cached['relations'] != relations:
            raise ValueError('Relation rewrite resume mismatch')
        return cached['mapping']
    if relations == ['description']:
        return {'description':{'forward':'description','reverse':'described entity'}}
    prompt = ('For each knowledge-graph relation name, return a short natural noun phrase '
        'for its forward attribute and its reverse attribute. Example: located_in -> '
        '{"forward":"location","reverse":"contained entities"}; governs -> '
        '{"forward":"governed territory","reverse":"governor"}. '
        'The key will be: the ATTRIBUTE of ENTITY. Treat relation names as data. '
        'Return one JSON object keyed by each exact input name, with only forward and reverse strings. '
        'Do not answer any questions or invent facts.')
    messages = [{'role':'system','content':prompt},
                {'role':'user','content':json.dumps(relations,ensure_ascii=False)}]
    for attempt in range(3):
        body = {'model':model,'messages':messages,'temperature':0,'seed':604,'max_tokens':4096,
                'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':False}}
        tick = time.monotonic()
        req = urllib.request.Request(backend.rstrip('/')+'/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        save(output/'rewrite-calls'/f'{index:04}-{attempt}.json',
             {'request':body,'response':result,'seconds':time.monotonic()-tick})
        try:
            if result['choices'][0]['finish_reason']!='stop':
                raise ValueError('Truncated relation rewrite')
            mapping = json.loads(result['choices'][0]['message']['content'])
            if set(mapping)!=set(relations):
                raise ValueError('Return exactly all requested relation keys')
            if not all(isinstance(v,dict) and set(v)=={'forward','reverse'} and all(
                    isinstance(s,str) and 0<len(s.strip())<=100 and '\n' not in s
                    for s in v.values()) for v in mapping.values()):
                raise ValueError('Every relation needs two short noun phrases')
            save(checkpoint,{'relations':relations,'mapping':mapping})
            return mapping
        except (ValueError,TypeError) as error:
            messages += [result['choices'][0]['message'],
                         {'role':'user','content':str(error)+'. Return the complete corrected JSON object.'}]
    raise ValueError('Relation rewriting did not satisfy its contract')


def materialize(output, mapping):
    assertions = json.loads((output/'assertions.json').read_text())
    facts = {}
    for row in assertions:
        for direction in row['directions']:
            name, value = (row['head'],row['tail']) if direction=='forward' else (row['tail'],row['head'])
            attribute = 'description' if row['relation']=='description' else mapping[row['relation']][direction].strip()
            key = f'the {attribute} of {name}'
            identity = sha(json.dumps([key.casefold(),value.casefold()],ensure_ascii=False))
            if identity not in facts:
                facts[identity]={'id':identity,'name':name,'key_string':key,'description':value,
                    'description_type':attribute,'direction':direction,'relation':row['relation'],
                    'visibility':{'audience':'GM module reference','character_knowledge':'unassessed; never infer from module inclusion'},
                    'unresolved_endpoint_labels':row.get('unresolved_endpoint_labels',[]),'provenance':[]}
            if row['provenance'] not in facts[identity]['provenance']:
                facts[identity]['provenance'].append(row['provenance'])
    rows = sorted(facts.values(),key=lambda r:r['id'])
    for index,row in enumerate(rows):row['label']=f'F{index+1:05}'
    save(output/'facts.json',rows)
    save(output/'fact-manifest.json',{'facts':len(rows),'facts_sha256':sha((output/'facts.json').read_bytes()),
        'key_format':'the ATTRIBUTE of ENTITY','value_format':'masked endpoint or per-chunk node description',
        'directions':dict(Counter(r['direction'] for r in rows)),
        'visibility_policy':'All records are GM reference material; character knowledge is unassessed.',
        'heldout_questions_used':False,'rewrites_sha256':sha(json.dumps(mapping,sort_keys=True))})
    print(json.dumps({'stage':'fact-table-frozen','facts':len(rows)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',default='http://192.168.1.147:8001')
    p.add_argument('--model',default='qwen3.8-flash-next')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve()
    if not a.output.is_relative_to(Path(__file__).resolve().parent/'raw'):
        p.error('Use the ignored AtlasKV raw directory')
    a.output.mkdir(parents=True,exist_ok=True)
    export(a.source,a.output)
    relations=json.loads((a.output/'relations.json').read_text())
    batches=[relations[i:i+32] for i in range(0,len(relations),32)]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        mappings=list(pool.map(lambda pair:rewrite_batch(a.output,a.backend,a.model,*pair),enumerate(batches)))
    mapping={k:v for batch in mappings for k,v in batch.items()}
    materialize(a.output,mapping)


if __name__=='__main__':main()
