"""Compare text retrieval and native AtlasKV attention on one frozen fact table.

Run inside the isolated trial container. Full prompts, answers, fact IDs and
memory samples remain in private data/output directories.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import threading
import time
import types


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def memory():
    import psutil
    values={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    p=Path('/sys/fs/cgroup/memory.current')
    return {'host_available_bytes':values['MemAvailable'],
            'host_swap_used_bytes':values['SwapTotal']-values['SwapFree'],
            'process_rss_bytes':psutil.Process().memory_info().rss,
            'cgroup_current_bytes':int(p.read_text()) if p.exists() else None}


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from transformers.generation.streamers import BaseStreamer
    from transformers.models.llama.modeling_llama import LlamaAttention as StockAttention
    from transformers.models.llama.modeling_llama import LlamaForCausalLM as StockLlama
    from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
    from accelerate import init_empty_weights
    from atlaskv.kb_encoder import KBEncoder
    from atlaskv.models.kblam_config import AtlasKVConfig
    from atlaskv.models.llama3_model import AtlaskvLlamaForCausalLM,set_llama_attention_classes

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--diagnostics-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True)
    data=a.data;assets=json.loads(a.assets.read_text())['assets']
    facts=json.loads((data/'facts.json').read_text())
    cases=json.loads((data/'cases.json').read_text())
    question=json.loads((data/'question.json').read_text())['question']
    cases += [{'id':'narrative','question':question}]
    state={'stage':'loading','started_unix':time.time(),'passed':False,'completed':{}}
    save(a.output/'progress.json',state)
    torch.set_num_threads(8);torch.manual_seed(604);np.random.seed(604)
    assert torch.cuda.is_available()
    set_llama_attention_classes(True)
    model,loading=AtlaskvLlamaForCausalLM.from_pretrained(assets['model']['path'],
        torch_dtype=torch.bfloat16,device_map={'':'cuda'},low_cpu_mem_usage=True,
        attn_implementation='eager',output_loading_info=True,local_files_only=True)
    save(a.output/'loading.json',loading)
    if loading['missing_keys'] or loading['unexpected_keys'] or loading.get('mismatched_keys'):
        raise RuntimeError('Checkpoint does not exactly match the native model class')
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    attention_configuration={'loaded_outer':model.config._attn_implementation,
                             'loaded_inner':model.model.config._attn_implementation}
    if not a.diagnostics_only:
        # The native constructor performs a nested from_pretrained without
        # forwarding the outer backend. SDPA can omit a mask, but the custom
        # attention always performs manual eager matrix multiplication.
        model.config._attn_implementation='eager'
        model.model.config._attn_implementation='eager'
    attention_configuration.update(effective_outer=model.config._attn_implementation,
                                   effective_inner=model.model.config._attn_implementation)
    save(a.output/'attention-configuration.json',attention_configuration)
    tokenizer=AutoTokenizer.from_pretrained(assets['tokenizer']['path'],local_files_only=True)
    tokenizer.pad_token_id=128004
    encoder=KBEncoder(encoder_name=assets['sentence_encoder']['path'],projector_type='linear',
        out_dim=model.config.hidden_size*(model.config.num_hidden_layers//3+1),
        endpoint_url='',endpoint_api_key='',device='cuda',encoding_batch_size=128)
    def encoder_base_digest():
        digest=hashlib.sha256()
        for name,tensor in sorted(encoder.base_model.state_dict().items()):
            digest.update(name.encode());digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()
    expected_base=encoder_base_digest()
    encoder.load_state_dict(torch.load(Path(assets['encoder']['path'])/'encoder.pt',
                                      map_location='cpu',weights_only=True),strict=True)
    actual_base=encoder_base_digest()
    save(a.output/'sentence-encoder-audit.json',{'precomputed_encoder_state_sha256':expected_base,
        'trained_checkpoint_base_state_sha256':actual_base,'identical':expected_base==actual_base})
    if actual_base!=expected_base:raise RuntimeError('Precomputed embeddings and checkpoint sentence encoder differ')
    encoder.eval();encoder.base_model.to('cpu')
    for param in encoder.parameters():param.requires_grad_(False)
    keys=np.load(data/'keys.npy');values=np.load(data/'values.npy')
    assert len(facts)==len(keys)==len(values)
    normalized_keys=keys/np.maximum(np.linalg.norm(keys,axis=1,keepdims=True),1e-12)
    source_hash=hashlib.sha256((data/'facts.json').read_bytes()).hexdigest()
    protocol={'model':assets['model'],'encoder':assets['encoder'],
        'sentence_encoder':assets['sentence_encoder'],'tokenizer':assets['tokenizer'],
        'facts':len(facts),'facts_sha256':source_hash,'seed':604,'do_sample':False,
        'top_k_text_retrieval':15,'hierarchy_top_k':[128,64,16],'kb_scale_factor':100,
        'kb_layer_frequency':3,'factual_max_new_tokens':512,'narrative_max_new_tokens':1200,
        'input_context_limit':8192,'same_checkpoint_for_all_arms':True,
        'cases_sha256':hashlib.sha256((data/'cases.json').read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pins':json.loads((a.assets.parent/'pins.json').read_text()),
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda},
        'attention_configuration':attention_configuration,
        'prompt_format':'Author user/assistant header format; shared question and evidence-only instruction.',
        'timing':'Includes per-query retrieval/embedding where applicable and complete generation; excludes model/knowledge preparation.',
        'memory':'Host available/swap and process/cgroup samples include shared-device conditions; CUDA allocator is this worker only.'}
    save(a.output/'protocol.json',protocol)

    class Stream(BaseStreamer):
        def __init__(self):self.times=[];self.prompt_seen=False
        def put(self,value):
            if not self.prompt_seen:self.prompt_seen=True;return
            self.times.append(time.monotonic())
            if len(self.times)%32==0 and memory()['host_available_bytes']<8*1024**3:
                raise MemoryError('Host memory floor reached')
        def end(self):pass

    selection={}
    originals={}
    native_model=True
    def instrument(pack):
        root_map,inter_map=pack[-2:]
        for index,layer in enumerate(model.model.layers):
            attention=layer.self_attn
            if index%3:continue
            original=attention.prune_key;originals[index]=original
            def wrapped(this,weights,indices=None,topk_size=20,_original=original,_index=index):
                kept=_original(weights,indices,topk_size)
                chosen=list(kept) if isinstance(kept,range) else kept.cpu().reshape(-1).tolist()
                local=selection.setdefault(_index,{'stage':0,'calls':0,'leaf_union':set(),'prefill_leaf_union':set()})
                stage=local['stage']%3;local['stage']+=1;local['calls']+=1
                if stage==0:local['roots']=chosen
                elif stage==1:
                    local['leaf_candidates']=sorted({v for i in chosen for v in inter_map[str(i)]})
                else:
                    assert len(local['leaf_candidates'])==weights.shape[-1]
                    selected=[local['leaf_candidates'][i] for i in chosen]
                    local['leaf_union'].update(selected)
                    if weights.shape[-2]>1:local['prefill_leaf_union'].update(selected)
                return kept
            attention.prune_key=types.MethodType(wrapped,attention)

    def clear_instrumentation():
        for index,method in originals.items():model.model.layers[index].self_attn.prune_key=method
        originals.clear();selection.clear()

    def generate(prompt,kb,config,max_tokens):
        encoded=tokenizer(prompt,return_tensors='pt').to('cuda')
        if encoded['input_ids'].shape[-1]>8192:raise ValueError('Prompt exceeds frozen context budget')
        streamer=Stream()
        with torch.inference_mode():
            extra={'kb_kvs':kb,'kb_config':config} if native_model else {}
            result=model.generate(**encoded,**extra,use_cache=True,
                max_new_tokens=max_tokens,do_sample=False,temperature=1.0,top_p=1.0,top_k=0,
                pad_token_id=128004,eos_token_id=[128001,128009],streamer=streamer,
                output_attentions=False)
        torch.cuda.synchronize()
        output_ids=result[0,encoded['input_ids'].shape[-1]:].cpu().tolist()
        assert len(streamer.times)==len(output_ids)
        return {'answer':tokenizer.decode(output_ids,skip_special_tokens=True),
                'prompt_tokens':int(encoded['input_ids'].shape[-1]),'output_tokens':len(output_ids),
                'finish_reason':'stop' if output_ids and output_ids[-1] in [128001,128009] else 'length',
                'token_times':streamer.times}

    def format_prompt(user):
        return '<|start_header_id|>user<|end_header_id|> '+user+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

    if a.diagnostics_only:
        # Synthetic controls only: no book questions or answer annotations.
        config=AtlasKVConfig(sep_query_head=True,kb_layer_frequency=3,kb_scale_factor=100)
        instruction=('Use only the available evidence. Say unknown when the evidence does not establish the answer. '
                     'Be concise, at most 120 words.\nQuestion: What color permit is needed in Asterwick, '
                     'and when does Bellhaven harbor close?\nEvidence facts:\n'
                     '[F1] The permit color of Asterwick is ochre.\n'
                     '[F2] The closing time of Bellhaven harbor is sunset.\n'
                     '[F3] The founder of Cloudmere is Ilyen Moss.')
        prompts={'author':format_prompt(instruction),
                 'standard':tokenizer.apply_chat_template([{'role':'user','content':instruction}],
                     tokenize=False,add_generation_prompt=True)}
        results={'initial_attention_configuration':{
            'outer':model.config._attn_implementation,'inner':model.model.config._attn_implementation}}
        native=model
        for implementation in ('native','native-eager','stock'):
            if implementation=='native-eager':
                native.model.config._attn_implementation='eager'
            if implementation=='stock':
                # Share the same checkpoint tensors; do not allocate a second
                # model's weights or fetch another checkpoint.
                LLAMA_ATTENTION_CLASSES['eager']=StockAttention
                with init_empty_weights():stock=StockLlama(copy.deepcopy(native.config))
                unmatched=stock.load_state_dict(native.state_dict(),strict=False,assign=True)
                assert not unmatched.missing_keys
                assert all('q_proj_kg' in key or 'score_shift' in key for key in unmatched.unexpected_keys)
                stock.to('cuda');stock.eval()
                assert stock.model.embed_tokens.weight.data_ptr()==native.model.embed_tokens.weight.data_ptr()
                assert stock.lm_head.weight.data_ptr()==native.lm_head.weight.data_ptr()
                model=stock;native_model=False
                results['shared_checkpoint_control']={'missing':unmatched.missing_keys,
                    'extra_kg_parameter_names':unmatched.unexpected_keys,'same_embedding_and_lm_head_storage':True}
            for name,prompt in prompts.items():
                captured=[]
                def capture_mask(module,args,kwargs):
                    if not captured:
                        mask=kwargs.get('attention_mask')
                        captured.append({'is_none':mask is None,'shape':list(mask.shape) if mask is not None else None})
                handle=model.model.layers[0].self_attn.register_forward_pre_hook(capture_mask,with_kwargs=True)
                answer=generate(prompt,None,config,192);answer.pop('token_times')
                handle.remove();answer['first_attention_mask']=captured
                answer['prompt']=prompt
                answer['contains_expected']=all(term in answer['answer'].lower() for term in ('ochre','sunset'))
                results[implementation+'-'+name]=answer
                save(a.output/'diagnostics.json',results)
                print('diagnostic',implementation,name,answer['contains_expected'],flush=True)
        state.update(stage='diagnostics-complete',passed=True,finished_unix=time.time())
        save(a.output/'progress.json',state);return

    with torch.inference_mode():
        smoke_cfg=AtlasKVConfig(sep_query_head=True,kb_layer_frequency=3,kb_scale_factor=100)
        core=generate(format_prompt('What is 17 multiplied by 19? Reply with only the integer.'),None,smoke_cfg,64)
        core.pop('token_times');core['contains_expected']='323' in core['answer']
        save(a.output/'text-smoke.json',core)
        if not core['contains_expected'] or core['finish_reason']!='stop':
            raise RuntimeError('Text-only backbone control did not pass')
        grounding_prompt=('Use only the evidence. What color permit is needed in Asterwick, '
            'and when does Bellhaven harbor close? Evidence: Asterwick requires an ochre permit. '
            'Bellhaven harbor closes at sunset. Reply concisely.')
        grounding=generate(format_prompt(grounding_prompt),None,smoke_cfg,128)
        grounding.pop('token_times')
        grounding['contains_expected']=all(term in grounding['answer'].lower() for term in ('ochre','sunset'))
        save(a.output/'text-grounding-smoke.json',grounding)
        if not grounding['contains_expected'] or grounding['finish_reason']!='stop':
            raise RuntimeError('Text evidence control did not pass')
        smoke_keys=encoder.base_model.encode(['the winter permit rule of Asterwick','the harbor access rule of Bellhaven',
            'the founding event of Cloudmere'],convert_to_numpy=True)
        smoke_values=encoder.base_model.encode([
            'Every winter visitor to Asterwick must carry an ochre-colored permit before entering the city harbor.',
            'Bellhaven harbor closes at sunset and reopens at sunrise, and visitors must wait outside during the night.',
            'Ilyen Moss founded Cloudmere after a flood destroyed the old settlement and the survivors moved to higher ground.'
            ],convert_to_numpy=True)
        smoke_k=encoder.encode_key_cpu(base_emb=smoke_keys);smoke_v=encoder.encode_val_cpu(base_emb=smoke_values)
        smoke=generate(format_prompt('What is the winter permit rule of Asterwick?'),
                       (smoke_k.to('cuda'),smoke_v.to('cuda')),smoke_cfg,192)
        smoke.pop('token_times');smoke['expected']='ochre'
        smoke['contains_expected']='ochre' in smoke['answer'].lower()
        save(a.output/'smoke.json',smoke)
        if not smoke['answer'].strip():raise RuntimeError('Native AtlasKV probe returned no text')
        del smoke_k,smoke_v;gc.collect();torch.cuda.empty_cache()
    if a.smoke_only:
        state.update(stage='smoke-complete',passed=True,finished_unix=time.time());save(a.output/'progress.json',state);return

    for arm in ['rag','atlas','pruned']:
        state.update(stage='preparing-'+arm);save(a.output/'progress.json',state)
        setup=time.monotonic();kb=None
        config=AtlasKVConfig(sep_query_head=True,kb_layer_frequency=3,kb_scale_factor=100,
            use_hierarchial_kv=arm=='pruned',root_top_k_kb=128,inter_top_k_kb=64,leaf_top_k_kb=16)
        with torch.inference_mode():
            if arm=='atlas':kb=(encoder.encode_key(base_emb=keys),encoder.encode_val(base_emb=values))
            elif arm=='pruned':
                root=encoder.encode_key(base_emb=np.load(data/'root.npy'))
                inter=encoder.encode_key_cpu(base_emb=np.load(data/'inter.npy'))
                leaf=encoder.encode_key_cpu(base_emb=keys);v=encoder.encode_val_cpu(base_emb=values)
                kb=(root,inter,leaf,v,torch.arange(len(root)),torch.arange(len(inter)),torch.arange(len(leaf)),
                    json.loads((data/'root-children.json').read_text()),json.loads((data/'inter-children.json').read_text()))
                instrument(kb)
        save(a.output/arm/'setup.json',{'seconds':time.monotonic()-setup,'memory':memory()})
        for case in cases:
            state.update(stage=arm,question=case['id']);save(a.output/'progress.json',state)
            selection.clear();samples=[];stop=threading.Event()
            def sample():
                while not stop.wait(.25):samples.append({'unix':time.time(),**memory()})
            samples.append({'unix':time.time(),**memory()})
            thread=threading.Thread(target=sample,daemon=True);thread.start()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
            narrative=case['id']=='narrative'
            instruction=('Use only the available module evidence. Preserve negation. Say unknown when the evidence '
                'does not establish the answer. '+('Explain the narrative connection, distinguishing explicit links '
                'from interpretation and GM knowledge from character knowledge. At most 600 words.' if narrative else
                'Be concise, at most 120 words.'))
            selected=[]
            if arm=='rag':
                query=encoder.base_model.encode([case['question']],convert_to_numpy=True,normalize_embeddings=True)[0]
                scores=normalized_keys@query
                selected=np.argsort(-scores,kind='stable')[:15].tolist()
            context='\n'.join(f'[{facts[i]["label"]}] {facts[i]["key_string"]} is {facts[i]["description"]}.' for i in selected)
            user=instruction+'\nQuestion: '+case['question']+ ('\nEvidence facts:\n'+context if context else '')
            prompt=format_prompt(user)
            if any(c['question'] in prompt for c in cases if c['id']!=case['id']):
                raise ValueError('Another benchmark question leaked into the reader prompt')
            try:
                answer=generate(prompt,kb,config,1200 if narrative else 512)
                seconds=time.monotonic()-tick
            finally:
                stop.set();thread.join();samples.append({'unix':time.time(),**memory()})
            token_times=answer.pop('token_times')
            used=sorted({i for v in selection.values() for i in v['leaf_union']})
            prefill=sorted({i for v in selection.values() for i in v['prefill_leaf_union']})
            save(a.output/arm/(case['id']+'.json'),{
                'id':case['id'],'arm':arm,'question':case['question'],'prompt':prompt,
                'seconds':seconds,'ttft_first_generated_token_seconds':token_times[0]-tick if token_times else None,
                'token_offsets_seconds':[t-tick for t in token_times],**answer,
                'retrieved_fact_indices':selected,'pruned_selected_fact_indices':used,
                'pruned_prefill_selected_fact_indices':prefill,
                'selection_calls':sum(v['calls'] for v in selection.values()),
                'all_facts_available':arm=='atlas','facts_sha256':source_hash,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'memory_samples':samples})
            state['completed'][arm]=state['completed'].get(arm,0)+1
            print(arm,case['id'],round(seconds,2),answer['finish_reason'],flush=True)
        clear_instrumentation();del kb
        if arm=='pruned':del root,inter,leaf,v
        gc.collect();torch.cuda.empty_cache()
    state.update(stage='complete',passed=True,finished_unix=time.time());save(a.output/'progress.json',state)


if __name__=='__main__':main()
