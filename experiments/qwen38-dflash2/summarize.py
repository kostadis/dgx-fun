"""Summarize recorded measurements without mixing native and wall timing."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parent
results={p.stem:json.loads(p.read_text()) for p in (root/'results').glob('*.json') if 'benchmarks' in json.loads(p.read_text())}
summary={'measurements':{},'output_comparisons':{}}
for label,data in results.items():
 rows=[]
 for row in data['benchmarks']:
  timing=row.get('timings') or {};usage=row['usage'];draft=timing.get('draft_n',0)
  rows.append({'prompt':row['name'],'wall_seconds':row['seconds'],'ttft_seconds':row['first_token_seconds'],'completion_tokens':usage['completion_tokens'],'wall_tokens_per_second':usage['completion_tokens']/row['seconds'],'native_decode_tokens_per_second':timing.get('predicted_per_second'),'native_prefill_tokens_per_second':timing.get('prompt_per_second'),'draft_tokens':draft,'accepted_draft_tokens':timing.get('draft_n_accepted'),'draft_acceptance':timing.get('draft_n_accepted',0)/draft if draft else None,'cached_tokens':timing.get('cache_n')})
 summary['measurements'][label]={'passed':data.get('passed'),'rows':rows}
 if 'plain' in results:
  comparisons=[]
  for plain,row in zip(results['plain']['benchmarks'],data['benchmarks']):
   a,b=plain['text'],row['text'];prefix=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
   comparisons.append({'prompt':row['name'],'equal_to_plain':a==b,'common_prefix_chars':prefix})
  summary['output_comparisons'][label]=comparisons
for base in ['plain-repeat','mtp-repeat']:
 if base in results and 'dflash-repeat' in results:
  ratios=[d['timings']['predicted_per_second']/b['timings']['predicted_per_second'] for d,b in zip(results['dflash-repeat']['benchmarks'],results[base]['benchmarks'])]
  summary['dflash_vs_'+base]={'ratios':ratios,'geometric_mean_speedup':math.prod(ratios)**(1/len(ratios))}
(root/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
