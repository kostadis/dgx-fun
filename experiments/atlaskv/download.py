"""Download pinned public trial assets; never read a Hugging Face token."""
import json
from pathlib import Path
import time
from huggingface_hub import snapshot_download

ASSETS = {
    'model': ('HaoyuHuang2/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000',
              '0a4572b85ffd818f5c3b22a407e428bb2d542c27', None),
    'encoder': ('HaoyuHuang2/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000_encoder',
                '30ef42250325b656ec977792cd6bf87ad19375e7', None),
    'tokenizer': ('unsloth/Meta-Llama-3.1-8B-Instruct',
                  'a2856192dd7c25b842431f39c179a6c2c2f627d1',
                  ['tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                   'config.json','generation_config.json']),
    'sentence_encoder': ('sentence-transformers/all-MiniLM-L6-v2',
                         '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
                         ['*.json','*.txt','*.safetensors','1_Pooling/*','2_Normalize/*']),
}


def main():
    root = Path('/work')
    result = {'started_unix': time.time(), 'assets': {}}
    for name, (repo, revision, patterns) in ASSETS.items():
        print('downloading', name, repo, flush=True)
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns,
                                 token=False, max_workers=4)
        result['assets'][name] = {'repo':repo,'revision':revision,'path':path}
        (root/'downloads-progress.json').write_text(json.dumps(result,indent=2)+'\n')
    result['finished_unix'] = time.time()
    (root/'downloads.json').write_text(json.dumps(result,indent=2)+'\n')
    print('all assets ready', flush=True)


if __name__ == '__main__':
    main()
