"""Check causal masking in the actual upstream method without loading weights."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def exercise(source):
    import torch
    from transformers.cache_utils import StaticCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_update_causal_mask')
    scope = {'torch': torch, 'StaticCache': StaticCache, 'Cache': object,
             'AttentionMaskConverter': AttentionMaskConverter}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    result = {}
    for policy in ('sdpa', 'eager'):
        owner = SimpleNamespace(config=SimpleNamespace(_attn_implementation=policy), training=False)
        mask = scope['_update_causal_mask'](owner, torch.ones(1, 5), torch.zeros(1, 5, 8),
                                           torch.arange(5), None, False)
        if policy == 'sdpa':
            assert mask is None
        else:
            assert mask.shape == (1, 1, 5, 5)
            for query in range(5):
                for key in range(5):
                    assert (mask[0, 0, query, key] < -1000) == (key > query)
            assert torch.all(mask[0, 0].tril() == 0)
        result[policy] = {'is_none': mask is None, 'shape': list(mask.shape) if mask is not None else None}
    return {'passed': True, 'upstream_mask_policy': result,
            'interpretation': 'Manual eager attention needs the explicit eager mask; SDPA assumes its kernel supplies causality.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(exercise(args.source), indent=2))
