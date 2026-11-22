"""Exercise upstream pruning expressions on non-contiguous candidate indices."""
import argparse
import ast
import json
from pathlib import Path


def exercise(path):
    import numpy as np
    import torch
    import torch.nn as nn
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'prune_key')
    scope = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    weights = torch.tensor([[[[0.0, 1.0, 8.0]]]])
    index = scope['prune_key'](None, weights, topk_size=1)
    leaf_assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'select_leaf_kb_values_cpu' for t in n.targets))
    values = eval(compile(ast.Expression(leaf_assignment.value), str(path), 'eval'), {
        'kb_values': torch.arange(20).reshape(1, 1, 20, 1), 'top_idx': index,
        'np': np, 'select_leaf_local_key_idx': np.array([3, 7, 11])})
    inter_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'prune_key'
        and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'attn_weights_2_inter')
    candidates = eval(compile(ast.Expression(inter_call.args[1]), str(path), 'eval'), {
        'torch': torch, 'inter_kb_indices': torch.arange(25),
        'select_inter_kb_indices': np.array([4, 10, 18])})
    selected = scope['prune_key'](None, weights, candidates, topk_size=1)
    return {'selected_leaf_value': int(values.flatten()[0]),
            'selected_intermediate_id': int(selected.flatten()[0]),
            'singleton_indices_remain_vector': isinstance(index, torch.Tensor) and index.ndim == 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--patched', type=Path, required=True)
    a = p.parse_args()
    before, after = exercise(a.original), exercise(a.patched)
    assert before['selected_leaf_value'] == 2 and before['selected_intermediate_id'] == 2
    assert after == {'selected_leaf_value': 11, 'selected_intermediate_id': 18,
                     'singleton_indices_remain_vector': True}
    print(json.dumps({'passed': True, 'before': before, 'after': after}, indent=2))


if __name__ == '__main__':
    main()
