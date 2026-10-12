"""Check a served tuned head against the replay: send K rows of the half it serves, compare probabilities.

    python3 ft_probe.py PART VARIANT HALF [K]      e.g. e3 triage-lenient 0 20
"""
import json, random, sys
import tasks as T, cascade as C
part, var, h = sys.argv[1], sys.argv[2], sys.argv[3]; k = int(sys.argv[4]) if len(sys.argv) > 4 else 20
E8 = C.DATA / 'results' / 'e8'
F = json.load(open(E8 / f'{var}.json'))['halves'][h]['test']
src, q = {'e3': ('e3_triage', T.Q_TRIAGE), 'e4_facts': ('e4_facts', T.Q_FACT)}[part]
rows = json.load(open(C.DATA / f'{src}.json'))
idx = random.Random(0).sample(sorted(F, key=int), k); worst = 0
for i in idx:
    ans, _, _ = T.systemone('nox', rows[int(i)]['state'], q)
    p = ans['kind']['probabilities']; worst = max(worst, max(abs(p[x] - F[i][x]) for x in p))
print(f'{part} {var} h{h}: {k} rows, max |served - replay| = {worst:.5f}')
sys.exit(0 if worst < 0.02 else 1)
