"""Serve a Vela 2.0 member with its own vela2_serve.py, plus two research hooks that leave weights and readout alone.

    python3 serve_vela.py SNAPSHOT_DIR [--abstain on|off|default] [vela2_serve.py args ...]

* every Choice answer also carries `abstain_probability` (the [ABS] option's share of the softmax including it,
  which vela2_inference computes but the SystemOne view drops), so a cascade can escalate on abstention;
* --abstain on/off forces the [ABS] option into / out of every Choice question's rendered block. SystemOne
  requests cannot set it (extra fields are rejected); `default` keeps the shipped behaviour (on).
Probabilities stay those of the real options, renormalised without [ABS], exactly as shipped.
"""
import sys
from pathlib import Path

snap = Path(sys.argv.pop(1)).resolve()
mode = 'default'
if len(sys.argv) > 2 and sys.argv[1] == '--abstain':
    mode = sys.argv[2]; del sys.argv[1:3]
assert mode in ('on', 'off', 'default'), mode
sys.path.insert(0, str(snap))
import vela2_inference as V   # noqa: E402
import vela2_serve as S       # noqa: E402

if mode != 'default':
    _tg = V.Vela2._to_generic
    def _to_generic(self, req):
        parts, qs, thr = _tg(self, req)
        for g in qs:
            if g['type'] == 'choice':
                g['abstain'] = mode == 'on'
        return parts, qs, thr
    V.Vela2._to_generic = _to_generic

_LAST = {}   # vela2_serve holds one lock around every request, so a single slot is safe
_pred = V.Vela2._predict
def _predict(self, req, span_limits=None):
    out = _pred(self, req, span_limits=span_limits); _LAST['res'] = out[0]; return out
V.Vela2._predict = _predict

_resp = V.system_one_response
def system_one_response(engine, state, questions, model=None):
    r = _resp(engine, state, questions, model)
    by = {a['id']: a for a in _LAST['res']['answers']}
    for qid, a in r['answers'].items():
        if a.get('type') == 'choice':
            a['abstain_probability'] = by.get(qid, {}).get('abstain_probability')
    return r
V.system_one_response = system_one_response

print(f'serve_vela: {snap} abstain={mode}', flush=True)
sys.argv = [str(snap / 'vela2_serve.py'), '--model', str(snap)] + sys.argv[1:]
S.main()
