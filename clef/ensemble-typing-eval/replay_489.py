"""Replay CampaignGenerator#489 (entity-level typing) on OOTA's per_chapter corpus,
restricted to the `monster` slice: every subject with at least one monster-typed fact.

Read-only. Reimplements the proposal's resolution chain beside the real bundling
code (same _norm_subject / slugify / registry aliases / known_names / monster_vocab
rules as facts_to_state.load_bundles) and scores it against the GM's type-merge
rulings. Nothing is written to CampaignGenerator or the campaign.

Arms:
  A  baseline          per-fact types, as today
  B  vote only          tier 4 (>=60%) + tier 5 queue -- no GM data; the "new chapter" case
  C  rulings + vote     tier 2 + tier 4 + tier 5 -- what is buildable today without §4 rulings
  D  C + registry       tier 2 + tier 3 (proposed mapping, deity/event/concept unmapped) + 4 + 5
                        -- WHAT-IF only: the tier-3 mapping is a pending GM decision (§4.2)
"""
import sys, json, glob, collections, yaml
sys.path.insert(0, '/home/kostadis/src/CampaignGenerator')
from pipelines.ensemble.facts_to_state import _norm_subject, slugify
from campaignlib.registry import load_registry

C = '/home/kostadis/out-of-the-abyss/out-of-the-abyss'
ENT = {'npc', 'monster', 'faction', 'location', 'object'}
REG_MAP = {'npc': 'npc', 'location': 'location', 'faction': 'faction', 'item': 'object'}
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6

reg = load_registry(f'{C}/docs/entity_registry.yaml')
aliases = {_norm_subject(k): v for k, v in reg.alias_to_canonical().items()}
party = [c['name'] for c in yaml.safe_load(open(f'{C}/config/party.yaml'))['characters']]
known_names = reg.known_names(extra=party)
reg_type = {}
for e in reg.entities:
    for s in [e.name, *e.aliases]:
        reg_type.setdefault(_norm_subject(s), e.type)

# ── corpus: per-fact (chapter, type, norm, slug) ────────────────────────────
facts = []   # (chapter_file, type, norm, slug, chapter_loc)
for p in sorted(glob.glob(f'{C}/docs/ensemble/per_chapter/*/merged.json')):
    fs = json.load(open(p))
    locs = collections.Counter(f['subject'] for f in fs if f.get('type') == 'location' and f.get('subject'))
    cl = _norm_subject(locs.most_common(1)[0][0]) if locs else 'unknown'
    for f in fs:
        raw = (f.get('subject') or '').strip()
        if not raw:
            continue
        disp = aliases.get(_norm_subject(raw), raw)
        norm = _norm_subject(disp)
        if norm:
            facts.append((p, f.get('type', ''), norm, slugify(disp), cl))
monster_vocab = {n for _, t, n, _, _ in facts if t == 'monster'}

shares = collections.defaultdict(collections.Counter)
slug_of = {}
for _, t, n, s, _ in facts:
    if t in ENT:
        shares[n][t] += 1
        slug_of.setdefault(n, s)
slice_ = {n for n in shares if shares[n]['monster']}

# ── GM rulings, keyed by slug ───────────────────────────────────────────────
G = json.load(open(f'{C}/docs/ensemble/.type_merge_decisions.json'))['groups']
primary, kept, merged_groups = {}, set(), []
for g in G:
    for r in g['resolution']:
        mem = [m[:-3].split('_', 1) for m in r['members']]
        slugs = {s for _, s in mem}
        if r['status'] == 'merged':
            pt, ps = r['primary'][:-3].split('_', 1)
            for s in slugs:
                primary[s] = pt
            merged_groups.append((g['key'], pt, slugs, {t for t, _ in mem}))
        else:
            kept |= {(t, s) for t, s in mem}   # per FILE (type, slug), not per subject

def resolve(n, tiers):
    """-> (type or None, tier) for subject n."""
    s = slug_of[n]
    if s not in primary and any(ks == s for _, ks in kept):
        return None, 'kept_separate'
    if 2 in tiers and s in primary:
        return primary[s], 2
    if 3 in tiers and reg_type.get(n) in REG_MAP:
        return REG_MAP[reg_type[n]], 3
    tot = sum(shares[n].values())
    top, c = shares[n].most_common(1)[0]
    if c / tot >= THRESH:
        return top, 4
    return None, 5

def dossiers(res):
    """Bundle keys per subject, mirroring load_bundles with known_names set."""
    out = collections.defaultdict(set)
    for _, t, n, s, cl in facts:
        if n not in slice_ or t not in ENT:
            continue
        r = res.get(n)
        t2 = t if (t, s) in kept else (r[0] if r and r[0] else t)
        if t2 == 'npc':
            known = n in known_names or n not in monster_vocab
        else:
            known = n in known_names
        out[n].add((t2,) if known else (t2, cl))
    return out

def type_split(keys):
    return len({k[0] for k in keys}) > 1

ARMS = {'A baseline': None, 'B vote only': {4}, 'C rulings+vote': {2, 4},
        'D +registry (what-if)': {2, 3, 4}}
report = {}
for name, tiers in ARMS.items():
    res = {} if tiers is None else {n: resolve(n, tiers) for n in slice_}
    ds = dossiers(res)
    knownsub = [n for n in slice_ if n in known_names]
    split = [n for n in knownsub if type_split(ds[n])]
    tiers_c = collections.Counter(r[1] for r in res.values())
    # GM-merged groups in the slice: one dossier type, and is it the GM's primary?
    hit = miss = still_split = queued = 0
    wrong = []
    for key, pt, slugs, mtypes in merged_groups:
        if 'monster' not in mtypes:
            continue
        ns = [n for n in slice_ if slug_of[n] in slugs]
        if not ns:
            continue
        ts = {k[0] for n in ns for k in ds[n]}
        if len(ts) > 1:
            still_split += 1
            if any(res.get(n, (None, None))[1] == 5 for n in ns):
                queued += 1
        elif ts == {pt}:
            hit += 1
        else:
            miss += 1
            wrong.append((key, pt, sorted(ts)))
    report[name] = dict(
        slice_subjects=len(slice_), known_in_slice=len(knownsub),
        known_type_split=len(split), dossiers=sum(len(v) for v in ds.values()),
        tiers=dict(tiers_c), gm_groups_one_type_correct=hit,
        gm_groups_one_type_wrong=miss, gm_groups_still_split=still_split,
        of_which_queued=queued, wrong=wrong,
        queue=sorted(slug_of[n] for n, r in res.items() if r[1] == 5),
        queue_known=sorted(slug_of[n] for n, r in res.items() if r[1] == 5 and n in known_names),
        split_examples=sorted(slug_of[n] for n in split)[:15])

# kept-separate check (any arm): facts of kept subjects keep original type by construction
kept_in_slice = sorted({s for t, s in kept if s in {slug_of[n] for n in slice_}})
# near 50/50 npc-vs-location (same-name-different-entity risk)
fifty = sorted(slug_of[n] for n in slice_
               if shares[n]['npc'] and shares[n]['location']
               and min(shares[n]['npc'], shares[n]['location']) / sum(shares[n].values()) >= 0.35)
report['_meta'] = dict(threshold=THRESH, groups_touching_monster=sum('monster' in m for *_, m in merged_groups),
                       kept_separate_in_slice=kept_in_slice, npc_location_near_even=fifty)
json.dump(report, open(f'replay_489_t{THRESH}.json', 'w'), indent=1)
for k, v in report.items():
    print('==', k)
    for kk, vv in v.items():
        if kk in ('queue', 'queue_known', 'wrong', 'split_examples') and isinstance(vv, list):
            print(f'  {kk}: {len(vv)}  {vv[:30]}')
        else:
            print(f'  {kk}: {vv}')
