"""Freeze every experiment input of the decision-models paper into one dataset directory.

The per-experiment builders (spell-pass-eval/build*.py, ensemble-typing-eval/build_*.py,
replay_489.py) read the live campaign tree, which keeps changing. This script snapshots what
they produced -- with every model-facing `state` string already rendered -- so that every model
is run against byte-identical inputs and the run can be repeated later.

    python3 freeze.py [OUT_DIR]      # default ~/data/decision-eval/v1

Run the builders first (see README.md). The output directory holds campaign text and real
names: it lives OUTSIDE the repo and is never committed. Only MANIFEST.json (hashes + counts,
no content) is copied into the repo.
"""
import collections, datetime, glob, hashlib, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEF = HERE.parent
SP, ET, NPC = CLEF / 'spell-pass-eval', CLEF / 'ensemble-typing-eval', CLEF / 'social-npc-eval'
CAMPAIGN = '/home/kostadis/out-of-the-abyss/out-of-the-abyss'
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/data/decision-eval/v1'))
if (OUT / 'MANIFEST.json').exists():
    sys.exit(f'{OUT} is already frozen; pick a new version directory')
OUT.mkdir(parents=True, exist_ok=True)
load = lambda p: json.load(open(p))
files = {}


def save(name, rows, note):
    p = OUT / f'{name}.json'
    json.dump(rows, open(p, 'w'), indent=1, ensure_ascii=False)
    files[name] = dict(n=len(rows), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), note=note)
    print(f'{name:18s} {len(rows):5d}  {note}')


# ---- E1: GM verdict on review cards -------------------------------------------------------
prose = [dict(session=r['session'], id=r['id'], verdict=r['verdict'], note=r['note'],
              state=f"Card: {r['title']}\n\nProposal: {r['proposal']}\n\nEvidence: {r['evidence']}")
         for r in load(SP / 'cards.json')]
save('e1_cards_prose', prose, 'review cards as the upstream model wrote them')
for arm, src, note in (('B', 'cards2.json', 'cards rebuilt from raw sources only (paper: "raw sources")'),
                       ('C', 'cards3.json', 'arm B + wider tape window, near-spelling canon, previous session')):
    save(f'e1_cards_raw{arm}', [{k: r[k] for k in ('session', 'id', 'verdict', 'note', 'state')}
                                for r in load(SP / src)], note)

# ---- E2: canonical-spelling choice --------------------------------------------------------
save('e2_pick', load(SP / 'pick.json'), 'token + tape + candidate names; gold = GM-approved canonical or LEAVE')


# ---- E3: triage of the unknown-token pile (state rendering copied from triage.py) ---------
def cues(path):
    out = []
    for blk in re.split(r'\n\s*\n', open(path, errors='ignore').read()):
        ls = [l for l in blk.strip().split('\n')
              if l and '-->' not in l and not l.strip().isdigit() and l.strip() != 'WEBVTT']
        if ls: out.append(' '.join(ls))
    return out


tapes = {}
def tape(sess):
    if sess not in tapes:
        ps = [p for p in glob.glob(f'{CAMPAIGN}/summaries/{sess}/*.vtt')
              if '.cleaned' not in p and '.speakers' not in p and 'RAW' not in p]
        ps.sort(key=lambda p: (os.path.basename(p) != 'transcript.vtt', p))
        tapes[sess] = cues(ps[0]) if ps else []
    return tapes[sess]


tri = []
for r in load(SP / 'triage_labels.json'):
    cs = tape(r['session']); t = r['token'].lower()
    hits = ([i for i, c in enumerate(cs) if re.search(r'\b' + re.escape(t) + r'\b', c.lower())]
            or [i for i, c in enumerate(cs) if t in c.lower()])
    lines = [' / '.join(cs[max(0, i - 1):i + 2]) for i in hits[:3]]
    tri.append(dict(session=r['session'], token=r['token'], label=r['label'], n=len(hits),
                    state=f"Capitalised token found in a D&D session transcript: {r['token']}\nOccurrences: {len(hits)}\n\n"
                          "Where it appears (up to 3 places, with the neighbouring lines):\n"
                          + ('\n'.join('- ' + l for l in lines) or '(not found)')))
save('e3_triage', tri, 'label: name (84) / chatter / leave')

# ---- E4: entity typing --------------------------------------------------------------------
save('e4_ent', load(ET / 'ent.json'), 'per-entity typing; merged groups + kept-separate members')
save('e4_facts', load(ET / 'facts.json'), 'per-fact typing, repair set (facts of GM-merged entities)')
save('e4_ctrl', load(ET / 'ctrl.json'), 'per-fact typing, control set (consistently typed entities)')
rep = {}
for t in ('0.5', '0.55', '0.6', '0.67', '0.75'):
    rep[t] = load(ET / f'replay_489_t{t}.json')
save('e4_replay', [dict(threshold=float(t), report=v) for t, v in rep.items()],
     'model-free entity-level typing replay (CampaignGenerator#489), monster slice')

# ---- E5: NPC behaviour (FlexAI baseline distributions frozen too: tables live on a Drive mount)
sys.path.insert(0, str(NPC))
from scenarios import PAIRS, PARTY  # noqa: E402
sys.path.insert(0, '/home/kostadis/src/mytools/flexai-combat')
import flexai_combat as fc  # noqa: E402
T = fc.load_tables(Path('/mnt/g/My Drive/DriveThru/Infinium Game Studios/'
                        'FlexAI Digital Resource Companion (unisystem_5E_Pathfinder_P2E_OSR)'))


def flex(role, size, stance):
    c = fc.get_cell(T, role, size, stance, 'B'); o = collections.Counter(); t = collections.Counter()
    for (oc, _), rg in c['outcomes'].items():
        if rg: o[oc] += rg[1] - rg[0] + 1
    for k, rg in c['targeting'].items():
        if rg: t[k] += rg[1] - rg[0] + 1
    n = lambda d: {k: v / sum(d.values()) for k, v in d.items()}
    return n(o), n(t)


npc = []
for P in PAIRS:
    row = dict(id=P['id'], metric=list(P['metric']), sign=P['sign'])
    for side in ('a', 'b'):
        desc, cell = P[side]
        fo, ft = flex(*cell)
        row[side] = dict(state=f"D&D 5e combat, the creature's turn.\nCreature: {desc}\n{PARTY}",
                         flex_cell=list(cell), flexai=dict(outcome=fo, target=ft))
    npc.append(row)
save('e5_npc', npc, 'paired scenarios (experimenter-written) + FlexAI d100 distributions')

# ---- E6: fine-tuning items (derived from E3 / E4 exactly as the 2026-10-03 run built them) --
sys.path.insert(0, str(HERE))
from tasks import Q_TRIAGE, Q_FACT  # noqa: E402
VALID = {'name': ['campaign'], 'chatter': ['realworld'], 'leave': ['filler', 'rules', 'realworld']}
save('e6_triage_items', [dict(session=r['session'], token=r['token'], label=r['label'], state=r['state'],
                              questions=Q_TRIAGE, valid=VALID[r['label']]) for r in tri],
     'E3 rows + valid option set (valid-k loss)')
save('e6_typing_items', [dict(key=r['key'], gold=r['gold'], orig=r['orig'], set=s, state=r['state'],
                              questions=Q_FACT, valid=[r['gold']])
                         for s, src in (('repair', 'facts.json'), ('control', 'ctrl.json')) for r in load(ET / src)],
     'E4 repair + control facts, gold = GM primary type')

git = lambda d: subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
manifest = dict(version=OUT.name, frozen_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                campaign_commit=git(CAMPAIGN), dgx_fun_commit=git(str(CLEF.parent)),
                campaigngenerator_commit=git('/home/kostadis/src/CampaignGenerator'), files=files)
json.dump(manifest, open(OUT / 'MANIFEST.json', 'w'), indent=1)
os.chmod(OUT, 0o755)
for p in OUT.glob('*.json'): os.chmod(p, 0o444)   # read-only: a frozen set is not edited
print('frozen ->', OUT)
