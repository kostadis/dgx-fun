"""Question definitions, prompts and model endpoints for the decision-models paper.

Questions and prompts are copied verbatim from the original per-experiment scripts
(spell-pass-eval/run*.py, triage.py, ensemble-typing-eval/run_*.py, social-npc-eval/run_npc.py)
so the frozen-dataset rerun asks exactly what the paper's runs asked.
"""
import json, os, re, time, urllib.error, urllib.request

# ---- endpoints -----------------------------------------------------------------------------
SPARK1, SPARK2 = '192.168.1.147', '192.168.1.121'
SYSTEMONE = {  # short name -> (url, served model id)
    'clef':       (f'http://{SPARK2}:8002/v1/systemone', 'clef'),
    'clef-flash': (f'http://{SPARK2}:8002/v1/systemone', 'clef-flash'),
    'lux':        (f'http://{SPARK2}:8003/v1/systemone', 'Decision-2.0-Lux-9B'),
    'kai':        (f'http://{SPARK2}:8004/v1/systemone', 'Decision-2.0-Kai-0.6B'),
    'nox':        (f'http://{SPARK2}:8005/v1/systemone', 'Decision-2.0-Nox-4B'),
    'jev':        ('https://api.typesafe.ai/v1/systemone', 'jev-1.13.0'),   # hosted, TypeSafe
}
CHAT = {'qwen': False, 'qwen-think': True}   # qwen3.8-flash-next on spark1, reasoning off / on
CHAT_URL, CHAT_MODEL = f'http://{SPARK1}:8001/v1/chat/completions', 'qwen3.8-flash-next'
MODELS = list(SYSTEMONE) + list(CHAT)


def _key():
    k = open(os.path.expanduser('~/.jev-api')).read().strip()
    return k.split('=', 1)[1].strip().strip('\'"') if '=' in k else k


def post(url, body, to=600, tries=6):
    h = {'content-type': 'application/json'}
    if 'typesafe.ai' in url: h['authorization'] = f'Bearer {_key()}'
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h), timeout=to))
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 503) and i < tries - 1:
                time.sleep(min(30, 2 ** i)); continue
            raise RuntimeError(f'{url} HTTP {e.code}: {e.read()[:300]!r}') from None


def systemone(model, state, questions):
    url, mid = SYSTEMONE[model]
    r = post(url, {'model': mid, 'state': state, 'questions': questions})
    return r['answers'], r.get('model', mid), r.get('usage', {})


def chat(prompt, think, max_tokens):
    body = {'model': CHAT_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': think}}
    return post(CHAT_URL, body)['choices'][0]['message']['content'] or ''


# ---- E1 ------------------------------------------------------------------------------------
Q_E1_PROSE = {
    "accept": {"type": "noul", "instructions": "A tool proposed this correction to a D&D session transcript. Will the dungeon master approve it exactly as written, with no change and no need to discuss it?"},
    "chatter": {"type": "noul", "instructions": "Is the name in question out-of-game table chatter (a real-world person, company, product, place, or a different game) rather than something inside this D&D campaign?"}}
Q_E1_RAW = {
    "accept": {"type": "noul", "instructions": "A tool proposed replacing the transcript token with the proposed replacement in a D&D session transcript. Will the dungeon master approve this replacement exactly as proposed, with no change and no need to discuss it?"}}

# ---- E2 ------------------------------------------------------------------------------------
E2_TEXT = "Speech recognition often misspells fantasy names in this D&D session transcript. The token under review will be replaced, word for word, by the option you pick. Which option is the correct spelling of exactly the words spoken in that token? Do not pick an option that adds words the speaker did not say (for example a first name or title when only a surname was spoken)."
E2_LEAVE = "Not a mangled campaign name: leave the words as spoken (table chatter, ordinary words, rules talk)"
E2_OTHER = "A misspelled campaign name, but none of the listed options is its correct spelling"


def e2_questions(r):
    crit = dict(r['options']); crit['LEAVE'] = E2_LEAVE; crit['OTHER'] = E2_OTHER
    return {"pick": {"type": "choice", "instructions": E2_TEXT, "criteria": crit}}


def e2_chat_prompt(r):
    opts = '\n'.join(f'- "{k}": {v}' for k, v in r['options'].items()) + f'\n- "LEAVE": {E2_LEAVE}\n- "OTHER": {E2_OTHER}'
    return (f"{r['state']}\n\nSpeech recognition often mangles fantasy names in this D&D session transcript. "
            f"{E2_TEXT} Options:\n{opts}\n\n"
            'Reply with JSON only: {"answer": "<option key exactly>", "confidence": <0 to 1>}')


def e2_parse(t):
    m = re.search(r'\{[^{}]*"answer"[^{}]*\}', t, re.S)
    try:
        j = json.loads(m.group(0)); return {"choice": j['answer'], "confidence": float(j.get('confidence', 0))}
    except Exception:
        return {"choice": None, "confidence": 0}


# ---- E3 ------------------------------------------------------------------------------------
TRI_CRIT = {"filler": "An ordinary English word, interjection, number, or sentence-start capital; not a name at all",
            "rules": "A D&D rules term, spell, monster type, or game mechanic",
            "realworld": "A real-world person, company, product, place, or another game or film; table chatter",
            "campaign": "A name from inside the fantasy campaign (character, place, faction, item, deity), possibly misheard by speech recognition"}
Q_TRIAGE = {"kind": {"type": "choice", "instructions": "Speech recognition produced this capitalised token in a transcript of a D&D game played over video call. What kind of token is it?", "criteria": TRI_CRIT}}

# ---- E4 ------------------------------------------------------------------------------------
FACT_CRIT = {"npc": "A specific named individual person or intelligent being (including a named, talking creature or sentient being)",
             "monster": "A kind of creature or an unnamed hostile creature encountered in play",
             "faction": "An organisation, house, family, guild, cult, government or group of people",
             "location": "A place: a city, settlement, building, room, region or landmark",
             "object": "An item, weapon, artifact, book or other physical thing",
             "event": "Something that happened: an action, fight, meeting or occurrence",
             "thread": "An open plot thread, goal, mystery, promise or unresolved question",
             "date": "A date, time, duration or calendar reference"}
Q_FACT = {"kind": {"type": "choice", "instructions": "This fact was extracted from a D&D campaign's session records. Which type should the fact be filed under? Choose by what the subject is (npc, monster, faction, location, object), unless the fact is really about an event, an open plot thread, or a date.", "criteria": FACT_CRIT}}
ENT_CRIT = {k: FACT_CRIT[k] for k in ('npc', 'monster', 'faction', 'location', 'object')}
Q_ENT = {"kind": {"type": "choice", "instructions": "These facts were extracted from records of a D&D campaign. What kind of entity is the named subject, for filing it in the campaign's dossiers?", "criteria": ENT_CRIT}}


def choice_chat_prompt(state, q):
    q = q['kind']
    return (state + "\n\n" + q['instructions'] + " Options:\n" + '\n'.join(f'- "{k}": {v}' for k, v in q['criteria'].items())
            + '\n\nReply with JSON only: {"answer": "<key>"}')


def choice_chat_parse(t, keys):
    m = re.search(r'"answer"\s*:\s*"(\w+)"', t); a = m.group(1) if m else None
    return {k: float(k == a) for k in keys}


# ---- E5 ------------------------------------------------------------------------------------
NPC_OUT = {"attack_main": "Attacks its target with its standard attack (melee for most creatures, ranged for some)",
           "attack_secondary": "Attacks with its other mode: ranged if its main attack is melee, melee if its main attack is ranged",
           "maneuver": "Moves: closes on its preferred target, evades the enemies around it, or takes advantage of the terrain",
           "use_defend": "Uses an item (potion, wand, staff) or, lacking one, takes a defensive stance",
           "ability": "Uses a special ability or spell against its target",
           "flee": "Tries to flee the fight entirely"}
NPC_TGT = {"frontline": "The front-most enemies", "rearguard": "The rear-most enemies", "closest": "The enemy physically closest to it",
           "farthest": "The enemy physically farthest from it", "strongest": "The healthiest enemy, furthest from death",
           "weakest": "The enemy closest to death", "ranged_enemy": "An enemy whose main attack is ranged", "melee_enemy": "An enemy whose main attack is melee"}
Q_NPC = {"outcome": {"type": "choice", "instructions": "What does this creature do on its turn?", "criteria": NPC_OUT},
         "target": {"type": "choice", "instructions": "Which enemy does this creature target this turn?", "criteria": NPC_TGT}}


def npc_chat_prompt(state):
    return (state + "\n\nGive a probability distribution for what this creature does this turn and whom it targets.\nOutcomes:\n"
            + '\n'.join(f'- "{k}": {v}' for k, v in NPC_OUT.items()) + "\nTargets:\n" + '\n'.join(f'- "{k}": {v}' for k, v in NPC_TGT.items())
            + '\n\nReply with JSON only: {"outcome": {<every outcome key>: probability}, "target": {<every target key>: probability}}')


def npc_chat_parse(t):
    norm = lambda d: {k: v / (sum(d.values()) or 1) for k, v in d.items()}
    j = json.loads(re.search(r'\{.*\}', t, re.S).group(0))
    return dict(outcome=norm({k: float(j['outcome'].get(k, 0)) for k in NPC_OUT}),
                target=norm({k: float(j['target'].get(k, 0)) for k in NPC_TGT}))
