"""Freeze campaign-based and synthetic fixtures for both serving arms."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--summaries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    sources = []
    parts = []
    for path in sorted(a.summaries.glob("*.md")):
        raw = path.read_bytes()
        sources.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        parts.append(f"\n\nARCHIVE DOCUMENT {path.name}\n" + raw.decode())
    corpus = "".join(parts)
    corpus_ids = tok.encode(corpus, add_special_tokens=False).ids
    if len(corpus_ids) < 200000:
        raise ValueError("Need 200K tokens of distinct campaign source; do not pad by repetition")

    def excerpt(n, offset=0):
        return tok.decode(corpus_ids[offset:offset + n], skip_special_tokens=False)

    facts = {"ORBIT": "cobalt-7391", "CEDAR": "amber-2846", "MAPLE": "silver-9052",
             "DELTA": "violet-6173", "ANCHOR": "copper-4385"}
    blocks = []
    for i, (key, value) in enumerate(facts.items()):
        blocks.append(excerpt(39700, i * 39700))
        blocks.append(f"\nBENCHMARK CONTROL FACT {key}={value}. This is a test marker, not campaign canon.\n")
    instruction = ("\nUsing only this archive, prepare the next session. Return a JSON object with "
                   "control_facts (the five BENCHMARK CONTROL FACT keys and exact values), "
                   "and prep (an array of three nonempty strings: situation, likely next scene, contingency). "
                   "Do not treat benchmark control facts as campaign canon. No Markdown.\n")
    campaign = "".join(blocks) + instruction
    # Fill from unused source, before the instruction, to approximately 200K user tokens.
    remaining = 200000 - len(tok.encode(campaign, add_special_tokens=False).ids)
    campaign = "".join(blocks) + excerpt(max(0, remaining), 198500) + instruction
    sessions = []
    for i, role in enumerate(("NPC innkeeper", "NPC courier", "coding agent", "coding reviewer")):
        code = f"session-{i}-key-{(i + 7) * 103}"
        sessions.append({"role": role, "code": code,
                         "context": f"Your private session memory code is {code}.\n" + excerpt(24000, i * 26000)})
    entities = [{"id": f"p{i:02}", "type": "person", "name": f"Agent {i:02}"} for i in range(40)]
    entities += [{"id": f"l{i:02}", "type": "place", "name": f"Depot {i:02}"} for i in range(40)]
    edges = [{"source": f"p{i:02}", "target": f"l{i:02}", "relation": "visits"} for i in range(40)]
    graph_text = "\n".join(f"Agent {i:02} (id p{i:02}) visits Depot {i:02} (id l{i:02})." for i in range(40))
    data = {"run_id": a.run_id, "sources": sources,
            "tokenizer_sha256": hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
            "campaign": {"text": campaign, "user_tokens": len(tok.encode(campaign, add_special_tokens=False).ids),
                         "facts": facts},
            "sessions": sessions,
            "structured": {"text": graph_text, "entities": entities, "relationships": edges}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, ensure_ascii=False))
    manifest = {"run_id": a.run_id, "campaign_user_tokens": data["campaign"]["user_tokens"],
                "source_documents": len(sources), "source_tokens": len(corpus_ids), "sources": sources,
                "tokenizer_sha256": data["tokenizer_sha256"],
                "fixture_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest()}
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}))


if __name__ == "__main__":
    main()
