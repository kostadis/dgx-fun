"""Render a local 5etools adventure once for all chunking arms.

Module text, rendered text and question evidence remain in ignored raw/.
No campaign state, index or database is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inline(text):
    def replace(match):
        parts = match[2].split("|")
        tag = match[1]
        if tag == "dc":
            return "DC " + parts[0]
        if tag == "recharge":
            return "Recharge " + parts[0] + ("–6" if parts[0] != "6" else "")
        if tag == "atk":
            return {"mw": "Melee Weapon Attack", "rw": "Ranged Weapon Attack",
                    "ms": "Melee Spell Attack", "rs": "Ranged Spell Attack"}.get(parts[0], parts[0])
        if tag in {"creature", "item", "spell", "background", "condition", "race", "class", "feat"}:
            return parts[2] if len(parts) > 2 and parts[2] else parts[0]
        # Area/quickref tail fields are renderer flags and indexes, not labels.
        return parts[0]
    result = str(text).replace("{@h}", "Hit: ").replace("{@recharge}", "Recharge 6")
    while True:
        updated = re.sub(r"\{@(\w+) ([^{}]+)\}", replace, result)
        if updated == result:
            return result
        result = updated


def render_document(node, index, module_id="t14", references=None, resolved=None):
    blocks = []
    references = references or {}
    resolved = resolved if resolved is not None else set()

    def emit(text, path, kind="paragraph", page=None):
        if text.strip():
            blocks.append({"text": text + "\n\n", "path": list(path), "kind": kind, "page": page})

    def cell(value):
        if isinstance(value, str):
            return inline(value).replace("|", "\\|")
        if isinstance(value, list):
            return "; ".join(cell(v) for v in value)
        if isinstance(value, dict):
            if "roll" in value:
                roll = value["roll"]
                if "exact" in roll:
                    return str(roll["exact"])
                return str(roll.get("min", "")) + "–" + str(roll.get("max", ""))
            name = inline(value.get("name", ""))
            content = value.get("entry", value.get("entries", value.get("text")))
            if content is not None:
                return (name + ": " if name else "") + cell(content)
            return "; ".join(str(k) + ": " + cell(v) for k, v in value.items() if k != "type")
        return str(value)

    def visit_entity(entity, prop, path, depth, page):
        if prop == "monsterFluff":
            visit(entity.get("entries", []), path, depth, page)
            visit(entity.get("images", []), path, depth, page)
            return
        prose_fields = {"entries", "trait", "action", "bonus", "reaction", "legendary", "mythic", "spellcasting"}
        ignore = {"name", "source", "page", "otherSources", "additionalSources", "reprintedAs", "srd",
                  "basicRules", "isNpc", "isNamedCreature", "token", "tokenHref", "soundClip", "fluff", "images"}
        for key, value in entity.items():
            if key in ignore or key.startswith("_") or key.startswith("has") or key.endswith("Tags"):
                continue
            if key in prose_fields:
                if key != "entries":
                    emit("#" * min(depth, 6) + " " + key.title(), path, "heading", page)
                visit(value, path, depth + 1, page)
            else:
                emit(key + ": " + cell(value), path, "statistic", page)
        if prop == "item":
            for uid in entity.get("items", []):
                if not isinstance(uid, str):
                    continue
                parts = uid.split("|")
                source = parts[1] if len(parts) > 1 else "DMG"
                if "|".join(("item", source, parts[0])) in references:
                    visit({"type": "statblock", "tag": "item", "name": parts[0], "source": source},
                          path, depth, page)

    def visit(value, path=(), depth=1, page=None):
        if isinstance(value, str):
            emit(inline(value), path, page=page)
        elif isinstance(value, list):
            for item in value:
                visit(item, path, depth, page)
        elif isinstance(value, dict):
            kind = value.get("type", "entries")
            page = value.get("page", page)
            if kind == "statblock":
                prop = value.get("tag", value.get("prop", "unknown"))
                key = "|".join((prop, value.get("source", module_id), value.get("name", "")))
                if value.get("name") and (not path or path[-1] != value["name"]):
                    path = (*path, inline(value["name"]))
                    emit("#" * min(depth, 6) + " " + inline(value["name"]), path, "heading", page)
                    depth += 1
                emit("[Statblock: " + key + "]", path, "reference", page)
                if key in references:
                    resolved.add(key)
                    entity = references[key]
                    visit_entity(entity, prop, path, depth, entity.get("page", page))
                elif references:
                    raise ValueError("Unresolved reference: " + key)
                return
            if kind == "gallery":
                visit(value.get("images", []), path, depth, page)
                return
            name = inline(value.get("name", ""))
            if name:
                path = (*path, name)
                emit("#" * min(depth, 6) + " " + name, path, "heading", page)
                depth += 1
            if kind == "table":
                lines = []
                if value.get("caption"):
                    lines.append(inline(value["caption"]))
                labels = value.get("colLabels", [])
                if labels:
                    lines += ["| " + " | ".join(map(cell, labels)) + " |",
                              "| " + " | ".join("---" for _ in labels) + " |"]
                lines += ["| " + " | ".join(map(cell, row.get("row", []) if isinstance(row, dict) else row)) + " |"
                          for row in value.get("rows", [])]
                emit("\n".join(lines), path, "table", page)
            elif kind == "list":
                emit("\n".join("- " + cell(item) for item in value.get("items", [])), path, "list", page)
            elif kind == "image":
                # Text retrieval does not interpret map pixels. Keep the caption.
                if value.get("title"):
                    emit("[Map/image: " + inline(value["title"]) + "]", path, "image_caption", page)
            elif kind != "internal":
                visit(value.get("entries", []), path, depth, page)
                if "entry" in value:
                    visit(value["entry"], path, depth, page)
                if "headerEntries" in value or "spells" in value or "daily" in value:
                    for key in ("headerEntries", "spells", "daily", "will", "rest", "weekly", "yearly",
                                "ritual", "recharge", "charges", "footerEntries"):
                        if key in value:
                            emit(key + ": " + cell(value[key]), path, "statistic", page)

    visit(node)
    offset = 0
    for block in blocks:
        block["start"] = offset
        offset += len(block["text"])
        block["end"] = offset
    return {"id": f"{module_id}-{index:02d}", "title": node.get("name", "Untitled"),
            "text": "".join(b["text"] for b in blocks), "blocks": blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-title")
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    wrapper = data["adventureData"][0]
    module_id = wrapper["id"].lower()
    refs = json.loads(args.references.read_text()) if args.references else {}
    resolved = set()
    documents = [render_document(node, i, module_id, refs.get("references"), resolved)
                 for i, node in enumerate(wrapper["data"])]
    if refs and resolved != set(refs["references"]):
        raise ValueError("Not every exported reference was rendered")
    result = {"source_path": str(args.source.resolve()),
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "module_id": module_id, "module_title": args.module_title or wrapper["data"][0]["name"],
              "resolved_references": len(resolved),
              "reference_export_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest() if args.references else None,
              "reference_source_files": refs.get("source_files", {}),
              "documents": documents}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps({"documents": len(documents), "characters": sum(len(d["text"]) for d in documents),
                      "blocks": sum(len(d["blocks"]) for d in documents)}))


if __name__ == "__main__":
    main()
