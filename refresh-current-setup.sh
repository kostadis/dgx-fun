#!/usr/bin/env bash
# refresh-current-setup.sh
#
# Probe both DGX Sparks for the model ids they're actually serving,
# update the top-of-doc anchor in current-setup.md (the copy-paste
# block + the snapshot date), commit, and push to origin.
#
# What it touches in current-setup.md:
#   1. The fenced model-id block right after "Current `vllm-chat`
#      model ids" (the line clients copy-paste from).
#   2. The "Snapshot ... as of YYYY-MM-DD" date line.
#
# What it does NOT touch:
#   The prose sections (§2 vllm-embed, §3 vllm-chat on spark1, §4
#   vllm-chat on spark2, §7 client configs, §8 rebuild order). Those
#   describe *why* and *how*, and need a human to rewrite when a
#   model swaps. If the live model id doesn't appear in the prose,
#   the script warns loudly but still commits the anchor — the top
#   line of the doc must stay honest.
#
# Bails out if:
#   - any /v1/models probe fails (can't sync from unknown truth)
#   - current-setup.md already has uncommitted changes (don't
#     bundle unrelated prose edits into an auto-commit)
#
# Exits:
#   0  — synced, or already in sync
#   1  — probe failure or dirty working tree
#   2  — anchor was updated and pushed, but prose drift detected
#        (live model id missing from §2/§3/§4) — human follow-up needed
set -euo pipefail

# -C / --cluster: cross-box mode. In a 2-node Ray cluster the model is
# served only on spark1 :8001; spark2 :8001 is a Ray worker and serves
# nothing, so its /v1/models probe legitimately fails. With this flag we
# treat spark1 :8001 as the single cluster endpoint and mirror its model
# id into the spark2 anchor line (matching the doc convention that both
# lines show the same cluster model). Without it, an unreachable
# spark2 :8001 is a hard error.
CLUSTER=0
usage() {
  echo "Usage: $(basename "$0") [-C|--cluster]" >&2
  echo "  -C, --cluster   cross-box mode: spark1 :8001 is the only" >&2
  echo "                  serving endpoint; don't fail on spark2 :8001." >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -C|--cluster) CLUSTER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC="$REPO_DIR/current-setup.md"
TODAY="$(date -u +%Y-%m-%d)"

SPARK1_HOST="192.168.1.147"
SPARK2_HOST="192.168.1.69"

probe() {
  local url="$1"
  curl -sS --max-time 5 "$url" 2>/dev/null || true
}

first_model_id() {
  python3 -c '
import json, sys
try:
    data = json.loads(sys.stdin.read() or "{}")
    items = data.get("data", [])
    print(items[0].get("id", "") if items else "")
except Exception:
    print("")
' <<<"$1"
}

# --- probe ---
SPARK1_EMBED_JSON="$(probe "http://${SPARK1_HOST}:8000/v1/models")"
SPARK1_CHAT_JSON="$(probe "http://${SPARK1_HOST}:8001/v1/models")"
SPARK2_CHAT_JSON="$(probe "http://${SPARK2_HOST}:8001/v1/models")"

SPARK1_EMBED_MODEL="$(first_model_id "$SPARK1_EMBED_JSON")"
SPARK1_CHAT_MODEL="$(first_model_id "$SPARK1_CHAT_JSON")"
SPARK2_CHAT_MODEL="$(first_model_id "$SPARK2_CHAT_JSON")"

# In cluster mode the cluster is served only on spark1 :8001. Mirror that
# model id into the spark2 anchor line so the doc shows both endpoints
# pointing at the same cluster model, and don't treat an idle spark2 :8001
# as a failure.
if [[ $CLUSTER -eq 1 ]]; then
  SPARK2_CHAT_MODEL="$SPARK1_CHAT_MODEL"
fi

echo "Live state:"
[[ $CLUSTER -eq 1 ]] && echo "  (cluster mode: spark1 :8001 is the single cluster endpoint)"
printf "  spark1 :8000 embed = %s\n" "${SPARK1_EMBED_MODEL:-(unreachable)}"
printf "  spark1 :8001 chat  = %s\n" "${SPARK1_CHAT_MODEL:-(unreachable)}"
if [[ $CLUSTER -eq 1 ]]; then
  printf "  spark2 :8001 chat  = %s (mirrored from cluster head)\n" "${SPARK2_CHAT_MODEL:-(unreachable)}"
else
  printf "  spark2 :8001 chat  = %s\n" "${SPARK2_CHAT_MODEL:-(unreachable)}"
fi

fail=0
[[ -z "$SPARK1_CHAT_MODEL" ]] && { echo "ERROR: spark1 :8001 unreachable or no model loaded." >&2; fail=1; }
if [[ $CLUSTER -eq 0 ]]; then
  [[ -z "$SPARK2_CHAT_MODEL" ]] && { echo "ERROR: spark2 :8001 unreachable or no model loaded." >&2; fail=1; }
fi
[[ -z "$SPARK1_EMBED_MODEL" ]] && echo "WARN: spark1 :8000 (embed) unreachable — not updated in doc." >&2
if [[ $fail -ne 0 ]]; then
  echo "Refusing to update doc from incomplete probe." >&2
  exit 1
fi

# --- guard against bundling unrelated edits ---
cd "$REPO_DIR"
if ! git diff --quiet -- current-setup.md; then
  echo "ERROR: current-setup.md has uncommitted changes." >&2
  echo "Commit or stash them before running this script, so the" >&2
  echo "auto-commit only contains the probe-driven anchor refresh." >&2
  exit 1
fi
if ! git diff --cached --quiet -- current-setup.md; then
  echo "ERROR: current-setup.md has staged changes. Unstage or commit them first." >&2
  exit 1
fi

# --- rewrite ---
DRIFT_FLAG="$(mktemp)"
trap 'rm -f "$DRIFT_FLAG"' EXIT

python3 - "$DOC" "$SPARK1_CHAT_MODEL" "$SPARK2_CHAT_MODEL" "$TODAY" "$DRIFT_FLAG" <<'PY'
import re, sys, pathlib

doc_path, s1_chat, s2_chat, today, drift_flag = sys.argv[1:6]
p = pathlib.Path(doc_path)
src = p.read_text()
original = src

# 1. Anchor block: replace the fenced spark1/spark2 model-id pair.
anchor_re = re.compile(
    r"```\nspark1 \(192\.168\.1\.147:8001\):\s+\S.*?\n"
    r"spark2 \(192\.168\.1\.69:8001\):\s+\S.*?\n```",
    re.DOTALL,
)
new_anchor = (
    "```\n"
    f"spark1 (192.168.1.147:8001):  {s1_chat}\n"
    f"spark2 (192.168.1.69:8001):   {s2_chat}\n"
    "```"
)
src, n_anchor = anchor_re.subn(new_anchor, src, count=1)
if n_anchor == 0:
    print("ERROR: could not find the model-id anchor block in current-setup.md.", file=sys.stderr)
    print("Doc structure may have changed; refusing to edit.", file=sys.stderr)
    sys.exit(3)

# 2. Snapshot date.
# Match only the date token; preserve whatever follows it (a bare "."
# in older docs, or a " (single-box steady state…)" parenthetical that
# was added by hand). Replacing just the date keeps that trailing prose.
date_re = re.compile(
    r"(Snapshot of what's actually running on \*\*both\*\* DGX Sparks as of\n)\d{4}-\d{2}-\d{2}"
)
src, n_date = date_re.subn(rf"\g<1>{today}", src, count=1)
if n_date == 0:
    print("ERROR: could not find the snapshot date line in current-setup.md.", file=sys.stderr)
    sys.exit(3)

# 3. Drift check — every live model id should appear somewhere in the prose.
drift = []
for tag, mid in (("spark1 :8001", s1_chat), ("spark2 :8001", s2_chat)):
    if mid not in src:
        drift.append(f"{tag} model {mid!r} does not appear in prose — §3/§4 likely stale")

if src == original:
    print("Doc already in sync with live state (anchor + date unchanged).")
    sys.exit(0)

p.write_text(src)
print(f"Updated {doc_path}: anchor + snapshot date.")

if drift:
    print()
    print("DRIFT WARNING — prose sections describe a different model than live state:")
    for line in drift:
        print(f"  - {line}")
    print()
    print("The top anchor was synced, but §3 (spark1 vllm-chat) and/or §4")
    print("(spark2 vllm-chat) prose, run-command blocks, and §7 client")
    print("configs still describe the OLD model. Edit by hand before the")
    print("next session.")
    pathlib.Path(drift_flag).write_text("drift")
PY

py_exit=$?
if [[ $py_exit -ne 0 ]]; then
  echo "Python rewrite step failed (exit $py_exit). Aborting." >&2
  exit "$py_exit"
fi

# --- commit + push ---
if git diff --quiet -- current-setup.md; then
  echo "Nothing to commit."
  exit 0
fi

drift_note=""
if [[ -s "$DRIFT_FLAG" ]]; then
  drift_note=$'\n\nDrift warning: at least one live model id does not appear in the\nprose. §3 and/or §4 still describe an older model — follow-up\ncommit needed by hand.'
fi

embed_line=""
if [[ -n "$SPARK1_EMBED_MODEL" ]]; then
  embed_line="- spark1 :8000 embed = ${SPARK1_EMBED_MODEL}"$'\n'
fi

git add current-setup.md
git commit -m "$(cat <<EOF
current-setup.md: refresh anchor + date from live spark state

Probed ${TODAY}:
${embed_line}- spark1 :8001 chat  = ${SPARK1_CHAT_MODEL}
- spark2 :8001 chat  = ${SPARK2_CHAT_MODEL}

Generated by refresh-current-setup.sh.${drift_note}
EOF
)"

branch="$(git rev-parse --abbrev-ref HEAD)"
echo "Pushing ${branch} to origin..."
git push origin "$branch"
echo "Done."

if [[ -s "$DRIFT_FLAG" ]]; then
  exit 2
fi
exit 0
