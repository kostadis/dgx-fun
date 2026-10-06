# Frozen-dataset rerun of the decision-models paper

`paper/main.tex` reports experiments E1–E7. The original scripts in `../spell-pass-eval`,
`../ensemble-typing-eval`, `../social-npc-eval` and `../finetune` rebuilt their inputs from the live
campaign tree on every run, so results drifted as the campaign grew. This directory freezes the
inputs once and runs every model against the same bytes.

| file | what |
|---|---|
| `freeze.py` | snapshot every experiment input (with the model-facing `state` already rendered) into `~/data/decision-eval/<version>/`, read-only, plus `MANIFEST.json` (sha256 + counts + source commits) |
| `tasks.py` | questions, prompts and model endpoints, copied verbatim from the original scripts |
| `run.py MODEL EXP...` | run one model on E1–E5; results to `<data>/results/<part>/<model>.json`; never overwrites |
| `e6.sh` | E6 fine-tuning of Decision-2.0-Nox-4B on spark2 from the frozen items |
| `score.py` | every table in the paper, from the results |
| `MANIFEST-v1.json` | copy of the frozen set's manifest (hashes only, no content) |

The dataset itself is **not in the repo**: it quotes play sessions verbatim and the table-chatter
labels name real people. It lives at `~/data/decision-eval/v1/` on the workstation.

## Reproduce

```bash
# 1. build inputs from the campaign (slow: build2.py shells out per card)
cd ../spell-pass-eval && python3 build.py && python3 build2.py && python3 build3.py && python3 build_pick.py
cd ../ensemble-typing-eval && python3 build_ent.py && python3 build_fact.py && python3 build_ctrl.py \
  && for t in 0.5 0.55 0.6 0.67 0.75; do python3 replay_489.py $t; done
# 2. freeze
cd ../eval && python3 freeze.py ~/data/decision-eval/v2
# 3. run (serve each model first; see current-setup.md and spin-up-clef.sh / spin-up-decision2.sh)
DATA=~/data/decision-eval/v2 python3 run.py nox e1 e2 e3 e4 e5     # ... per model
DATA=~/data/decision-eval/v2 ./e6.sh
DATA=~/data/decision-eval/v2 python3 score.py --json scores.json
```

Models: `clef`, `clef-flash` (spark2:8002), `lux` (:8003), `kai` (:8004), `nox` (:8005),
`jev` (TypeSafe hosted API, key in `~/.jev-api`), `qwen` / `qwen-think` (qwen3.8-flash-next on
spark1:8001, reasoning off / on). Local SystemOne models run single-stream; Jev's latency
includes the internet round trip and is not comparable to the GB10 numbers.

Running `jev` sends the frozen states (campaign transcript excerpts) to TypeSafe's API.
