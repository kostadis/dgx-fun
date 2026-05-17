# Spark management CLI — design + starter implementation

**Goal:** From the workstation, run one command instead of writing/re-running
a shell script every time you want to bring vllm-embed or vllm-chat up,
down, or with different parameters.

**Status:** not built. This doc is the spec + a ready-to-paste v1 starter.

**Authors:** Kostadis + Claude.

## What you actually want

From the workstation:

```bash
spark status                  # what's running on the Spark right now
spark up chat                 # start vllm-chat with its saved profile
spark down chat               # stop it
spark restart chat --max-num-batched-tokens 16384   # restart with one tweaked flag
spark logs chat -f            # tail container logs over the LAN
spark gpu                     # nvidia-smi on the Spark, here in the terminal
spark probe chat              # one curl to verify the endpoint responds
spark switch chat --model qwen2.5-32b   # swap which model the chat profile serves
spark profile list            # what profiles are saved
spark profile show chat       # the docker run invocation it expands to
spark install qwen2.5-32b     # pull a model from HF on the Spark side
```

That's the contract. Underneath: SSH to `kostadis@192.168.1.147` and run
`docker` / `nvidia-smi` / `huggingface-cli` commands. Profiles are YAML
files on the workstation describing each container's intended shape.

## Does this exist already?

**Closest existing OSS:** [OpenLLM by BentoML](https://github.com/bentoml/OpenLLM).

```bash
pip install openllm
openllm start Qwen/Qwen2.5-14B-Instruct-AWQ \
  --backend vllm --port 8001 --gpu-memory-utilization 0.5
```

It does abstract `docker run vllm/vllm-openai:...` for you. Has an
OpenAI-compatible endpoint by default. Models are first-class — `openllm
list models` shows what's supported, `openllm download X` fetches.

**Why it's not enough for this use case:**

* **Local-only by design.** Runs on the same host as the inference.
  You'd still need an SSH wrapper to drive it from the workstation.
  Half the problem unsolved.
* **Abstracts vLLM CLI flags.** Some of the things you actually want to
  tune — `--max-num-batched-tokens`, `--enable-chunked-prefill`,
  `--enable-lora`, `--swap-space` — go through OpenLLM's intermediate
  layer and are either renamed, not exposed, or have to be passed
  through an `--extra-args` escape hatch. The pass-through works but
  it's exactly the friction you're trying to remove.
* **One container at a time as a first-class concept.** Running
  vllm-embed *and* vllm-chat simultaneously with different
  `--gpu-memory-utilization` caps is fine but not the primary shape
  the CLI models — you'd be running two `openllm start` instances and
  tracking them yourself.

**Other tools considered and rejected:**

* `vllm` CLI itself — bare metal, no orchestration. What's already in
  use.
* `docker compose` — works fine for fixed configs, no model-awareness,
  fiddly to swap models in/out.
* NVIDIA Triton Inference Server — production-grade, heavyweight,
  geared at multi-replica clusters. Massive overkill for one box.
* Kubernetes + helm chart for vLLM — same problem at greater expense.
* RunPod / Lambda / SageMaker CLIs — cloud-only.
* NVIDIA NIM — packaged inference containers, requires NVIDIA account,
  not generic.

**Verdict:** a thin custom CLI is the right call. ~200–300 lines of
Python. You keep the full vLLM CLI surface (the part you actually want
to tune) and add a remote-control layer + profile system on top.

## Architecture

```
┌─────────────────────────┐         ssh         ┌─────────────────────┐
│ workstation             │  ──────────────►    │ Spark (192.168.1.147)│
│                         │                     │                      │
│ ~/.config/spark/        │                     │ docker daemon        │
│   profiles/             │                     │   ┌──────────────┐   │
│     chat.yaml           │                     │   │ vllm-chat    │   │
│     embed.yaml          │                     │   ├──────────────┤   │
│   config.toml           │                     │   │ vllm-embed   │   │
│                         │                     │   └──────────────┘   │
│ ~/src/spark-cli/        │                     │                      │
│   spark/__main__.py     │                     │ ~/.cache/huggingface │
└─────────────────────────┘                     └─────────────────────┘
```

### Profiles

YAML at `~/.config/spark/profiles/<name>.yaml`. One file per container.
Field names mirror the `docker run` invocation 1:1 — **no abstraction**.

```yaml
# ~/.config/spark/profiles/chat.yaml
container_name: vllm-chat
image: vllm/vllm-openai:latest
runtime: nvidia
gpus: all
ipc: host
port: 8001
volume:
  - ~/.cache/huggingface:/root/.cache/huggingface
model: Qwen/Qwen2.5-14B-Instruct-AWQ
vllm_args:
  max-model-len: 32768
  gpu-memory-utilization: 0.5
  max-num-batched-tokens: 4096          # tweakable knob
  host: 0.0.0.0
  port: 8001
```

```yaml
# ~/.config/spark/profiles/embed.yaml
container_name: vllm-embed
image: vllm/vllm-openai:latest
runtime: nvidia
gpus: all
ipc: host
port: 8000
volume:
  - ~/.cache/huggingface:/root/.cache/huggingface
model: nomic-ai/nomic-embed-text-v1.5
vllm_args:
  trust-remote-code: true
  gpu-memory-utilization: 0.05
  host: 0.0.0.0
  port: 8000
```

A profile expands deterministically to a `docker run` invocation. CLI
flags can override profile values for a single invocation:

```bash
spark up chat --max-num-batched-tokens 16384
# expands to: docker run ... --max-num-batched-tokens 16384 ... (overrides profile)
```

### Workstation config

```toml
# ~/.config/spark/config.toml
[host]
ssh_target = "kostadis@192.168.1.147"
ssh_key = "~/.ssh/id_ed25519"

[defaults]
hf_cache = "~/.cache/huggingface"
```

### CLI commands

| command | does what |
| --- | --- |
| `spark status` | `ssh ... docker ps --format ...` formatted as a table |
| `spark up <profile>` | renders profile → `docker run` → ssh to Spark → executes |
| `spark down <profile>` | `ssh ... docker stop <container_name>` |
| `spark restart <profile> [--flag val]` | down + up with optional overrides |
| `spark logs <profile> [-f]` | `ssh -t ... docker logs <container_name> [-f]` |
| `spark gpu` | `ssh ... nvidia-smi` |
| `spark probe <profile>` | `curl <host>:<port>/v1/models` and report |
| `spark switch <profile> --model <hf-id>` | rewrite profile's `model` and restart |
| `spark install <model>` | `ssh ... huggingface-cli download <model>` |
| `spark profile list` | enumerate `~/.config/spark/profiles/*.yaml` |
| `spark profile show <name>` | render the profile to stdout (the docker run invocation) |
| `spark profile edit <name>` | `$EDITOR ~/.config/spark/profiles/<name>.yaml` |
| `spark profile new <name>` | scaffold a new profile from a template |

### Failure modes the tool handles

* **SSH fails:** report the SSH error cleanly, don't pretend it worked.
* **Container already running** on `spark up`: ask if you want `restart`
  instead, exit non-zero if not confirmed.
* **Wait for healthy** on `spark up`: poll `/v1/models` for up to 5 min
  before returning (vllm-chat takes ~3 min to warm-restart per the setup
  doc). Tail logs while waiting so you see the
  `Application startup complete.` line in real time.
* **Profile parse error:** `spark profile show <name>` shows the
  invocation, lets you diff before running.
* **HF model not present:** detect by inspecting the cache, prompt to
  `spark install <model>` first.

## Starter implementation (paste-ready)

Single file Python with `click`. ~200 lines. Lives at
`~/src/spark-cli/spark/__main__.py`. Distribute via `pipx install -e
~/src/spark-cli`.

```python
# ~/src/spark-cli/spark/__main__.py
"""Spark management CLI. SSH-shim around docker on the DGX Spark."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import click
import tomllib  # py >= 3.11
import yaml

CONFIG_DIR = Path(os.environ.get("SPARK_CONFIG_DIR", "~/.config/spark")).expanduser()
PROFILES_DIR = CONFIG_DIR / "profiles"


def _load_config() -> dict:
    cfg_path = CONFIG_DIR / "config.toml"
    if not cfg_path.exists():
        sys.exit(f"No config at {cfg_path}. Create one with `spark init`.")
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def _load_profile(name: str) -> dict:
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"No profile {name!r} at {path}.")
    with open(path) as f:
        return yaml.safe_load(f)


def _save_profile(name: str, data: dict) -> None:
    path = PROFILES_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _ssh_target() -> str:
    cfg = _load_config()
    return cfg["host"]["ssh_target"]


def _ssh_run(remote_cmd: str, *, capture: bool = False, tty: bool = False) -> str:
    """Run a command on the Spark over SSH."""
    target = _ssh_target()
    ssh_args = ["ssh"]
    if tty:
        ssh_args.append("-t")
    ssh_args += [target, remote_cmd]
    if capture:
        result = subprocess.run(ssh_args, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"SSH failed:\n{result.stderr}")
        return result.stdout
    result = subprocess.run(ssh_args)
    if result.returncode != 0:
        sys.exit(f"SSH command failed (exit {result.returncode}).")
    return ""


def _profile_to_docker_run(profile: dict, overrides: dict | None = None) -> str:
    """Render profile + flag overrides to a docker run invocation."""
    overrides = overrides or {}
    p = profile
    parts = ["docker", "run", "-d"]
    parts += ["--runtime", p.get("runtime", "nvidia")]
    if p.get("gpus"):
        parts += ["--gpus", str(p["gpus"])]
    parts += ["--name", p["container_name"]]
    if p.get("port"):
        parts += ["-p", f"{p['port']}:{p['port']}"]
    if p.get("ipc"):
        parts += ["--ipc", str(p["ipc"])]
    for vol in p.get("volume", []):
        parts += ["-v", vol]
    parts.append(p["image"])
    parts.append(p["model"])

    vllm_args = {**p.get("vllm_args", {}), **overrides}
    for k, v in vllm_args.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts += [flag, str(v)]

    return shlex.join(parts)


@click.group()
def main():
    """Spark management CLI."""


@main.command()
def status():
    """List containers running on the Spark."""
    out = _ssh_run(
        "docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
        capture=True,
    )
    click.echo(out)


@main.command()
@click.argument("profile_name")
@click.option("--max-num-batched-tokens", type=int, default=None)
@click.option("--gpu-memory-utilization", type=float, default=None)
@click.option("--max-model-len", type=int, default=None)
def up(profile_name: str, **overrides):
    """Bring up a profile."""
    overrides = {k.replace("_", "-"): v for k, v in overrides.items() if v is not None}
    profile = _load_profile(profile_name)
    cmd = _profile_to_docker_run(profile, overrides=overrides)
    click.echo(f"$ {cmd}")
    _ssh_run(cmd)
    click.echo(f"\nStarted {profile['container_name']}.")


@main.command()
@click.argument("profile_name")
def down(profile_name: str):
    """Stop a profile's container."""
    profile = _load_profile(profile_name)
    name = profile["container_name"]
    _ssh_run(f"docker stop {name} && docker rm {name}")
    click.echo(f"Stopped + removed {name}.")


@main.command()
@click.argument("profile_name")
@click.option("--max-num-batched-tokens", type=int, default=None)
@click.option("--gpu-memory-utilization", type=float, default=None)
@click.option("--max-model-len", type=int, default=None)
@click.pass_context
def restart(ctx, profile_name: str, **overrides):
    """Down + up with optional flag overrides."""
    ctx.invoke(down, profile_name=profile_name)
    ctx.invoke(up, profile_name=profile_name, **overrides)


@main.command()
@click.argument("profile_name")
@click.option("-f", "--follow", is_flag=True)
def logs(profile_name: str, follow: bool):
    """Tail container logs."""
    profile = _load_profile(profile_name)
    flag = "-f" if follow else ""
    _ssh_run(f"docker logs {flag} {profile['container_name']}", tty=follow)


@main.command()
def gpu():
    """nvidia-smi on the Spark."""
    out = _ssh_run("nvidia-smi", capture=True)
    click.echo(out)


@main.command()
@click.argument("profile_name")
def probe(profile_name: str):
    """Smoke-test an endpoint."""
    profile = _load_profile(profile_name)
    port = profile["port"]
    target = _ssh_target().split("@")[-1]
    url = f"http://{target}:{port}/v1/models"
    click.echo(f"$ curl {url}")
    subprocess.run(["curl", "-sS", "--max-time", "5", url])
    click.echo()


@main.command()
@click.argument("model")
def install(model: str):
    """Download a model into the Spark's HF cache."""
    _ssh_run(f"huggingface-cli download {shlex.quote(model)}")


@main.command()
@click.argument("profile_name")
@click.option("--model", required=True, help="New HF model id, e.g. Qwen/Qwen2.5-32B-Instruct-AWQ")
@click.pass_context
def switch(ctx, profile_name: str, model: str):
    """Swap the model in a profile and restart it."""
    profile = _load_profile(profile_name)
    profile["model"] = model
    _save_profile(profile_name, profile)
    click.echo(f"Updated {profile_name}.yaml model → {model}")
    ctx.invoke(restart, profile_name=profile_name)


@main.group()
def profile():
    """Profile management."""


@profile.command("list")
def profile_list():
    if not PROFILES_DIR.exists():
        sys.exit(f"No profiles dir at {PROFILES_DIR}.")
    for p in sorted(PROFILES_DIR.glob("*.yaml")):
        click.echo(p.stem)


@profile.command("show")
@click.argument("name")
def profile_show(name: str):
    profile = _load_profile(name)
    click.echo(_profile_to_docker_run(profile))


@profile.command("edit")
@click.argument("name")
def profile_edit(name: str):
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        sys.exit(f"No profile {name!r}.")
    editor = os.environ.get("EDITOR", "vim")
    subprocess.run([editor, str(path)])


@main.command("init")
def init_cmd():
    """Scaffold ~/.config/spark/ with a starter config and profiles."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_DIR / "config.toml"
    if not cfg_path.exists():
        cfg_path.write_text(
            '[host]\n'
            'ssh_target = "kostadis@192.168.1.147"\n'
            'ssh_key = "~/.ssh/id_ed25519"\n\n'
            '[defaults]\n'
            'hf_cache = "~/.cache/huggingface"\n'
        )
        click.echo(f"Wrote {cfg_path}")
    chat_path = PROFILES_DIR / "chat.yaml"
    if not chat_path.exists():
        chat_path.write_text(
            "container_name: vllm-chat\n"
            "image: vllm/vllm-openai:latest\n"
            "runtime: nvidia\n"
            "gpus: all\n"
            "ipc: host\n"
            "port: 8001\n"
            "volume:\n"
            "  - ~/.cache/huggingface:/root/.cache/huggingface\n"
            "model: Qwen/Qwen2.5-14B-Instruct-AWQ\n"
            "vllm_args:\n"
            "  max-model-len: 32768\n"
            "  gpu-memory-utilization: 0.5\n"
            "  host: 0.0.0.0\n"
            "  port: 8001\n"
        )
        click.echo(f"Wrote {chat_path}")
    embed_path = PROFILES_DIR / "embed.yaml"
    if not embed_path.exists():
        embed_path.write_text(
            "container_name: vllm-embed\n"
            "image: vllm/vllm-openai:latest\n"
            "runtime: nvidia\n"
            "gpus: all\n"
            "ipc: host\n"
            "port: 8000\n"
            "volume:\n"
            "  - ~/.cache/huggingface:/root/.cache/huggingface\n"
            "model: nomic-ai/nomic-embed-text-v1.5\n"
            "vllm_args:\n"
            "  trust-remote-code: true\n"
            "  gpu-memory-utilization: 0.05\n"
            "  host: 0.0.0.0\n"
            "  port: 8000\n"
        )
        click.echo(f"Wrote {embed_path}")


if __name__ == "__main__":
    main()
```

```toml
# ~/src/spark-cli/pyproject.toml
[project]
name = "spark-cli"
version = "0.1.0"
description = "Manage vLLM containers on the DGX Spark from the workstation."
requires-python = ">=3.11"
dependencies = ["click", "pyyaml"]

[project.scripts]
spark = "spark.__main__:main"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

## Bootstrap (sub-1-hour to working v1)

```bash
mkdir -p ~/src/spark-cli/spark
cd ~/src/spark-cli
# paste the pyproject.toml and the spark/__main__.py from above

# install in editable mode (pipx isolates each tool in its own venv)
pipx install -e .

# scaffold config + default profiles
spark init

# smoke-test
spark status        # should list vllm-chat and vllm-embed (currently running)
spark probe chat    # should hit http://192.168.1.147:8001/v1/models
spark gpu           # should print nvidia-smi from the Spark
```

After this, restarting vllm-chat with a tweak becomes:

```bash
spark restart chat --max-num-batched-tokens 16384
```

The 3-minute warm-restart still happens on the Spark side — but you only
type one command instead of crafting a docker run line by hand.

## Phase 2 features (not in v1)

* **`spark up chat --wait-ready`** — poll `/v1/models` until the
  container reports startup complete. Hide the 3-minute warm-restart
  from the terminal experience entirely.
* **`spark snapshot`** — write a timestamped record of which container
  is serving which model with which args. Useful if you want to roll
  back to "Tuesday's config" without remembering the flags.
* **`spark exec chat <cmd>`** — `docker exec` shim. e.g.
  `spark exec chat python -c "import torch; print(torch.cuda.is_available())"`.
* **`spark fine-tune <profile> --data ./examples.jsonl`** — wraps the
  unsloth + LoRA workflow from `finetune-qwen-on-dnd-plan.md` so the
  entire fine-tune lifecycle is one command per stage.
* **Multi-Spark support** — if you ever get a second box, profile files
  could include `host: spark2` and the tool picks the right SSH target.
  Trivial to add to `_ssh_target()`.

## What's deliberately NOT here

* **No daemon, no API server, no UI.** Just a CLI. You already know how
  to script around CLIs.
* **No assumption about model registry.** The tool defers to HF cache
  on the Spark side. If a model isn't pulled, `docker run` fails and
  the tool surfaces the failure. You re-run with `spark install` to
  fix.
* **No vLLM flag abstraction.** Every vLLM CLI argument is a direct
  pass-through. When vLLM adds a new flag, you write the new flag name
  in your profile YAML and it Just Works. No tool update needed for
  vLLM evolution.
* **No state caching on workstation side.** Source of truth is `docker
  ps` on the Spark. The tool never gets out of sync with reality.

## How to pick this up

Open a new Claude session at `~/src/` and start with:

> "We have a spec at `~/src/dgx/spark-cli-design.md`. Build the v1
> starter — the Python CLI in the doc — at `~/src/spark-cli/`. Run
> `spark init` and verify `spark status` works against the Spark."

The work is sub-1-hour at v1 scope (CLI is mostly already in the doc;
just paste, `pipx install -e`, run `init`, test).
