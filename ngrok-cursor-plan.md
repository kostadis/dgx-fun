# Plan: expose `vllm-chat` to Cursor via ngrok

**Status:** not started. To be done next week.
**Target host:** `spark` (= spark1, `192.168.1.147`, hostname `gx10-46ea`).
**Goal:** point Cursor's "OpenAI API" model integration at the
`vllm-chat` container on the Spark, over a public ngrok tunnel.

## Why ngrok at all

Cursor's "Custom OpenAI endpoint" integration calls the URL from
Cursor's backend, not from the local app. A LAN IP like
`192.168.1.147:8001` won't work — the endpoint must be reachable from
the public internet. ngrok is the cheapest way to get there without
poking holes in the home router.

Tradeoff being accepted:

- Adds a hop (laptop → Cursor backend → ngrok → spark) so latency is
  worse than local.
- Free-tier ngrok URLs rotate on every restart of the tunnel.
- Free-tier rate limits exist but are generous enough for one-user
  coding flows.

This is exploration-grade infrastructure, not production. See the
"Local AI Hardware Exploration" note in `~/.claude/CLAUDE.md`.

## Preconditions to confirm before starting

Run these and confirm reality matches expectations. If anything
disagrees, fix it before continuing.

```bash
# vllm-chat is up and serving on 8001
curl -sS http://192.168.1.147:8001/v1/models | jq

# What model is actually being served — plug this exact id into Cursor
ssh spark 'docker inspect vllm-chat --format "{{.Args}}"'
```

As of 2026-05-22 the live model is
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` and there is **no `--api-key`
flag** on the container, so any string works as the Cursor API key.
`current-setup.md` already reflects this correctly (snapshot dated
2026-05-21). If anything has changed by the time this is executed,
update step 4 below and reconcile `current-setup.md` in the same
change per `CLAUDE.md`.

## Prerequisite from the user

Before starting the work, have an **ngrok authtoken** in hand:

1. Go to https://dashboard.ngrok.com/.
2. Sign up / log in.
3. Copy "Your Authtoken" from the left sidebar.

Free tier is fine for this. Paid tier buys a reserved domain (stable
URL across restarts) — worth it only if Cursor becomes a daily
driver.

## Step 1 — install ngrok on spark

```bash
ssh spark '
  curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
  echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
    | sudo tee /etc/apt/sources.list.d/ngrok.list
  sudo apt update
  sudo apt install -y ngrok
  ngrok version
'
```

Verify: `ngrok version` prints something like `ngrok version 3.x.y`.

## Step 2 — configure the authtoken

```bash
ssh spark 'ngrok config add-authtoken <PASTE_TOKEN>'
```

This writes `~/.config/ngrok/ngrok.yml` on the Spark. One-time setup;
survives reboots.

## Step 3 — start the tunnel

Decide between two flavors:

### 3a. Quick / no auth (fine for a 10-minute trial)

```bash
ssh spark 'nohup ngrok http 8001 --log=stdout > ~/ngrok.log 2>&1 &'
sleep 2
ssh spark 'curl -s http://127.0.0.1:4040/api/tunnels | jq -r ".tunnels[0].public_url"'
```

The second command prints the public URL,
e.g. `https://ab12-203-0-113-4.ngrok-free.app`.

### 3b. With basic auth (recommended even for a free-tier tunnel)

```bash
ssh spark 'nohup ngrok http 8001 --basic-auth="cursor:<LONG_PASSWORD>" \
  --log=stdout > ~/ngrok.log 2>&1 &'
```

In Cursor's "OpenAI API Key" field, supply the password directly —
Cursor sends it as a Bearer token, not as HTTP Basic, so basic auth
on the tunnel won't actually let Cursor through. **Skip 3b for
Cursor specifically.** Use 3a + the Step-5 hardening path below.

### 3c. Run ngrok as a systemd service (do this once it's working)

Once you've confirmed the flow end-to-end with 3a, promote ngrok to
a systemd unit so it survives reboots:

```ini
# /etc/systemd/system/ngrok-vllm.service
[Unit]
Description=ngrok tunnel for vllm-chat (port 8001)
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=simple
User=kostadis
ExecStart=/usr/local/bin/ngrok http 8001 --log=stdout
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
ssh spark '
  sudo systemctl daemon-reload
  sudo systemctl enable --now ngrok-vllm
  sudo systemctl status ngrok-vllm --no-pager
'
```

Note: every restart of the ngrok process rotates the free-tier URL.
Cursor will need to be reconfigured each time. If that becomes
annoying, buy ngrok's $10/mo plan for a reserved domain.

## Step 4 — wire Cursor

In Cursor → Settings → Models → "OpenAI API" section:

- **Override OpenAI Base URL:** `https://<NGROK_HOST>/v1`
  - Yes, include the trailing `/v1`. vLLM serves OpenAI-compatible
    routes under `/v1/...`.
- **OpenAI API Key:** any non-empty string (e.g. `sk-local`) — vLLM
  is not checking, so this is just to satisfy Cursor's UI.
- **Model name:** add a custom model with id
  `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (or whatever
  `/v1/models` returned in the preconditions). The id must match
  exactly — vLLM is strict about this.
- Click **Verify** in Cursor. It sends a tiny chat completion. If
  this fails, jump to "Troubleshooting" below.

## Step 5 — add real auth to vLLM (recommended)

Free-tier ngrok URLs are unauthenticated and the model is exposed to
anyone who guesses the URL. Before leaving this running for long,
restart `vllm-chat` with an API key:

1. Pick a token, e.g. `sk-spark-$(openssl rand -hex 16)`.
2. Edit the active spin-up script
   (`spin-up-vllm-qwen3-next-80b.sh` per the working-tree state) to
   add `--api-key "$VLLM_API_KEY"` to the `docker run` command.
3. Export `VLLM_API_KEY` before running the script, or hard-code it
   in the script (less ideal — it ends up in git).
4. Re-spin the container.
5. **Update `current-setup.md`** in the same change — §3
   "vllm-chat", §"Client-side configuration", and bump the snapshot
   date. (This is the hard rule in `CLAUDE.md`.)
6. Put the same `VLLM_API_KEY` value into Cursor's "OpenAI API Key"
   field, replacing the dummy `sk-local`.

After this, the only attack surface left is "someone guesses the
ngrok URL **and** the API key" — acceptable for an exploration box.

## Step 6 — sanity check from the laptop

Independent of Cursor, prove the tunnel works end-to-end:

```bash
NGROK_URL="https://<paste-host>"   # e.g. https://ab12-...ngrok-free.app
curl -sS "$NGROK_URL/v1/models" | jq
curl -sS "$NGROK_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    "messages": [{"role":"user","content":"say hi in 3 words"}],
    "max_tokens": 16
  }' | jq
```

If both succeed, Cursor will succeed too. If `/v1/models` works but
`/v1/chat/completions` 404s, you forgot the `/v1` prefix in Cursor.

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Cursor "Verify" hangs ~30s then errors | model id mismatch | Copy id from `/v1/models` verbatim into Cursor |
| Cursor returns 401 from vLLM | Step 5 done with key; Cursor still has `sk-local` | Update Cursor API key field |
| ngrok URL works once then 502 | vllm-chat OOM'd or crashed | `ssh spark 'docker logs --tail=200 vllm-chat'` |
| ngrok URL went dead overnight | Free-tier tunnel rotated on restart | Re-grab URL via `curl 127.0.0.1:4040/api/tunnels`; reconfigure Cursor |
| Cursor sends tool calls and vLLM 400s | Tool-call parser flag missing | Already passing `--enable-auto-tool-choice --tool-call-parser hermes` — should be fine. If not, check container args |
| Generation is very slow (>5s first token) | Cursor sends huge context; Qwen3-Next prefill dominates | Expected on Spark; this is the read-heavy / prefill-bound tradeoff |

## What to do after it's working

- Decide whether Cursor is actually pleasant over this hop. If yes,
  pay for the ngrok reserved domain and put it in `current-setup.md`
  as a documented service. If no, kill the tunnel and call the
  experiment done.
- Either way, **update `current-setup.md`** with a new §"ngrok
  tunnel" subsection or remove this plan doc, so the working tree
  reflects the decided state.
- Consider whether the same pattern is worth doing for
  `vllm-embed` (port 8000). Probably not — embeddings are cheap to
  re-host elsewhere and don't benefit from Cursor specifically.

## Out of scope for this plan

- Putting a real reverse proxy (Caddy, Cloudflare Tunnel) in front
  instead of ngrok. Better long-term, more setup. Defer until the
  ngrok version has been used enough to justify the upgrade.
- Exposing the Spark via Tailscale instead. Different model — works
  great for laptop ↔ spark, but Cursor's backend can't reach a
  Tailscale-only host, so it doesn't solve the original problem.
- Hardening the Spark beyond the vLLM API key (firewall rules,
  fail2ban on ngrok hits, etc.). Overkill for the threat model here.
