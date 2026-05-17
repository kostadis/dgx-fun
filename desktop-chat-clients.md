# Chatting with vLLM from a Windows Desktop

Three solid options for talking to the Spark's `vllm-chat` container
(`http://192.168.1.147:8001/v1`, model `Qwen/Qwen2.5-14B-Instruct-AWQ`)
from a Windows desktop. Ranked by how fast you can be sending your first
message.

## 1. Jan (jan.ai) — recommended for fastest setup

Native Windows installer, free, open source. ChatGPT-like UI, conversation
history stored locally.

Get the installer from `https://jan.ai`. After install:

- Settings → Model Providers → **+ Add Provider**
- Provider type: **OpenAI Compatible**
- API URL: `http://192.168.1.147:8001/v1`
- API Key: anything non-empty (vLLM ignores it but the field is required)
- Click **Refresh Models** — should pick up `Qwen/Qwen2.5-14B-Instruct-AWQ`

Pick the model in the chat dropdown and you're chatting. ~5 minutes from
download to first message.

## 2. Cherry Studio — more polish, more knobs

`https://cherry-ai.com` — native Windows, free, prettier UI than Jan,
supports many providers in one app.

Setup pattern: **Settings → Model Provider → custom OpenAI endpoint**.
Same fields as Jan. Slightly more clicks but better if you'll add more
endpoints later (Claude API, OpenRouter, a local Ollama, etc.).

## 3. Open WebUI in Docker — full ChatGPT clone

If you have Docker Desktop on Windows installed:

```powershell
docker run -d -p 3000:8080 `
  -e OPENAI_API_BASE_URL=http://192.168.1.147:8001/v1 `
  -e OPENAI_API_KEY=anything `
  -v open-webui:/app/backend/data `
  --name open-webui `
  --restart unless-stopped `
  ghcr.io/open-webui/open-webui:main
```

Browse to `http://localhost:3000`. Heaviest setup, most featureful:
prompt library, RAG against uploaded files, multi-user, full ChatGPT
clone. Overkill for "just chat" but excellent if you'll live in it.

---

## Picking among them

- **Just want to chat, fastest**: Jan.
- **Will use multiple model providers (Claude, OpenAI, local) from one app**: Cherry Studio.
- **Want a permanent ChatGPT replacement, fine with running Docker**: Open WebUI.

All three are free and actively maintained. Switching between them later
is cheap — they each store conversation history in their own local format
but the model endpoint config moves over in a minute.

## Notes specific to this Spark setup

- Endpoint is `http://192.168.1.147:8001/v1` (port 8001 = vllm-chat;
  port 8000 is the embedding container, not for chat).
- Model name must be the literal HuggingFace ID:
  `Qwen/Qwen2.5-14B-Instruct-AWQ`. Not `qwen2.5:14b` (that's the
  Ollama-style name).
- Single-sequence decode is ~15 tok/s — feels slow compared to
  cloud Claude/GPT but is the bandwidth ceiling for a 14B model on
  this hardware. (See `spark-llm-serving-learnings.md` for the math.)
- Conversation context is bounded by `--max-model-len 32768` on the
  chat container — comfortably long for normal use.
