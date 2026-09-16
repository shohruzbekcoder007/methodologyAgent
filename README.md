# Hermes host agent — starter

A minimal, production-shaped FastAPI service around a **Hermes host agent**:
conversation, multi-turn session memory, and a tool-calling loop. No domain
logic — add your own tools and build from here.

## Architecture

```
Gateway (Open WebUI, …) → POST /v1/chat
    → Hermes host agent  (system prompt + session history)
        └─ tools from agents/hermes_host.py::_host_langchain_tools()
```

The real Hermes framework is a hard requirement. Its build backend refuses pip
wheel builds by design, so the image clones the source to `/opt/hermes-agent`
and puts it on `PYTHONPATH` rather than installing it. The build then verifies
the import and **fails** if it is not usable — an image never ships silently
without Hermes.

Two backends, chosen automatically at startup:

| Backend | When | What |
|---|---|---|
| `hermes` | normal operation | real Hermes `AIAgent` — plugins, toolsets, its own conversation loop |
| `hermes_lite` | safety net if the above cannot start | LangGraph `create_react_agent` with the same design |

Hermes resolves its own provider credentials from the environment — no
interactive `hermes login` or `hermes setup` step, so deployment stays a
single command. See **LLM provider** below for which variables it reads.

## LLM provider

`LLM_PROVIDER` is the only switch. It picks one block of `.env`; the other
block sits there untouched, so an OpenAI key and a local server can coexist
and neither leaks into the other.

```env
LLM_PROVIDER=openai            # openai | ollama | vllm | lmstudio

# 1: OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1
HERMES_TASK_MODEL=gpt-4.1-mini

# 2: local OpenAI-compatible server (Ollama / vLLM / LM Studio)
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_MODEL=qwen3.8:27b
OLLAMA_TASK_MODEL=qwen3:8b
```

| `LLM_PROVIDER` | Key | Endpoint | Model | Key required |
|---|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | `LLM_MODEL` | yes |
| `ollama` | `OLLAMA_API_KEY` | `OLLAMA_BASE_URL` | `OLLAMA_MODEL` | no |
| `vllm` | `OLLAMA_API_KEY` | `OLLAMA_BASE_URL` | `OLLAMA_MODEL` | no |
| `lmstudio` | `OLLAMA_API_KEY` | `OLLAMA_BASE_URL` | `OLLAMA_MODEL` | no |

The three local providers share one env block on purpose — point
`OLLAMA_BASE_URL` at whichever server is running. They differ only in the
provider profile handed to Hermes, and that profile is not cosmetic: the
`ollama` one sends `think=false`, detects `num_ctx` and lifts the `max_tokens`
floor. Without it Ollama truncates every reply at its internal
`num_predict=128` default.

`HERMES_INFERENCE_PROVIDER` follows `LLM_PROVIDER` automatically; set it only
to override. From Docker, the host machine's Ollama is reachable at
`host.docker.internal` — compose maps it for Linux hosts too.

The model must support tool calling, since the host agent is a tool-calling
loop (`ollama show <model>` lists `tools` under Capabilities).

### Task model

A chat UI runs small jobs behind the scenes — Open WebUI generates chat
titles, tags and follow-up suggestions by sending a prompt marked `### Task:`
through the normal chat route. When `HERMES_TASK_MODEL` (OpenAI) or
`OLLAMA_TASK_MODEL` (local) is set, those go to that model in a single call:
no tools, no history, no memory writes. Everything else reaches the full
agent. If the task model errors, the request falls through to the main agent
rather than failing.

Unset the variable, or set `HERMES_TASK_ROUTING=false`, and the main model
answers them as before.

## Hermes toolsets

`HERMES_ENABLED_TOOLSETS` selects which of Hermes' 59 toolsets the host agent
gets. The default enables persistence and self-improvement, and nothing that
reaches outside the container:

| Toolset | Tools | What it gives the agent |
|---|---|---|
| `memory` | `memory` | durable facts, re-injected into every later turn |
| `session_search` | `session_search` | recall and summarize past conversations |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | write and revise its own skill documents |
| `todo` | `todo` | plan multi-step work |

Everything Hermes learns lives under `HERMES_HOME` — `memories/`, `skills/`,
`sessions/`, `state.db`, `SOUL.md` — kept in the `methodologyagent-hermes-home` named
volume so a redeploy does not wipe it. `config/hermes_config.yaml` is copied in
only when the volume has no `config.yaml` yet; delete the volume to re-seed it.

Toolsets such as `terminal`, `code_execution`, `file` and `browser` let the
agent act inside the container. Set `API_BEARER_TOKEN` before enabling any of
them — `/v1/chat` is unauthenticated while it is unset.

## Layout

```
agents/
  hermes_host.py    host agent: sessions, backends, tool loop
  example_tool.py   template tool (`echo`) — copy this for your own
app/
  api.py            FastAPI routes
  main.py           process entrypoint (uvicorn)
  logging_setup.py  JSON logging
config/
  hermes_config.yaml  Hermes profile (plugins/toolsets off by default)
  logging.yaml
prompts/
  hermes_coordinator.md  host system prompt
scripts/
  start.sh          container entrypoint
  healthcheck.sh
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness — never touches the LLM |
| GET | `/ready` | host readiness; `503` when not ready |
| GET | `/v1/info` | backend, provider, model, task model, registered tools |
| POST | `/v1/chat` | chat; `{"message": "...", "session_id": "...", "reset_session": false}` |
| GET | `/docs` | OpenAPI UI |

Set `API_BEARER_TOKEN` to require `Authorization: Bearer …` on `/v1/chat`.

## Run

Local: see [install_local.md](install_local.md) · Docker: see [install.md](install.md)

```bash
# Docker: host port is HOST_PORT (9095 by default). Running locally: 8080.
curl -s localhost:9095/v1/chat -H 'content-type: application/json' \
  -d '{"message":"salom"}'
```

## Adding a tool

1. Copy `agents/example_tool.py`, rename the function, write a real docstring —
   the LLM reads it to decide when to call the tool.
2. Register it in `agents/hermes_host.py` → `_host_langchain_tools()`.
3. Describe it in `prompts/hermes_coordinator.md` under **Tools**.

## Configuration

All via environment (`.env`, see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `ollama` \| `vllm` \| `lmstudio` |
| `OPENAI_API_KEY` | — | required when `LLM_PROVIDER=openai` |
| `OPENAI_BASE_URL` | OpenAI | any compatible gateway |
| `LLM_MODEL` | `gpt-4.1` | also `HERMES_MODEL`, `OPENAI_MODEL` |
| `HERMES_TASK_MODEL` | — | small model for `### Task:` prompts |
| `OLLAMA_BASE_URL` | `localhost:11434/v1` | local server; used by all three local providers |
| `OLLAMA_MODEL` | `qwen3:8b` | must support tool calling |
| `OLLAMA_TASK_MODEL` | — | small model for `### Task:` prompts |
| `OLLAMA_API_KEY` | — | only behind an authenticating proxy |
| `HERMES_TASK_ROUTING` | `true` | `false` = never route to the task model |
| `HERMES_INFERENCE_PROVIDER` | from `LLM_PROVIDER` | override only |
| `HERMES_SYSTEM_PROMPT_PATH` | `prompts/hermes_coordinator.md` | host prompt |
| `HERMES_ENABLED_TOOLSETS` | `memory,session_search,skills,todo` | comma-separated Hermes toolsets |
| `HERMES_MAX_ITERATIONS` | `12` | tool-loop cap |
| `HERMES_SESSION_HISTORY_LIMIT` | `6` | turns kept per session |
| `HERMES_SKIP_MEMORY` | `false` | `true` = stateless |
| `HERMES_REASONING_ENABLED` | `false` | keep `false` on gpt-4* |
| `API_BEARER_TOKEN` | — | unset = no auth |
| `CORS_ORIGINS` | `*` | comma-separated |

Sessions are in-process and per-worker: `API_WORKERS>1` will split them.
