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

## Knowledge stores

Two databases ship with the stack, restored from a backup rather than built by
an ingest run:

| Service | Image | Holds | Host port (loopback) |
|---|---|---|---|
| `mongo` | `mongo:7.0` | 45 collections, 251 728 documents, GridFS files | `27118` |
| `neo4j` | `neo4j:5.26-community` | 53 133 nodes, 195 029 relationships | `9096` (browser), `9097` (bolt) |

The versions are pinned to the backups: a Neo4j dump never loads into an older
store, and a Mongo archive tracks the server that wrote it. Neo4j is published
in this stack's own block beside the app's 9095 rather than on the 7474/7687
ladder every other Neo4j container competes for, so a graph container started
later will not collide with it. The app reaches both by service name on the compose network, so
nothing here depends on the published ports — they are for `mongosh`, Neo4j
Browser and other tools you run by hand.

The two stores reference each other: every Neo4j node carries `mongo_id`,
`mongo_db` and `mongo_collection`, and `edges` in Mongo mirrors the graph. They
are backed up as a pair and should be restored as a pair. Field-by-field
reference: [data/MONGODB_SCHEMA.md](data/MONGODB_SCHEMA.md),
[data/NEO4J_SCHEMA.md](data/NEO4J_SCHEMA.md).

### Restore

Backups live in `data/backups/` (git-ignored). The script picks the newest of
each kind, or takes an explicit file:

```bash
./scripts/restore_backups.sh              # newest of each, asks first
./scripts/restore_backups.sh --yes        # no prompt
./scripts/restore_backups.sh --list       # what is available
./scripts/restore_backups.sh --only mongo
./scripts/restore_backups.sh --neo4j-file nmc-neo4j-5.26-20260915-2111.dump
```

Run it as often as you like: Mongo restores with `--drop` and Neo4j with
`--overwrite-destination`, so the second run ends where the first did. Both
stores lose whatever they held, which is why it asks before starting and
refuses outright when it is not attached to a terminal and `--yes` is absent.

Mongo restores while it serves. Neo4j Community loads a dump only into a
stopped store, so the script stops the service, runs `neo4j-admin database
load` in a throwaway container attached to the `methodologyagent-neo4j-data`
volume, and starts it again — a few minutes of downtime for the graph.

Afterwards it prints what landed, which is the check worth reading:

```
[restore] mongo: 45 collections, 251728 documents
[restore] neo4j: 53133 nodes
```

Data survives `docker compose down`; only `down -v` clears the volumes
(`methodologyagent-mongo-data`, `methodologyagent-mongo-config`,
`methodologyagent-neo4j-data`, `methodologyagent-neo4j-logs`) — and that is one
re-run away from being back.

### Semantic search

Retrieval is hybrid: BM25 over Neo4j's `search_text` finds an exact wording,
vector search over `embedding` finds a passage that answers the question
without repeating its words, and the two rankings are fused with RRF. On a
worded-around question the vector half contributes almost all the recall — in
a check on the restored corpus, 17 of its top 20 passages were ones BM25 never
returned.

`scripts/embed_chunks.sh` builds the vector side:

```bash
./scripts/embed_chunks.sh                    # everything still pending
./scripts/embed_chunks.sh --status           # what is embedded, index state
./scripts/embed_chunks.sh --only terms
./scripts/embed_chunks.sh --limit 200        # a slice, to try it
./scripts/embed_chunks.sh --force            # rebuild every vector
./scripts/embed_chunks.sh --check "narx o'zgarishi qanday o'lchanadi"
```

| What | Count | Embedded from |
|---|---|---|
| `Chunk` | 26 342 | `chunks.text` in MongoDB, section heading prepended |
| `Term` | 1 308 | name + definition |

The text comes from MongoDB, not from Neo4j, and that is deliberate. Neo4j's
`search_text` is a complete copy of the corpus (29.3M characters against 28.8M
of original) but it is transliterated to Latin, and transliteration flattens
what a multilingual model reads best — abbreviations and proper names above
all. The vectors are written back onto the Neo4j nodes, so one Cypher query
can search and walk the graph together.

The embedding service is external: this stack does not run it, it points at
one. `EMBEDDING_BASE_URL` defaults to `host.docker.internal:8090`, and the
script refuses to start if nothing answers there. Whatever the service reports
as its `index_key` (provider, model and dimensions) is stored beside every
vector, which is what makes the run resumable and a model change safe:

* no vector, or one from another model → pending;
* interrupted run → the next one continues where it stopped;
* different model → everything is rebuilt, because vectors from two models are
  not comparable and mixing them would spoil every later search with no error
  anywhere to explain it.

With BAAI/bge-m3 on a GPU the full corpus takes about 7 minutes end to end
(~60 chunks/s including the MongoDB reads and Neo4j writes).

### Documents with no text

361 of the 1 185 documents converted to nothing, so they carry no chunk and no
search can reach their contents. Every one of them has a `READABLE_SIBLING`
edge to a copy that did convert, and `agents/knowledge/retrieval.py` follows
it: the document card comes back with `has_text: false`, the sibling under
`read_instead`, and a note saying the text is missing and a readable copy is
being shown instead. The sibling's text is never passed off as the original's.

Their real content is not in the databases at all — `l1_uri` names the source
PDF/DOCX, but that drive was not mounted during the ingest and the GridFS `l1`
bucket was never created.

### Taking a backup

The dump side is not scripted, because it belongs to whichever stack owns the
data. Against this one:

```bash
# Mongo — online
MSYS_NO_PATHCONV=1 docker exec methodologyagent-mongo mongodump   --username nmc --password '<MONGO_ROOT_PASSWORD>' --authenticationDatabase admin   --db nmc --gzip --archive=/tmp/nmc.archive.gz
docker cp methodologyagent-mongo:/tmp/nmc.archive.gz   "data/backups/nmc-mongo-7.0-$(date +%Y%m%d-%H%M).archive.gz"

# Neo4j — offline
docker compose stop neo4j
MSYS_NO_PATHCONV=1 docker run --rm   -v methodologyagent-neo4j-data:/data -v "$(pwd)/data/backups:/backups"   neo4j:5.26-community neo4j-admin database dump neo4j --to-path=/backups
docker compose start neo4j
```

## Layout

```
agents/
  hermes_host.py    host agent: sessions, backends, tool loop
  user_profiles.py  one Hermes home per user; the slug is the boundary
  example_tool.py   template tool (`echo`) — copy this for your own
  knowledge/
    store.py        Neo4j, MongoDB and embedding connections (singletons)
    retrieval.py    hybrid search, graph expansion, readable-sibling redirect
    text.py         Cyrillic → Latin, matching what the ingest indexed
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
  start.sh             container entrypoint
  healthcheck.sh
  restore_backups.sh   restore Mongo + Neo4j from data/backups/
  embed_chunks.sh      build the vector side of search (driver)
  embed_chunks.py      …its worker, run inside the app container
data/
  backups/             *.archive.gz, *.dump — git-ignored
  MONGODB_SCHEMA.md    every collection, field and index
  NEO4J_SCHEMA.md      every label, property, relationship and index
```

## Scripts

Both are re-runnable and both refuse to guess: they check what they are about
to touch, say so, and stop rather than half-finish.

| Script | What it is for |
|---|---|
| [`scripts/restore_backups.sh`](scripts/restore_backups.sh) | Loads MongoDB and Neo4j from `data/backups/`. Picks the newest archive and dump, or takes named ones. Mongo restores while it serves; Neo4j Community loads a dump only into a stopped store, so the script stops the service, loads in a throwaway container against the data volume, and starts it again. Prints what landed. |
| [`scripts/embed_chunks.sh`](scripts/embed_chunks.sh) | Embeds the corpus and writes the vectors onto the Neo4j nodes, then creates the vector indexes. Resumable: only nodes without a current vector are touched. `--status` shows what is embedded, `--check` runs a real search through the retrieval layer. |
| [`scripts/start.sh`](scripts/start.sh) | Container entrypoint. Fixes volume ownership, seeds `config.yaml` and `SOUL.md` into the shared home, drops from root to `appuser`, starts the server. Not run by hand. |
| [`scripts/healthcheck.sh`](scripts/healthcheck.sh) | What Docker's `HEALTHCHECK` calls. Not run by hand. |

Order matters once, on a fresh machine: restore first, embed second. Embedding
an empty database succeeds and does nothing.

```bash
./scripts/restore_backups.sh --yes
./scripts/embed_chunks.sh
./scripts/embed_chunks.sh --status
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
| `MONGO_ROOT_USER` | `nmc` | Mongo root user |
| `MONGO_ROOT_PASSWORD` | — | required; compose refuses to start without it |
| `MONGO_DB` | `nmc` | database the backup restores into |
| `MONGO_PORT` | `27118` | loopback host port |
| `NEO4J_USER` | `neo4j` | Community Edition has no other |
| `NEO4J_PASSWORD` | — | required; compose refuses to start without it |
| `NEO4J_DATABASE` | `neo4j` | Community serves one user database |
| `NEO4J_HTTP_PORT` | `9096` | Neo4j Browser |
| `NEO4J_BOLT_PORT` | `9097` | driver connections from the host |
| `NEO4J_HEAP` / `NEO4J_PAGECACHE` | `1G` | raise both for an ingest run |
| `BIND_ADDRESS` | `127.0.0.1` | interface every published port binds to |
| `EMBEDDING_BASE_URL` | `host.docker.internal:8090` | external embedding service |
| `EMBEDDING_BATCH` | `64` | must not exceed the service's `batch_max` |
| `EMBEDDING_TIMEOUT` | `120` | seconds per request |

Sessions are in-process and per-worker: `API_WORKERS>1` will split them.
