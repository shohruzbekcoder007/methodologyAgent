# Install — Hermes host agent (Docker)

1. `cp .env.example .env`, then pick a provider with `LLM_PROVIDER`:
   - `openai` — set `OPENAI_API_KEY` (and `LLM_MODEL` if not `gpt-4.1`)
   - `ollama` — set `OLLAMA_MODEL`; the default `OLLAMA_BASE_URL` already
     points at the host machine's Ollama via `host.docker.internal`
2. `docker compose build && docker compose up -d`
3. `curl http://127.0.0.1:9095/ready` (host port — `HOST_PORT`; inside the
   container the service stays on 8080)
4. `POST /v1/chat` with `{"message":"..."}`

Host prompt: `prompts/hermes_coordinator.md`
Tools: `agents/hermes_host.py` → `_host_langchain_tools()`
