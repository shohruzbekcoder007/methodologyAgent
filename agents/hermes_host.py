"""
Hermes host agent — starter.

Architecture:

  User → Hermes host (conversation + memory)
            └─ tools registered in `_host_langchain_tools()`

If the Hermes package is unavailable, a Hermes-lite outer agent is used
(same design: host tool-calling loop + tools + session history).
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes_host")

_lock = threading.RLock()
_service: Optional["HermesHostService"] = None


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_coordinator_prompt() -> str:
    path = Path(
        _env("HERMES_SYSTEM_PROMPT_PATH")
        or str(
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "hermes_coordinator.md"
        )
    )
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return (
        "You are a helpful host agent. Use the tools available to you "
        "when they apply. Never invent facts."
    )


# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
# `LLM_PROVIDER` is the only switch. It picks one row of this table, and the
# row says which env vars carry the key, endpoint and models -- so an OpenAI
# block and a local block can sit side by side in .env and neither leaks into
# the other. Add a provider by adding a row.
#
# `hermes_provider` is the profile name handed to Hermes. It matters: the
# `ollama` profile sends think=false, detects num_ctx and lifts the max_tokens
# floor. Without it Ollama truncates replies at its num_predict=128 default.
_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "aliases": ("oai",),
        "key_vars": ("OPENAI_API_KEY", "LLM_API_KEY", "HERMES_API_KEY"),
        "url_vars": ("OPENAI_BASE_URL", "HERMES_BASE_URL"),
        "model_vars": ("HERMES_MODEL", "LLM_MODEL", "OPENAI_MODEL"),
        "task_model_vars": ("HERMES_TASK_MODEL", "OPENAI_TASK_MODEL"),
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4.1",
        "key_required": True,
        # Hermes infers the OpenAI profile from the base URL on its own.
        "hermes_provider": None,
    },
    # Local OpenAI-compatible servers. They share one env block -- point
    # OLLAMA_BASE_URL at whichever server is running -- but keep separate
    # Hermes profiles, because the wire quirks differ.
    "ollama": {
        "aliases": ("local",),
        "key_vars": ("OLLAMA_API_KEY",),
        "url_vars": ("OLLAMA_BASE_URL",),
        # Deliberately not LLM_MODEL / HERMES_MODEL: an OpenAI model name left
        # behind in the environment must not leak onto a local endpoint.
        "model_vars": ("OLLAMA_MODEL",),
        "task_model_vars": ("OLLAMA_TASK_MODEL",),
        "default_base_url": "http://localhost:11434/v1",
        "default_model": "qwen3:8b",
        "key_required": False,
        "hermes_provider": "ollama",
    },
    "vllm": {
        "aliases": (),
        "key_vars": ("OLLAMA_API_KEY",),
        "url_vars": ("OLLAMA_BASE_URL",),
        "model_vars": ("OLLAMA_MODEL",),
        "task_model_vars": ("OLLAMA_TASK_MODEL",),
        "default_base_url": "http://localhost:8000/v1",
        "default_model": "",
        "key_required": False,
        "hermes_provider": "vllm",
    },
    "lmstudio": {
        "aliases": ("lm-studio", "lm_studio"),
        "key_vars": ("OLLAMA_API_KEY",),
        "url_vars": ("OLLAMA_BASE_URL",),
        "model_vars": ("OLLAMA_MODEL",),
        "task_model_vars": ("OLLAMA_TASK_MODEL",),
        "default_base_url": "http://localhost:1234/v1",
        "default_model": "",
        "key_required": False,
        "hermes_provider": "lmstudio",
    },
}

# Local servers accept any key, but the OpenAI SDK still demands a non-empty
# string. This is the placeholder Hermes itself substitutes.
_PLACEHOLDER_KEY = "no-key-required"

# A chat UI runs one-shot jobs behind the scenes -- Open WebUI generates chat
# titles, tags and follow-up suggestions by sending a prompt carrying this
# marker through the normal chat route. They need no tools, no history and no
# memory, so they go to the small task model instead of the main agent.
_TASK_PROMPT_MARKER = "### task:"


def _first_env(names: tuple[str, ...]) -> str:
    """First non-empty value among `names`."""
    for name in names:
        value = _env(name)
        if value:
            return value
    return ""


def _resolve_provider() -> tuple[str, Optional[str]]:
    """
    Resolve `LLM_PROVIDER` to a key of `_PROVIDERS`.

    An unknown name falls back to openai and returns an error string rather
    than raising, so a typo surfaces as a clear `/ready` message instead of a
    crash loop.
    """
    requested = (_env("LLM_PROVIDER") or "openai").lower()
    for canonical, spec in _PROVIDERS.items():
        if requested == canonical or requested in spec["aliases"]:
            return canonical, None
    return (
        "openai",
        f"Unknown LLM_PROVIDER={requested!r}; supported: "
        + ", ".join(sorted(_PROVIDERS)),
    )


def _is_task_prompt(message: str) -> bool:
    """True for a chat UI's background title/tag/follow-up prompt."""
    return _TASK_PROMPT_MARKER in (message or "")[:400].lower()


class HermesHostService:
    """
    Host agent with session memory and a tool-calling loop.
    """

    name = "hermes_host"

    def __init__(self) -> None:
        self.provider, self._provider_error = _resolve_provider()
        spec = _PROVIDERS[self.provider]
        self._key_var = spec["key_vars"][0]
        self._model_var = spec["model_vars"][-1]
        self._key_required = bool(spec["key_required"])
        self._hermes_provider = spec["hermes_provider"]

        self.model_name = _first_env(spec["model_vars"]) or spec["default_model"]
        self.base_url = _first_env(spec["url_vars"]) or spec["default_base_url"] or None
        self.api_key = _first_env(spec["key_vars"]) or (
            "" if self._key_required else _PLACEHOLDER_KEY
        )
        # Optional cheap model for a chat UI's background jobs. Unset = the
        # main agent answers them, exactly as before.
        self.task_model = _first_env(spec["task_model_vars"])
        self.task_routing = _env_bool("HERMES_TASK_ROUTING", True)
        self._task_llm_obj: Any = None

        # Hermes' own internals (memory provider, sub-agents) read this env var
        # directly; keep it in step with LLM_PROVIDER unless the operator set
        # it explicitly.
        if not _env("HERMES_INFERENCE_PROVIDER"):
            os.environ["HERMES_INFERENCE_PROVIDER"] = (
                self._hermes_provider or self.provider
            )

        self.max_iterations = _env_int("HERMES_MAX_ITERATIONS", 12)
        self.session_limit = _env_int("HERMES_SESSION_HISTORY_LIMIT", 6)
        self.skip_memory = _env_bool("HERMES_SKIP_MEMORY", False)
        self.system_prompt = _load_coordinator_prompt()
        self._backend: str | None = None  # hermes | hermes_lite
        self._ready = False
        self._last_error: str | None = None
        self._sessions: dict[str, list[Any]] = {}
        # hermes_lite: langgraph graph; hermes: factory for AIAgent
        self._lite_graph: Any = None
        # Whether that graph already carries the system prompt (prompt=) or
        # expects it in the message list -- the two shapes below differ.
        self._lite_has_prompt = False
        self._hermes_ok = False

    def initialize(self) -> dict[str, Any]:
        with _lock:
            if self._ready:
                return self.readiness()

            if self._provider_error:
                self._last_error = self._provider_error
                self._ready = False
                return self.readiness()

            if self._key_required and not self.api_key:
                self._last_error = (
                    f"{self._key_var} not set (LLM_PROVIDER={self.provider})"
                )
                self._ready = False
                return self.readiness()

            if not self.model_name:
                self._last_error = (
                    f"{self._model_var} not set (LLM_PROVIDER={self.provider})"
                )
                self._ready = False
                return self.readiness()

            try:
                from hermes_cli.plugins import discover_plugins  # type: ignore

                discover_plugins()
            except Exception as exc:  # noqa: BLE001
                logger.debug("discover_plugins: %s", exc)

            # Prefer real Hermes AIAgent
            if self._try_init_hermes():
                self._backend = "hermes"
                self._ready = True
                self._last_error = None
                logger.info(
                    "Hermes host ready (backend=hermes provider=%s model=%s "
                    "task_model=%s base_url=%s)",
                    self.provider,
                    self.model_name,
                    self.task_model or "-",
                    self.base_url,
                )
                return self.readiness()

            # Hermes-lite: same architecture without hermes package
            if self._try_init_hermes_lite():
                self._backend = "hermes_lite"
                self._ready = True
                self._last_error = None
                logger.info(
                    "Hermes host ready (backend=hermes_lite provider=%s model=%s "
                    "task_model=%s base_url=%s) — Hermes package missing or "
                    "failed; using tool-calling host",
                    self.provider,
                    self.model_name,
                    self.task_model or "-",
                    self.base_url,
                )
                return self.readiness()

            self._ready = False
            self._last_error = self._last_error or "Failed to init Hermes host"
            return self.readiness()

    def _try_init_hermes(self) -> bool:
        try:
            from run_agent import AIAgent  # type: ignore[import-not-found]

            # Smoke-construct (don't store shared AIAgent — not always thread-safe)
            kwargs = self._hermes_kwargs()
            _ = AIAgent(**kwargs)
            self._hermes_ok = True
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hermes AIAgent unavailable: %s", exc)
            self._hermes_ok = False
            self._last_error = f"Hermes unavailable: {exc}"
            return False

    def _reasoning_config(self) -> dict[str, Any]:
        """
        Hermes defaults to reasoning.effort which OpenAI gpt-4* / gpt-4.1 reject
        with HTTP 400. Keep off unless explicitly enabled (o-series etc.).
        """
        if _env_bool("HERMES_REASONING_ENABLED", False):
            effort = _env("HERMES_REASONING_EFFORT") or "medium"
            return {"enabled": True, "effort": effort}
        return {"enabled": False}

    def _hermes_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "quiet_mode": _env_bool("HERMES_QUIET_MODE", True),
            "max_iterations": self.max_iterations,
            "enabled_toolsets": self._enabled_toolsets(),
            "skip_memory": self.skip_memory,
            "skip_context_files": _env_bool("HERMES_SKIP_CONTEXT_FILES", True),
            "ephemeral_system_prompt": self.system_prompt,
            "platform": "hermes-host",
            "api_key": self.api_key,
            "reasoning_config": self._reasoning_config(),
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        # Set only where the profile matters -- on the OpenAI path Hermes
        # infers it from the base URL, but for a local server the string is
        # what selects the right wire quirks.
        if self._hermes_provider:
            kwargs["provider"] = self._hermes_provider
        return kwargs

    def _enabled_toolsets(self) -> list[str]:
        """Parse HERMES_ENABLED_TOOLSETS (empty by default in the starter)."""
        raw = _env("HERMES_ENABLED_TOOLSETS")
        return [t.strip() for t in raw.split(",") if t.strip()]

    def _host_langchain_tools(self) -> list[Any]:
        """
        Tools exposed to the host agent.

        Add your own here — see `agents/example_tool.py` for the shape.
        """
        tools: list[Any] = []
        try:
            from agents.example_tool import as_langchain_tools

            tools.extend(as_langchain_tools())
        except Exception as exc:  # noqa: BLE001
            logger.debug("example tools not loaded: %s", exc)
        return tools

    def _try_init_hermes_lite(self) -> bool:
        try:
            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "api_key": self.api_key,
                "temperature": 0,
            }
            if self.base_url:
                llm_kwargs["base_url"] = self.base_url
            llm = ChatOpenAI(**llm_kwargs)
            tools = self._host_langchain_tools()
            self._lite_graph = create_react_agent(
                llm,
                tools,
                prompt=self.system_prompt,
            )
            self._lite_has_prompt = True
            return True
        except TypeError:
            # older create_react_agent without prompt=
            try:
                from langchain_openai import ChatOpenAI
                from langgraph.prebuilt import create_react_agent

                llm_kwargs = {
                    "model": self.model_name,
                    "api_key": self.api_key,
                    "temperature": 0,
                }
                if self.base_url:
                    llm_kwargs["base_url"] = self.base_url
                llm = ChatOpenAI(**llm_kwargs)
                self._lite_graph = create_react_agent(
                    llm, self._host_langchain_tools()
                )
                self._lite_has_prompt = False
                return True
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"hermes_lite init failed: {exc}"
                logger.error("%s", self._last_error)
                return False
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"hermes_lite init failed: {exc}"
            logger.error("%s", self._last_error)
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    def readiness(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "backend": self._backend,
            "host": "hermes_host",
            "architecture": "Hermes host (memory/context) → tools",
            "model": self.model_name,
            "provider": self.provider,
            "base_url": self.base_url,
            "task_model": self.task_model or None,
            "skip_memory": self.skip_memory,
            "session_count": len(self._sessions),
            "error": self._last_error,
            "tools": [getattr(t, "name", str(t)) for t in self._host_langchain_tools()],
            "toolsets": self._enabled_toolsets(),
        }

    def clear_session(self, session_id: str) -> None:
        with _lock:
            self._sessions.pop(session_id, None)

    def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        reset_session: bool = False,
    ) -> dict[str, Any]:
        """
        Host chat with optional multi-turn session.
        Never raises to callers.
        """
        try:
            message = (message or "").strip()
            if not message:
                return {
                    "success": False,
                    "response": None,
                    "error": "message must not be empty",
                    "error_code": "validation",
                    "session_id": session_id,
                }

            if not self._ready:
                self.initialize()
            if not self._ready:
                return {
                    "success": False,
                    "response": None,
                    "error": self._last_error or "Hermes host not ready",
                    "error_code": "not_ready",
                    "session_id": session_id,
                }

            client_sid = (session_id or "").strip() or None
            sid = client_sid or str(uuid.uuid4())
            if reset_session:
                self.clear_session(sid)

            prior_len = 0
            with _lock:
                prior_len = len(self._sessions.get(sid, []))
            logger.info(
                "host.chat client_session_id=%r effective_sid=%s prior_history=%d backend=%s",
                client_sid,
                sid,
                prior_len,
                self._backend,
            )

            # A chat UI's background title/tag/follow-up job: answer it on the
            # small model. Falls through to the main agent if it fails.
            if self.task_routing and self.task_model and _is_task_prompt(message):
                task_result = self._chat_task(message, sid)
                if task_result is not None:
                    return task_result

            if self._backend == "hermes":
                return self._chat_hermes(message, sid)
            return self._chat_hermes_lite(message, sid)
        except Exception as exc:  # noqa: BLE001
            logger.error("hermes_host.chat failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "response": None,
                "error": f"Host agent error: {exc}",
                "error_code": "agent_error",
                "retryable": True,
                "session_id": session_id,
            }

    def _task_llm(self) -> Any:
        """Lazily built client for the task model — same endpoint, small model."""
        if self._task_llm_obj is None:
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": self.task_model,
                "api_key": self.api_key,
                "temperature": 0,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._task_llm_obj = ChatOpenAI(**kwargs)
        return self._task_llm_obj

    def _chat_task(self, message: str, sid: str) -> dict[str, Any] | None:
        """
        One-shot completion on the task model: no tools, no history, no memory.

        Returns None when the call fails, so the caller can fall through to the
        main agent rather than surface an error for a background job.
        """
        from langchain_core.messages import HumanMessage

        try:
            result = self._task_llm().invoke([HumanMessage(content=message)])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "task model %s failed, falling back to the main agent: %s",
                self.task_model,
                exc,
            )
            return None

        content = getattr(result, "content", "")
        if isinstance(content, list):
            final = " ".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in content
            )
        else:
            final = str(content or "")

        if not final.strip():
            logger.warning("task model %s returned nothing", self.task_model)
            return None

        logger.info("task prompt answered by task_model=%s", self.task_model)
        return {
            "success": True,
            "response": final,
            "error": None,
            "session_id": sid,
            "backend": self._backend,
            "tool_call_count": 0,
            "agents_used": ["task_model"],
            "mode": "task",
        }

    def _chat_hermes(self, message: str, sid: str) -> dict[str, Any]:
        from run_agent import AIAgent  # type: ignore[import-not-found]

        history: list[Any] | None = None
        with _lock:
            if sid in self._sessions:
                history = list(self._sessions[sid])

        agent = AIAgent(**self._hermes_kwargs())
        try:
            if history:
                result = agent.run_conversation(
                    user_message=message,
                    conversation_history=history,
                    system_message=self.system_prompt,
                )
            else:
                result = agent.run_conversation(
                    user_message=message,
                    system_message=self.system_prompt,
                )
        except TypeError:
            # older signature
            result = agent.run_conversation(user_message=message)

        failed = False
        err: str | None = None
        if isinstance(result, dict):
            final = result.get("final_response") or result.get("response")
            messages = result.get("messages")
            # Hermes reports failures in-band: `error` carries the summary and
            # `final_response` repeats it as prose. Without this check an auth
            # or billing failure reaches callers as a successful answer.
            err = result.get("error") or None
            failed = bool(result.get("failed")) or bool(err)
        else:
            final = str(result)
            messages = None

        if messages is not None and not self.skip_memory and not failed:
            trimmed = list(messages)
            limit = max(4, self.session_limit * 2)
            if len(trimmed) > limit:
                trimmed = trimmed[-limit:]
            with _lock:
                self._sessions[sid] = trimmed

        if failed:
            return {
                "success": False,
                "response": None,
                "error": err or "Hermes conversation failed",
                "error_code": str(result.get("failure_reason") or "agent_error"),
                "error_detail": str(final)[:500] if final else None,
                "retryable": bool(result.get("retryable")),
                "session_id": sid,
                "backend": "hermes",
                "agents_used": ["hermes_host"],
                "mode": "hermes_tool_host",
            }

        return {
            "success": bool((final or "").strip()),
            "response": final or None,
            "error": None if (final or "").strip() else "Empty host response",
            "session_id": sid,
            "backend": "hermes",
            "agents_used": ["hermes_host"],
            "mode": "hermes_tool_host",
        }

    def _chat_hermes_lite(self, message: str, sid: str) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        with _lock:
            prior = list(self._sessions.get(sid, []))

        # Build message list: system (only when the graph lacks it) +
        # history + user. `create_react_agent(prompt=...)` prepends its own
        # SystemMessage, so adding a second one leaves a system message at
        # index 1 -- which vLLM rejects outright with "System message must
        # be at the beginning." OpenAI tolerates the duplicate, which is why
        # this stayed invisible until a local server was pointed at.
        messages: list[Any] = []
        if not self._lite_has_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.extend(prior)
        messages.append(HumanMessage(content=message))

        recursion = max(8, self.max_iterations * 2)
        try:
            result = self._lite_graph.invoke(
                {"messages": messages},
                config={"recursion_limit": recursion},
            )
        except TypeError:
            # State schema refused the message list -- retry with the user
            # turn alone, folding the prompt in only when the graph is not
            # already carrying it.
            content = (
                message
                if self._lite_has_prompt
                else f"{self.system_prompt}\n\nUser: {message}"
            )
            result = self._lite_graph.invoke(
                {"messages": prior + [HumanMessage(content=content)]},
                config={"recursion_limit": recursion},
            )

        out_msgs = result.get("messages") if isinstance(result, dict) else messages
        final = ""
        if out_msgs:
            last = out_msgs[-1]
            content = getattr(last, "content", None)
            if isinstance(content, list):
                final = " ".join(
                    b.get("text", str(b)) if isinstance(b, dict) else str(b)
                    for b in content
                )
            else:
                final = str(content if content is not None else last)

        # Persist history (human + ai turns only, bounded)
        if not self.skip_memory:
            new_hist = list(prior)
            new_hist.append(HumanMessage(content=message))
            new_hist.append(AIMessage(content=final or ""))
            limit = max(4, self.session_limit * 2)
            if len(new_hist) > limit:
                new_hist = new_hist[-limit:]
            with _lock:
                self._sessions[sid] = new_hist

        # Count tool calls made during this turn
        tool_hits = 0
        for m in out_msgs or []:
            tool_hits += len(getattr(m, "tool_calls", None) or [])

        return {
            "success": bool((final or "").strip()),
            "response": final or None,
            "error": None if (final or "").strip() else "Empty host response",
            "session_id": sid,
            "backend": "hermes_lite",
            "tool_call_count": tool_hits,
            "agents_used": ["hermes_host"],
            "mode": "hermes_tool_host",
        }


def get_hermes_host() -> HermesHostService:
    global _service
    with _lock:
        if _service is None:
            _service = HermesHostService()
            _service.initialize()
        elif not _service.ready:
            _service.initialize()
        return _service
