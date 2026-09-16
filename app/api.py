"""
FastAPI — Hermes host starter.

  Open WebUI → Gateway → POST /v1/chat
       → Hermes host (context/memory)
            → tools registered in agents.hermes_host
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.user_profiles import InvalidUserId, UserProfile, resolve_profile
from app import __version__
from app.rate_limit import RateLimiter

logger = logging.getLogger("app")

# Header carrying the end user's identity, set by the gateway *after* it has
# authenticated them. It is trusted only because the gateway bearer token is
# required on the same request -- see `_check_bearer`. Never accept it from an
# unauthenticated caller.
USER_ID_HEADER = "X-User-Id"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
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


def _max_message_chars() -> int:
    return max(1, _env_int("MAX_MESSAGE_CHARS", 8000))


class ChatRequest(BaseModel):
    """Hermes-compatible chat body (gateway Open WebUI platform)."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=_max_message_chars(),
        description="User question",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Multi-turn session id (Hermes host memory)",
    )
    reset_session: bool = Field(
        default=False,
        description="Clear Hermes host session history",
    )


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    retryable: Optional[bool] = None
    tools_called: Optional[list[dict[str, Any]]] = None
    tool_call_count: Optional[int] = None
    agents_used: Optional[list[str]] = None
    mode: Optional[str] = None
    backend: Optional[str] = None


def _cors_origins() -> list[str]:
    """Allowed browser origins.

    `*` together with `allow_credentials=True` is the combination browsers
    refuse anyway and that Starlette silently downgrades, so it is not offered:
    an unset or wildcard value means no cross-origin access at all. The gateway
    is a server-side caller and needs none.
    """
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw or raw == "*":
        if raw == "*":
            logger.warning(
                "CORS_ORIGINS=* is refused with credentialed requests; "
                "cross-origin access disabled. List exact origins to enable it."
            )
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


def _gateway_token() -> str:
    """The shared secret the gateway must present.

    `GATEWAY_TOKEN` is the current name; `API_BEARER_TOKEN` stays accepted so
    an existing deployment keeps working.
    """
    return (
        os.getenv("GATEWAY_TOKEN", "").strip()
        or os.getenv("API_BEARER_TOKEN", "").strip()
    )


def _require_auth_configured() -> None:
    """Refuse to serve without a gateway token.

    Previously an unset token disabled authentication entirely, which meant a
    forgotten setting silently published /v1/chat. Failing at startup is the
    safer direction for the mistake to fall. `ALLOW_UNAUTHENTICATED=true` is
    the explicit opt-out for local development.
    """
    if _gateway_token():
        return
    if _env_bool("ALLOW_UNAUTHENTICATED", False):
        logger.warning(
            "ALLOW_UNAUTHENTICATED=true and no GATEWAY_TOKEN set — /v1/chat is "
            "open. Never do this on a reachable network."
        )
        return
    raise RuntimeError(
        "GATEWAY_TOKEN is not set. Set it to the shared secret the gateway "
        "sends, or set ALLOW_UNAUTHENTICATED=true for local development only."
    )


def _check_bearer(
    authorization: Optional[str] = Header(default=None),
) -> None:
    expected = _gateway_token()
    if not expected:
        # Only reachable with ALLOW_UNAUTHENTICATED=true; startup refuses
        # otherwise.
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _resolve_caller(
    x_user_id: Optional[str] = Header(default=None),
) -> UserProfile:
    """Map the gateway's `X-User-Id` header to the caller's own Hermes profile.

    Requiring the header by default is deliberate: a missing identity used to
    mean everyone shared one memory, which is the leak this exists to close.
    `HERMES_REQUIRE_USER_ID=false` restores a single shared profile for a
    genuinely single-user deployment.
    """
    require = _env_bool("HERMES_REQUIRE_USER_ID", True)

    if x_user_id is None or not x_user_id.strip():
        if require:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{USER_ID_HEADER} header is required. The gateway must "
                    "send the authenticated end user's id, or set "
                    "HERMES_REQUIRE_USER_ID=false to share one profile."
                ),
            )
        return resolve_profile(None)

    try:
        return resolve_profile(x_user_id)
    except InvalidUserId as exc:
        # Logged at warning because the rejected values worth seeing here are
        # traversal attempts, not typos.
        logger.warning("rejected %s=%r: %s", USER_ID_HEADER, x_user_id[:120], exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {USER_ID_HEADER}: {exc}",
        ) from exc


_limiter = RateLimiter(
    rate_per_minute=float(_env_int("RATE_LIMIT_PER_MINUTE", 30)),
    burst=_env_int("RATE_LIMIT_BURST", 10),
)


def _check_rate_limit(profile: UserProfile, request: Request) -> None:
    """One bucket per user, falling back to the peer address for the shared
    profile so it cannot be drained by a single client on everyone's behalf."""
    if not _limiter.enabled:
        return
    key = profile.slug
    if profile.is_shared:
        client = request.client.host if request.client else "unknown"
        key = f"{profile.slug}:{client}"
    allowed, retry_after = _limiter.check(key)
    if not allowed:
        logger.warning("rate limit hit for %s (retry in %ss)", key, retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )


def create_app() -> FastAPI:
    _require_auth_configured()

    app = FastAPI(
        title=os.getenv("APP_NAME", "methodologyagent"),
        version=__version__,
        description=(
            "Hermes host agent with session memory and a tool-calling loop. "
            "Open WebUI gateway compatible (POST /v1/chat)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        logger.info("Starting Hermes host service v%s", __version__)
        try:
            from agents.hermes_host import get_hermes_host

            host = get_hermes_host()
            logger.info("Hermes host readiness: %s", host.readiness())
        except Exception:
            logger.exception("Hermes host init failed — /ready may be 503")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": os.getenv("APP_NAME", "methodologyagent")}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        """Readiness probe. Unauthenticated, so the body stays minimal.

        The status code carries the signal an orchestrator needs; the full
        diagnostic view (including the upstream address and the last error)
        is on the authenticated /v1/info, and the detail is logged here.
        """
        from agents.hermes_host import get_hermes_host

        host = get_hermes_host()
        if not host.ready:
            host.initialize()
        rd = host.readiness()
        if not rd.get("ready"):
            logger.warning("readiness probe failed: %s", rd)
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready"},
            )
        return {"status": "ready", "backend": rd.get("backend")}

    @app.get("/v1/info")
    def info(_: None = Depends(_check_bearer)) -> dict[str, Any]:
        from agents.hermes_host import get_hermes_host

        host = get_hermes_host()
        rd = host.readiness()
        return {
            "service": os.getenv("APP_NAME", "methodologyagent"),
            "version": __version__,
            "design": "hermes-host",
            "architecture": rd.get("architecture"),
            "backend": rd.get("backend"),
            "gateway_compatible": True,
            "hermes_chat_path": "/v1/chat",
            "tools": rd.get("tools"),
            "toolsets": rd.get("toolsets"),
            "ready": host.ready,
            "provider": rd.get("provider"),
            "model": rd.get("model"),
            "task_model": rd.get("task_model"),
            # `base_url` is deliberately omitted: it is the internal vLLM /
            # Ollama address and nothing downstream needs it.
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        request: Request,
        _: None = Depends(_check_bearer),
        caller: UserProfile = Depends(_resolve_caller),
    ) -> ChatResponse:
        """Gateway entry: Hermes host keeps context and calls its tools.

        Runs against `caller`'s own Hermes profile, so memory, session history
        and `session_search` see only that user's data.
        """
        from agents.hermes_host import get_hermes_host

        _check_rate_limit(caller, request)

        try:
            host = get_hermes_host()
            logger.info(
                "POST /v1/chat user=%s session_id=%r reset=%s msg_len=%d "
                "msg_preview=%r",
                caller.slug,
                body.session_id,
                body.reset_session,
                len(body.message or ""),
                (body.message or "")[:80],
            )
            result = host.chat(
                body.message,
                session_id=body.session_id,
                reset_session=body.reset_session,
                profile=caller,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("chat endpoint failed: %s", exc, exc_info=True)
            return ChatResponse(
                success=False,
                response=None,
                session_id=body.session_id,
                error="Ichki server xatosi. Iltimos keyinroq urinib ko'ring.",
                error_code="internal",
                error_detail=str(exc)[:500],
                retryable=True,
            )
        return ChatResponse(
            success=bool(result.get("success")),
            response=result.get("response"),
            session_id=result.get("session_id") or body.session_id,
            error=result.get("error"),
            error_code=result.get("error_code"),
            error_detail=result.get("error_detail"),
            retryable=result.get("retryable"),
            tools_called=result.get("tools_called"),
            tool_call_count=result.get("tool_call_count"),
            agents_used=result.get("agents_used"),
            mode=result.get("mode"),
            backend=result.get("backend"),
        )

    return app


app = create_app()
