"""The HTTP boundary: gateway token, user header, and what leaks without them."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api as api
from agents.user_profiles import reset_seed_cache

TOKEN = "test-gateway-token"


class _StubHost:
    """Stands in for the Hermes host so no request reaches a model."""

    ready = True

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def readiness(self) -> dict:
        return {"ready": True, "backend": "stub", "base_url": "http://secret:8003/v1"}

    def initialize(self) -> dict:
        return self.readiness()

    def chat(self, message, *, session_id=None, reset_session=False, profile=None):
        self.seen.append({"message": message, "profile": profile})
        return {"success": True, "response": "ok", "session_id": session_id or "s"}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_USERS_HOME", str(tmp_path / "users"))
    monkeypatch.setenv("GATEWAY_TOKEN", TOKEN)
    monkeypatch.delenv("API_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("HERMES_REQUIRE_USER_ID", raising=False)
    reset_seed_cache()
    api._limiter.reset()
    yield
    reset_seed_cache()


@pytest.fixture
def stub_host(monkeypatch):
    host = _StubHost()
    monkeypatch.setattr("agents.hermes_host.get_hermes_host", lambda: host)
    return host


@pytest.fixture
def client(stub_host):
    # Not a context manager on purpose: no lifespan, so startup never builds a
    # real host.
    return TestClient(api.create_app())


def _post(client, **headers):
    return client.post("/v1/chat", json={"message": "salom"}, headers=headers)


# --- startup ---------------------------------------------------------------

def test_startup_refuses_without_a_token(monkeypatch):
    monkeypatch.delenv("GATEWAY_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GATEWAY_TOKEN"):
        api.create_app()


def test_startup_allows_the_explicit_local_opt_out(monkeypatch):
    monkeypatch.delenv("GATEWAY_TOKEN", raising=False)
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
    assert api.create_app() is not None


# --- gateway token ---------------------------------------------------------

def test_chat_without_authorization_is_401(client):
    assert _post(client, **{"X-User-Id": "shohruz"}).status_code == 401


def test_chat_with_a_wrong_token_is_401(client):
    r = _post(client, Authorization="Bearer nope", **{"X-User-Id": "shohruz"})
    assert r.status_code == 401


def test_info_requires_the_token_too(client):
    assert client.get("/v1/info").status_code == 401


def test_health_stays_open(client):
    assert client.get("/health").status_code == 200


# --- user header -----------------------------------------------------------

def test_chat_without_a_user_header_is_refused(client):
    r = _post(client, Authorization=f"Bearer {TOKEN}")
    assert r.status_code == 400
    assert "X-User-Id" in r.json()["detail"]


def test_traversal_in_the_user_header_is_refused(client):
    r = _post(client, Authorization=f"Bearer {TOKEN}", **{"X-User-Id": "../../etc"})
    assert r.status_code == 400


def test_valid_caller_reaches_the_host_with_its_own_profile(client, stub_host):
    r = _post(client, Authorization=f"Bearer {TOKEN}", **{"X-User-Id": "shohruz"})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert stub_host.seen[-1]["profile"].slug == "shohruz"


def test_shared_profile_only_when_explicitly_allowed(monkeypatch, stub_host):
    monkeypatch.setenv("HERMES_REQUIRE_USER_ID", "false")
    client = TestClient(api.create_app())
    r = _post(client, Authorization=f"Bearer {TOKEN}")
    assert r.status_code == 200
    assert stub_host.seen[-1]["profile"].is_shared


# --- input limits ----------------------------------------------------------

def test_oversized_message_is_rejected_before_the_model(client, stub_host):
    huge = "x" * (api._max_message_chars() + 1)
    r = client.post(
        "/v1/chat",
        json={"message": huge},
        headers={"Authorization": f"Bearer {TOKEN}", "X-User-Id": "shohruz"},
    )
    assert r.status_code == 422
    assert stub_host.seen == []


# --- information disclosure ------------------------------------------------

def test_ready_does_not_expose_the_upstream_address(client):
    body = client.get("/ready").text
    assert "secret" not in body


def test_info_does_not_expose_the_upstream_address(client):
    body = client.get("/v1/info", headers={"Authorization": f"Bearer {TOKEN}"}).text
    assert "secret" not in body
