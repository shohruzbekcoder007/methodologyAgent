"""Session history is owned by a user, not by whoever names the id.

These drive `HermesHostService.chat()` with the backend stubbed out, so they
assert the routing and keying, never the model.
"""

from __future__ import annotations

import pytest

from agents.hermes_host import HermesHostService
from agents.user_profiles import reset_seed_cache, resolve_profile


@pytest.fixture(autouse=True)
def _isolated_base(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_USERS_HOME", str(tmp_path / "users"))
    reset_seed_cache()
    yield
    reset_seed_cache()


@pytest.fixture
def host(monkeypatch):
    """A ready host whose backend records what it was handed."""
    service = HermesHostService()
    service._ready = True
    service._backend = "hermes_lite"
    service.task_model = ""  # never divert to the task model
    calls: list[dict] = []

    def fake_lite(message: str, sid: str, key: str) -> dict:
        calls.append({"message": message, "sid": sid, "key": key})
        # Mimic the real path: history is stored under the owner-prefixed key.
        service._sessions.setdefault(key, []).append(message)
        return {"success": True, "response": "ok", "session_id": sid}

    monkeypatch.setattr(service, "_chat_hermes_lite", fake_lite)
    service._calls = calls
    return service


def test_same_session_id_from_two_users_does_not_collide(host):
    a = resolve_profile("shohruz")
    b = resolve_profile("aziza")

    host.chat("birinchi", session_id="chat-1", profile=a)
    host.chat("ikkinchi", session_id="chat-1", profile=b)

    keys = [c["key"] for c in host._calls]
    assert keys[0] != keys[1]
    assert keys == ["shohruz:chat-1", "aziza:chat-1"]


def test_guessing_another_users_session_id_reveals_no_history(host):
    victim = resolve_profile("shohruz")
    host.chat("maxfiy gap", session_id="chat-1", profile=victim)

    attacker = resolve_profile("intruder")
    host.chat("nima deding?", session_id="chat-1", profile=attacker)

    victim_history = host._sessions[victim.session_key("chat-1")]
    attacker_history = host._sessions[attacker.session_key("chat-1")]
    assert "maxfiy gap" in victim_history
    assert "maxfiy gap" not in attacker_history


def test_returned_session_id_is_not_prefixed(host):
    """The client gets its own id back, so a round-trip cannot stack prefixes."""
    profile = resolve_profile("shohruz")
    result = host.chat("salom", session_id="chat-1", profile=profile)
    assert result["session_id"] == "chat-1"

    host.chat("yana", session_id=result["session_id"], profile=profile)
    assert host._calls[-1]["key"] == "shohruz:chat-1"


def test_reset_clears_only_the_callers_own_session(host):
    a = resolve_profile("shohruz")
    b = resolve_profile("aziza")
    host.chat("a-ning gapi", session_id="chat-1", profile=a)
    host.chat("b-ning gapi", session_id="chat-1", profile=b)

    host.chat("qaytadan", session_id="chat-1", reset_session=True, profile=b)

    assert host._sessions[a.session_key("chat-1")] == ["a-ning gapi"]
    assert host._sessions[b.session_key("chat-1")] == ["qaytadan"]


def test_missing_profile_falls_back_to_shared_not_to_someone_else(host):
    named = resolve_profile("shohruz")
    host.chat("shaxsiy", session_id="chat-1", profile=named)

    host.chat("kimman?", session_id="chat-1", profile=None)

    key = host._calls[-1]["key"]
    assert key.startswith("_shared:")
    assert key != named.session_key("chat-1")


def test_absent_session_id_still_gets_a_per_user_key(host):
    profile = resolve_profile("shohruz")
    result = host.chat("salom", profile=profile)
    assert host._calls[-1]["key"].startswith("shohruz:")
    # A generated id is handed back so the client can continue the thread.
    assert result["session_id"]
