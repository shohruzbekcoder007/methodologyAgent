"""Concurrent requests must not see each other's Hermes profile.

The whole design rests on `set_hermes_home_override()` being context-local.
This asserts that rather than trusting it: the risk flagged in the plan was
two users overlapping inside one process.

Skipped where the Hermes package is absent (local dev on the hermes_lite
path); it runs inside the container, which is where it matters.
"""

from __future__ import annotations

import threading

import pytest

from agents.hermes_host import _hermes_home

hermes_constants = pytest.importorskip(
    "hermes_constants", reason="Hermes package not installed (hermes_lite path)"
)
get_hermes_home = hermes_constants.get_hermes_home


def test_override_applies_and_is_undone(tmp_path):
    before = get_hermes_home()
    with _hermes_home(tmp_path / "shohruz"):
        assert str(get_hermes_home()) == str(tmp_path / "shohruz")
    assert get_hermes_home() == before, "override must not outlive the block"


def test_override_is_undone_even_when_the_block_raises(tmp_path):
    before = get_hermes_home()
    with pytest.raises(RuntimeError):
        with _hermes_home(tmp_path / "shohruz"):
            raise RuntimeError("tool blew up")
    assert get_hermes_home() == before


def test_concurrent_users_never_see_each_others_home(tmp_path):
    """Threads overlap on purpose: each must stay in its own profile."""
    users = [f"user-{i}" for i in range(8)]
    start = threading.Barrier(len(users))
    seen: dict[str, list[str]] = {}
    errors: list[str] = []

    def run(name: str) -> None:
        home = tmp_path / name
        try:
            with _hermes_home(home):
                # Hold every thread inside its override at the same time.
                start.wait(timeout=10)
                observed = [str(get_hermes_home()) for _ in range(20)]
            seen[name] = observed
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    threads = [threading.Thread(target=run, args=(u,)) for u in users]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, errors
    assert len(seen) == len(users)
    for name, observed in seen.items():
        expected = str(tmp_path / name)
        assert set(observed) == {expected}, f"{name} saw {set(observed)}"
