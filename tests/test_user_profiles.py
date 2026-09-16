"""Slug validation and profile containment — the header-to-filesystem boundary."""

from __future__ import annotations

import hashlib

import pytest

from agents.user_profiles import (
    SHARED_SLUG,
    InvalidUserId,
    resolve_profile,
    reset_seed_cache,
    slugify_user_id,
    users_home,
)


@pytest.fixture(autouse=True)
def _isolated_base(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_USERS_HOME", str(tmp_path / "users"))
    reset_seed_cache()
    yield
    reset_seed_cache()


# --- slug validation -------------------------------------------------------

@pytest.mark.parametrize("raw", ["shohruz", "aziza", "user-1", "a.b_c", "A1", "x" * 64])
def test_readable_ids_pass_through(raw):
    assert slugify_user_id(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc",
        "..",
        ".",
        "...",
        "a/b",
        "a\b",
        "/absolute",
        "x\x00y",
        "line\nbreak",
        "tab\there",
        "",
        "   ",
        "x" * 201,
    ],
)
def test_dangerous_ids_are_rejected(raw):
    with pytest.raises(InvalidUserId):
        slugify_user_id(raw)


def test_missing_id_is_rejected():
    with pytest.raises(InvalidUserId):
        slugify_user_id(None)


@pytest.mark.parametrize("raw", ["a@b.com", "Шохруз", "has space", "x" * 65, "_leading"])
def test_unusual_but_legitimate_ids_are_hashed(raw):
    slug = slugify_user_id(raw)
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    assert slug == f"u-{expected}"
    # Still a single safe path segment.
    assert "/" not in slug and "\\" not in slug and ".." not in slug


def test_hashing_is_stable_and_distinct():
    assert slugify_user_id("a@b.com") == slugify_user_id("a@b.com")
    assert slugify_user_id("a@b.com") != slugify_user_id("c@d.com")


def test_shared_slug_cannot_be_claimed_by_a_user():
    """A caller sending the shared profile's name must not land on it."""
    assert slugify_user_id(SHARED_SLUG) != SHARED_SLUG


# --- containment -----------------------------------------------------------

def test_profile_stays_inside_base():
    base = users_home().resolve()
    for raw in ["shohruz", "a@b.com", "Шохруз", "x" * 64]:
        profile = resolve_profile(raw)
        assert profile.home.is_relative_to(base)
        assert profile.home.parent == base


def test_traversal_never_reaches_the_filesystem():
    with pytest.raises(InvalidUserId):
        resolve_profile("../../../etc/passwd")


def test_distinct_users_get_distinct_homes():
    a = resolve_profile("shohruz")
    b = resolve_profile("aziza")
    assert a.home != b.home


def test_profile_is_seeded_on_first_use():
    profile = resolve_profile("shohruz")
    for sub in ("memories", "skills", "sessions"):
        assert (profile.home / sub).is_dir(), sub
    assert (profile.home / "config.yaml").is_file()


def test_shared_profile_used_when_no_id():
    profile = resolve_profile(None)
    assert profile.slug == SHARED_SLUG
    assert profile.is_shared
    assert profile.raw_id is None


# --- session keys ----------------------------------------------------------

def test_session_keys_are_namespaced_per_user():
    a = resolve_profile("shohruz")
    b = resolve_profile("aziza")
    # The IDOR: both ask for the same session id and must not collide.
    assert a.session_key("chat-1") != b.session_key("chat-1")
    assert a.session_key("chat-1").startswith("shohruz:")
