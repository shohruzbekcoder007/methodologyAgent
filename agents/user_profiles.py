"""
Per-user Hermes profile isolation.

Each user gets a full Hermes home of their own:

    <HERMES_USERS_HOME>/<slug>/
        config.yaml  memories/  skills/  sessions/  plugins/  logs/

`agents.hermes_host` activates one of these per request with Hermes'
`set_hermes_home_override()` context var, so memory, session history and
`session_search` all resolve inside the caller's own directory. Hermes reads
its home through `get_hermes_home()` on every call for exactly this reason
(see `tools/memory_tool.py::get_memory_dir`), so the redirect reaches the
tools too — not just the agent object.

The slug is the security boundary: it comes from a request header and ends up
as a directory name, so it is validated here and nowhere else.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("user_profiles")

# Readable ids pass through as-is. Anchored, and the first character must be
# alphanumeric, which is what rules out "." and ".." (and dotfiles, and names
# starting with "-") without needing a reserved-word list.
_SAFE_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Longer than any plausible user id; a header is attacker-controlled, so it is
# bounded before it reaches hashlib or the filesystem.
_MAX_RAW_LEN = 200

# Anything here is a path-traversal attempt or a protocol error, not an
# unusual-but-real user id, so it is refused loudly instead of being hashed
# into something harmless. Hashing these would work, but it would also hide
# an attack from the logs.
_ATTACK_CHARS = ("/", "\\", "\x00")

# Profile used when no user id is supplied and HERMES_REQUIRE_USER_ID=false.
# Cannot collide with a real slug: slugs must start alphanumeric.
SHARED_SLUG = "_shared"

# Prefix for hashed ids, so a hashed profile is recognisable on disk and can
# never collide with a passthrough slug of the same text.
_HASH_PREFIX = "u-"


class InvalidUserId(ValueError):
    """The supplied user id cannot be mapped to a profile directory."""


@dataclass(frozen=True)
class UserProfile:
    """A resolved, verified per-user Hermes home."""

    slug: str
    home: Path
    # The id as it arrived, for Hermes' own identity fields and for logs.
    # None for the shared profile.
    raw_id: Optional[str]

    @property
    def is_shared(self) -> bool:
        return self.slug == SHARED_SLUG

    def session_key(self, session_id: str) -> str:
        """Namespace a client-supplied session id to this user.

        Guessing someone else's `session_id` then buys nothing: the key it
        lands on is prefixed with the caller's own slug, so the lookup finds
        their own (empty) session rather than the owner's history.
        """
        return f"{self.slug}:{session_id}"


def slugify_user_id(raw_id: Optional[str]) -> str:
    """Map a request-supplied user id to a safe single path segment.

    Raises `InvalidUserId` for empty input, over-long input, and anything
    carrying a path separator or NUL. Ids that are legitimate but not
    directory-safe (emails, non-ASCII names, spaces) are hashed rather than
    rejected, so a real user is never locked out by the character set.
    """
    if raw_id is None:
        raise InvalidUserId("user id is missing")

    raw = raw_id.strip()
    if not raw:
        raise InvalidUserId("user id is empty")
    if len(raw) > _MAX_RAW_LEN:
        raise InvalidUserId(f"user id longer than {_MAX_RAW_LEN} characters")
    for bad in _ATTACK_CHARS:
        if bad in raw:
            raise InvalidUserId("user id must not contain path separators")
    # Control characters (including newlines, which would also corrupt logs).
    if any(unicodedata.category(ch) == "Cc" for ch in raw):
        raise InvalidUserId("user id must not contain control characters")
    if raw in {".", ".."} or set(raw) == {"."}:
        raise InvalidUserId("user id must not be a path reference")

    if _SAFE_SLUG_RE.match(raw) and raw != SHARED_SLUG:
        return raw

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{_HASH_PREFIX}{digest}"


def users_home() -> Path:
    """Base directory holding every per-user profile.

    Defaults to a sibling of `HERMES_HOME` (`/home/appuser/.hermes-users`
    in the image) rather than a subdirectory of it, so a per-user home is
    never nested inside the shared one.
    """
    raw = os.getenv("HERMES_USERS_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    hermes_home = os.getenv("HERMES_HOME", "").strip()
    base = Path(hermes_home).expanduser() if hermes_home else Path.home() / ".hermes"
    return base.parent / ".hermes-users"


def _seed_template() -> Optional[Path]:
    """Hermes config copied into each new profile, if one is shipped."""
    raw = os.getenv("HERMES_PROFILE_TEMPLATE", "").strip()
    candidates = [Path(raw)] if raw else []
    candidates.append(Path(__file__).resolve().parent.parent / "config" / "hermes_config.yaml")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _soul_template() -> Optional[Path]:
    """The agent's own `SOUL.md`, seeded into every profile.

    The framework writes a stock SOUL.md on first run, and its opening sentence
    names the framework and its vendor. That text is injected into the model's
    context on every turn, so leaving it in place means the agent is being told
    one identity while the system prompt gives it another. Ours replaces it.
    """
    raw = os.getenv("AGENT_SOUL_TEMPLATE", "").strip()
    candidates = [Path(raw)] if raw else []
    candidates.append(Path(__file__).resolve().parent.parent / "prompts" / "agent_soul.md")
    for path in candidates:
        if path.is_file():
            return path
    return None


# Markers of the framework's stock SOUL.md. A file carrying one of these is a
# default we may overwrite; anything else is either ours already or something
# the agent wrote about itself, and is left alone.
_STOCK_SOUL_MARKERS = ("Hermes Agent", "Nous Research")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install_soul(home: Path) -> None:
    """Put our SOUL.md in `home`, keeping it in step with the template.

    Three cases, and the third is the reason this is not a one-line copy:

    * no file, or the framework's stock one -> write ours;
    * ours, unchanged since we wrote it, but the template has moved on ->
      write the new one, so editing `prompts/agent_soul.md` reaches profiles
      that already exist rather than only new ones;
    * edited since we wrote it -> leave it. That is either an operator's
      change or the agent writing about itself, and neither should be lost.

    The sidecar records what we last wrote, which is what makes "unchanged
    since we wrote it" answerable. It is a hash, not a copy, and it stays out
    of SOUL.md so nothing extra reaches the model's context.
    """
    template = _soul_template()
    if template is None:
        return
    soul = home / "SOUL.md"
    sidecar = home / ".soul_installed_sha256"

    try:
        wanted = template.read_bytes()
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not read SOUL template %s: %s", template, exc)
        return

    reason = "seeding"
    if soul.exists():
        try:
            current = soul.read_bytes()
        except OSError as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", soul, exc)
            return
        if current == wanted:
            return
        text = current.decode("utf-8", errors="replace")
        if any(marker in text for marker in _STOCK_SOUL_MARKERS):
            reason = "replacing stock"
        elif sidecar.is_file() and sidecar.read_text(
            encoding="utf-8", errors="replace"
        ).strip() == _digest(current):
            reason = "updating from template"
        else:
            # Edited by hand or by the agent — not ours to overwrite.
            return
        logger.info("%s SOUL.md in %s", reason, home)

    soul.write_bytes(wanted)
    try:
        sidecar.write_text(_digest(wanted), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        # Only costs us the ability to auto-update this profile later.
        logger.warning("could not write %s: %s", sidecar, exc)


# Seeding touches the filesystem, so it happens once per slug per process
# rather than on every request.
_seeded: set[str] = set()
_seed_lock = threading.Lock()


def _seed_profile(home: Path) -> None:
    for sub in ("memories", "skills", "sessions", "plugins", "logs"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    config = home / "config.yaml"
    if not config.exists():
        template = _seed_template()
        if template is not None:
            shutil.copyfile(template, config)
    _install_soul(home)


def resolve_profile(raw_id: Optional[str], *, create: bool = True) -> UserProfile:
    """Resolve a user id to its verified profile home, creating it on first use.

    The slug is validated before it is joined, and the joined path is then
    resolved and checked to be inside the base directory. Either check alone
    would do; both are here because this is the one place a header reaches
    the filesystem.
    """
    if raw_id is None:
        slug, raw = SHARED_SLUG, None
    else:
        slug, raw = slugify_user_id(raw_id), raw_id.strip()

    base = users_home()
    base.mkdir(parents=True, exist_ok=True)
    # strict=True: base exists by now, so this collapses any symlink in it and
    # the containment check below compares fully-resolved paths.
    base = base.resolve(strict=True)

    home = (base / slug).resolve(strict=False)
    if home == base or not home.is_relative_to(base):
        # Unreachable via slugify_user_id; kept because it is the check that
        # actually enforces the boundary if that regex is ever loosened.
        raise InvalidUserId("resolved profile escapes the base directory")

    if create and slug not in _seeded:
        with _seed_lock:
            if slug not in _seeded:
                existed = home.exists()
                _seed_profile(home)
                _seeded.add(slug)
                if not existed:
                    logger.info("created Hermes profile slug=%s home=%s", slug, home)

    return UserProfile(slug=slug, home=home, raw_id=raw)


def reset_seed_cache() -> None:
    """Forget which profiles were seeded (tests)."""
    with _seed_lock:
        _seeded.clear()
