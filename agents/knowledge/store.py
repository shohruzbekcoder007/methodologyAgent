"""Connections to the three services retrieval needs: Neo4j, MongoDB, embeddings.

All three are process-wide singletons. The Neo4j driver and `MongoClient` both
own connection pools and are documented as thread-safe, so building one per
request would throw the pool away on every call -- and the host agent's tool
loop is synchronous and runs in a thread pool, so that would be every tool
call. Synchronous drivers for the same reason: the loop is `def`, not
`async def`, and a `motor` client would need an event loop that is not there.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import json

logger = logging.getLogger("knowledge.store")

_lock = threading.Lock()
_driver: Any = None
_mongo: Any = None
_identity: Optional[dict[str, Any]] = None


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
def neo4j_driver() -> Any:
    global _driver
    if _driver is not None:
        return _driver
    with _lock:
        if _driver is None:
            from neo4j import GraphDatabase

            uri = _env("NEO4J_URI") or "bolt://localhost:7687"
            user = _env("NEO4J_USER") or "neo4j"
            password = _env("NEO4J_PASSWORD")
            if not password:
                raise RuntimeError("NEO4J_PASSWORD is not set")
            _driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("neo4j driver created uri=%s", uri)
    return _driver


def neo4j_database() -> str:
    return _env("NEO4J_DATABASE") or "neo4j"


def read_query(cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Run a read-only query.

    `default_access_mode=READ` is not decoration: Neo4j Community has no role
    system, so a read-only *session* is the only thing standing between a
    query and a write.
    """
    from neo4j import READ_ACCESS

    with neo4j_driver().session(
        database=neo4j_database(), default_access_mode=READ_ACCESS
    ) as session:
        return [dict(r) for r in session.run(cypher, **params)]


def write_query(cypher: str, **params: Any) -> Any:
    with neo4j_driver().session(database=neo4j_database()) as session:
        return session.run(cypher, **params).consume()


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
def mongo_db() -> Any:
    global _mongo
    if _mongo is None:
        with _lock:
            if _mongo is None:
                from pymongo import MongoClient

                uri = _env("MONGO_URI")
                if not uri:
                    raise RuntimeError("MONGO_URI is not set")
                _mongo = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=10_000)
                logger.info("mongo client created")
    return _mongo[_env("MONGO_DB") or "nmc"]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _embedding_base_url() -> str:
    url = _env("EMBEDDING_BASE_URL") or "http://host.docker.internal:8090"
    return url.rstrip("/")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = Request(
        _embedding_base_url() + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urlopen(req, timeout=_env_int("EMBEDDING_TIMEOUT", 120)) as resp:
        return json.load(resp)


def embedding_identity(*, refresh: bool = False) -> dict[str, Any]:
    """What the embedding service is serving right now.

    `index_key` (provider + model + dimensions) is stored next to every vector.
    Vectors from two different models are not comparable, so that key is what
    lets a re-run notice the model changed and re-embed instead of mixing them.
    """
    global _identity
    if _identity is None or refresh:
        try:
            with urlopen(
                _embedding_base_url() + "/v1/info",
                timeout=_env_int("EMBEDDING_TIMEOUT", 120),
            ) as resp:
                info = json.load(resp)
        except (HTTPError, URLError, OSError) as exc:
            raise RuntimeError(
                f"embedding service unreachable at {_embedding_base_url()}: {exc}"
            ) from exc
        ident = dict(info.get("identity") or {})
        ident["batch_max"] = int(info.get("batch_max") or 32)
        ident["max_chars"] = int(info.get("max_chars") or 8000)
        if not ident.get("dim"):
            raise RuntimeError("embedding service did not report a vector dimension")
        _identity = ident
    return _identity


def embed_texts(
    texts: Iterable[str], *, input_type: str = "document"
) -> list[list[float]]:
    """Embed a batch. `input_type` matters: bge-m3 encodes a question and a
    passage differently, and mixing the two costs retrieval quality."""
    batch = [t if t else " " for t in texts]
    if not batch:
        return []
    out = _post("/v1/embed/batch", {"texts": batch, "input_type": input_type})
    vectors = out.get("embeddings") or []
    if len(vectors) != len(batch):
        raise RuntimeError(
            f"embedding service returned {len(vectors)} vectors for {len(batch)} texts"
        )
    return vectors


def embed_query(text: str) -> list[float]:
    out = _post("/v1/embed", {"text": text or " ", "input_type": "query"})
    return out.get("embedding") or []


def close() -> None:
    """Release both pools (tests, shutdown)."""
    global _driver, _mongo
    with _lock:
        if _driver is not None:
            _driver.close()
            _driver = None
        if _mongo is not None:
            _mongo.close()
            _mongo = None
