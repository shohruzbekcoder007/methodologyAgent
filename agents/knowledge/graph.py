"""Walking the graph, reading a document through, and counting.

`retrieval.py` answers "what does the corpus say about X". This answers the
questions that are about structure rather than content: what is this document
connected to, what does the rest of it say, how many of them are there.

Together they are what makes the tool set cover the catalogue rather than a
corner of it: search alone reaches the 26 342 passages, and these reach the
other 77 relation types and the dimensions nobody can search their way to.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from agents.knowledge.retrieval import get_document
from agents.knowledge.store import mongo_db, read_query

logger = logging.getLogger("knowledge.graph")


# ---------------------------------------------------------------------------
# Document text, read in order
# ---------------------------------------------------------------------------
def document(
    sha256: str,
    *,
    include_text: bool = False,
    section: Optional[str] = None,
    offset: int = 0,
    limit_chars: int = 6000,
) -> dict[str, Any]:
    """One document: its card, and its text when asked for.

    Card and text were two tools until they were not. The inputs are the same
    and so is the document, and a model choosing between "tell me about this
    document" and "show me this document" is making a distinction the caller
    never has to: `include_text` says it in one argument.

    With text, chunks come back by `ordinal` -- the order they appear in the
    document -- so consecutive calls read it the way a person would, and
    `section` narrows to one part by a substring of its heading path.
    """
    card = get_document(sha256)
    if card is None:
        return {"error": f"no document with sha256 {sha256}"}
    if not include_text:
        return card
    if not card["has_text"]:
        # The card already carries the note and the readable sibling; an empty
        # string here would otherwise read as a document that says nothing,
        # rather than one whose text was never extracted.
        card["text"] = ""
        return card

    query: dict[str, Any] = {"sha256": sha256}
    if section:
        query["section_path"] = {"$regex": re.escape(section), "$options": "i"}
    cursor = (
        mongo_db()
        .chunks.find(query, {"text": 1, "section_path": 1, "ordinal": 1})
        .sort("ordinal", 1)
    )

    parts: list[str] = []
    spent = 0
    seen = 0
    next_offset: Optional[int] = None
    for doc in cursor:
        seen += 1
        if seen <= offset:
            continue
        body = doc.get("text") or ""
        if parts and spent + len(body) > limit_chars:
            next_offset = seen - 1
            break
        parts.append(body)
        spent += len(body)

    card.update(
        {
            "section": section,
            "text": "\n\n".join(parts),
            "chunks_returned": len(parts),
            "next_offset": next_offset,
            "has_more": next_offset is not None,
            "citation": {
                "sha256": sha256,
                "title": card.get("title"),
                "rel_path": card.get("rel_path"),
                "collection": card.get("collection"),
            },
        }
    )
    return card


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------
# A relationship type cannot be a query parameter -- Cypher needs it at parse
# time -- so it has to be written into the query string. That makes validation
# the only thing between a tool argument and an injected clause, which is why
# there are two checks: the shape, and membership of the set the database
# actually has.
_RELATION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_relation_types: Optional[set[str]] = None

# `:Nmc` sits on every node so the corpus can be told apart from anything else
# in the database, and `:Readable` and the kind labels are extras. None of
# them says what a node is, so they are skipped when naming a neighbour.
_LABEL_NOISE = ("Nmc", "Readable")


def relation_types(*, refresh: bool = False) -> set[str]:
    """Every relation type in the database, cached."""
    global _relation_types
    if _relation_types is None or refresh:
        rows = read_query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t"
        )
        _relation_types = {r["t"] for r in rows}
        logger.info("relation types loaded: %d", len(_relation_types))
    return _relation_types


_NEIGHBOUR_RETURN = """
RETURN [l IN labels(m) WHERE NOT l IN $noise][0]              AS label,
       coalesce(m.title, m.name, m.key, m.code, toString(m.value),
                m.rel_path, m.path, m.sha256)                 AS name,
       m.sha256             AS sha256,
       m.mongo_collection   AS collection,
       m.review_state       AS review_state,
       m.primary_collection AS doc_collection
LIMIT $limit
"""


def _start_clause(
    sha256: Optional[str], key: Optional[str]
) -> tuple[str, dict[str, Any]]:
    if sha256:
        return "MATCH (n:Document {sha256: $sha})", {"sha": sha256}
    if key:
        # Keyed nodes: ActReference, Term, LegalAct, Law and the rest all
        # carry `key`, so one clause reaches any of them.
        return "MATCH (n {key: $key})", {"key": key}
    raise ValueError("either sha256 or key is required")


def related_summary(
    *, sha256: Optional[str] = None, key: Optional[str] = None
) -> list[dict[str, Any]]:
    """What this node is connected to, by relation type and count.

    The catalogue has 77 relation types. Handing all of them to a model and
    asking it to choose is a worse bet than showing it the handful this node
    actually has, and letting it drill into one.
    """
    start, params = _start_clause(sha256, key)
    rows = read_query(
        f"{start}-[r]->(m) RETURN type(r) AS relation, 'out' AS direction, count(*) AS count "
        f"UNION {start}<-[r]-(m) RETURN type(r) AS relation, 'in' AS direction, count(*) AS count",
        **params,
    )
    return sorted(rows, key=lambda r: -r["count"])


def find_related(
    *,
    sha256: Optional[str] = None,
    key: Optional[str] = None,
    relation: Optional[str] = None,
    direction: str = "both",
    limit: int = 20,
) -> dict[str, Any]:
    """Walk one relation type out of one node.

    With no `relation` it returns the summary instead, so a caller that does
    not know what is there finds out in one call rather than guessing.
    """
    if not (sha256 or key):
        return {"error": "either sha256 or key is required"}
    if not relation:
        return {"summary": related_summary(sha256=sha256, key=key)}

    relation = relation.strip().upper()
    if not _RELATION_RE.match(relation):
        return {"error": f"invalid relation name: {relation!r}"}
    known = relation_types()
    if relation not in known:
        near = sorted(t for t in known if relation in t or t in relation)[:5]
        return {"error": f"unknown relation {relation!r}", "did_you_mean": near}

    start, params = _start_clause(sha256, key)
    params["limit"] = max(1, min(int(limit), 100))
    params["noise"] = list(_LABEL_NOISE)

    if direction == "out":
        patterns = [(f"{start}-[:{relation}]->(m)", "out")]
    elif direction == "in":
        patterns = [(f"{start}<-[:{relation}]-(m)", "in")]
    else:
        patterns = [
            (f"{start}-[:{relation}]->(m)", "out"),
            (f"{start}<-[:{relation}]-(m)", "in"),
        ]

    nodes: list[dict[str, Any]] = []
    for pattern, side in patterns:
        for row in read_query(pattern + _NEIGHBOUR_RETURN, **params):
            row["direction"] = side
            nodes.append(row)

    return {
        "relation": relation,
        "direction": direction,
        "count": len(nodes),
        "nodes": nodes[: params["limit"]],
    }


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
# Structured filters rather than Cypher. A model writing its own aggregation
# over a graph this shape gets it wrong often enough to matter, and a wrong
# count reads exactly like a right one.
_DIMENSIONS: dict[str, tuple[str, str, str]] = {
    "collection": ("IN_COLLECTION", "Collection", "name"),
    "year": ("IN_YEAR_DIR", "Year", "value"),
    "kind": ("OF_KIND", "DocumentKind", "name"),
    "folder": ("IN_FOLDER", "Folder", "code"),
}


def dimension_values() -> dict[str, list[Any]]:
    """What each filter accepts, so a caller can choose instead of guess."""
    out: dict[str, list[Any]] = {}
    for name, (_rel, label, prop) in _DIMENSIONS.items():
        rows = read_query(f"MATCH (n:{label}) RETURN n.{prop} AS v ORDER BY v")
        out[name] = [r["v"] for r in rows if r["v"] is not None]
    return out


def count_documents(
    *,
    collection: Optional[str] = None,
    year: Optional[int] = None,
    kind: Optional[str] = None,
    folder: Optional[str] = None,
    group_by: Optional[str] = None,
) -> dict[str, Any]:
    """Count documents by folder, year, kind and collection.

    `collection` is where a file was found, not its legal standing: a document
    under `bekor` is not thereby repealed, and an answer built on this count
    must not say that it is.
    """
    filters = {"collection": collection, "year": year, "kind": kind, "folder": folder}
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for name, value in filters.items():
        if value is None or value == "":
            continue
        rel, label, prop = _DIMENSIONS[name]
        try:
            params[name] = int(value) if name == "year" else str(value)
        except (TypeError, ValueError):
            return {"error": f"{name} must be a number, got {value!r}"}
        clauses.append(f"MATCH (d)-[:{rel}]->(:{label} {{{prop}: ${name}}})")

    applied = {k: v for k, v in filters.items() if v not in (None, "")}

    if group_by:
        if group_by not in _DIMENSIONS:
            return {"error": f"group_by must be one of {sorted(_DIMENSIONS)}"}
        rel, label, prop = _DIMENSIONS[group_by]
        rows = read_query(
            "MATCH (d:Document) "
            + " ".join(clauses)
            + f" MATCH (d)-[:{rel}]->(g:{label}) "
            f"RETURN g.{prop} AS group, count(DISTINCT d) AS count ORDER BY count DESC",
            **params,
        )
        return {
            "filters": applied,
            "group_by": group_by,
            "groups": rows,
            "total": sum(r["count"] for r in rows),
        }

    rows = read_query(
        "MATCH (d:Document) " + " ".join(clauses) + " RETURN count(DISTINCT d) AS count",
        **params,
    )
    return {"filters": applied, "count": rows[0]["count"] if rows else 0}
