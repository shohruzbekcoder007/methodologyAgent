"""Retrieval over the corpus: find in Neo4j, read in MongoDB.

The shape of a search here is hybrid-then-expand:

    query -> BM25 over `search_text`  ┐
          -> vector over `embedding`  ┘-> RRF -> chunks
          -> graph expansion (document, section, neighbours, references)
          -> original text from MongoDB
          -> citations and flags

Both halves of the hybrid are needed and for different reasons. BM25 finds an
exact wording -- an act number, a term as written; the vector half finds a
passage that answers the question without repeating its words. Neither alone
covers a corpus where users ask both kinds of question.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.knowledge.store import embed_query, mongo_db, read_query
from agents.knowledge.text import normalize_query

logger = logging.getLogger("knowledge.retrieval")

CHUNK_FULLTEXT_INDEX = "nmc_chunk_text"
CHUNK_VECTOR_INDEX = "nmc_chunk_vec"
TERM_FULLTEXT_INDEX = "nmc_term"
TERM_VECTOR_INDEX = "nmc_term_vec"
DOCUMENT_FULLTEXT_INDEX = "nmc_document_text"

# Reciprocal rank fusion. 60 is the constant from the original paper and the
# usual default; it flattens the head of each list enough that one engine's
# very confident top hit cannot outvote a passage both engines agree on.
RRF_K = 60


def _rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _bm25_chunks(query: str, k: int) -> list[str]:
    q = normalize_query(query)
    if not q:
        return []
    rows = read_query(
        f"CALL db.index.fulltext.queryNodes('{CHUNK_FULLTEXT_INDEX}', $q, {{limit: $k}}) "
        "YIELD node AS c, score "
        "RETURN c.chunk_id AS key ORDER BY score DESC",
        q=q,
        k=k,
    )
    return [r["key"] for r in rows if r.get("key")]


def _vector_chunks(query: str, k: int) -> list[str]:
    try:
        vector = embed_query(query)
    except Exception as exc:  # noqa: BLE001
        # A missing embedding service degrades the search to BM25 rather than
        # failing it: half a result set beats none.
        logger.warning("vector half unavailable, falling back to BM25 only: %s", exc)
        return []
    if not vector:
        return []
    try:
        rows = read_query(
            f"CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', $k, $vec) "
            "YIELD node AS c, score "
            "RETURN c.chunk_id AS key ORDER BY score DESC",
            k=k,
            vec=vector,
        )
    except Exception as exc:  # noqa: BLE001
        # Same degradation for a store that has not been embedded yet (no
        # vector index after a fresh restore): BM25 still answers.
        logger.warning("vector half unavailable, falling back to BM25 only: %s", exc)
        return []
    return [r["key"] for r in rows if r.get("key")]


_EXPAND_CYPHER = """
UNWIND $keys AS key
MATCH (c:Chunk {chunk_id: key})
MATCH (d:Document)-[:HAS_CHUNK]->(c)
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
OPTIONAL MATCH (c)-[:NEXT]->(nxt:Chunk)
OPTIONAL MATCH (d)-[:MENTIONS_ACT]->(a)
WITH key, c, d, prev, nxt, collect(DISTINCT a.key)[..6] AS mentions
OPTIONAL MATCH (d)-[:DEFINES_TERM|DEFINES]-(t:Term)
RETURN key,
       c.mongo_id          AS chunk_id,
       c.section_path      AS section_path,
       c.section_title     AS section_title,
       c.paragraph_numbers AS paragraphs,
       c.ordinal           AS ordinal,
       prev.mongo_id       AS prev_id,
       nxt.mongo_id        AS next_id,
       d.sha256            AS sha256,
       d.title             AS title,
       d.rel_path          AS rel_path,
       d.primary_collection     AS collection,
       d.review_state           AS review_state,
       d.legal_status_verified  AS legal_status_verified,
       d.identity_warning       AS identity_warning,
       mentions,
       collect(DISTINCT t.name)[..6] AS defines_terms
"""


def _chunk_texts(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Original text for these chunks. MongoDB, not Neo4j: the graph carries
    only the transliterated copy, and an answer has to quote the document as
    it is actually written."""
    if not ids:
        return {}
    cursor = mongo_db().chunks.find(
        {"_id": {"$in": ids}},
        {"text": 1, "section_path": 1, "section_title": 1, "char_count": 1},
    )
    return {doc["_id"]: doc for doc in cursor}


# A tool result is spent context: everything it returns crowds out the
# conversation and every later turn carries it. These caps are per call, and
# they are characters because that is what we can measure without a tokenizer
# -- roughly four per token for this corpus.
TEXT_CHARS = 1500
CONTEXT_CHARS = 300
BUDGET_CHARS = 14_000


def search_chunks(
    query: str,
    *,
    limit: int = 8,
    candidates: int = 50,
    neighbours: bool = True,
    budget_chars: int = BUDGET_CHARS,
) -> list[dict[str, Any]]:
    """Hybrid search returning passages with their place in the corpus."""
    query = (query or "").strip()
    if not query:
        return []

    bm25 = _bm25_chunks(query, candidates)
    vector = _vector_chunks(query, candidates)
    if not bm25 and not vector:
        return []

    fused = _rrf([bm25, vector])
    ordered = sorted(fused, key=lambda key: -fused[key])

    # Over-fetch: duplicates of the same document collapse below, and without
    # slack a result set of six PDF/DOCX copies of one order would come back
    # as a single hit.
    rows = read_query(_EXPAND_CYPHER, keys=ordered[: limit * 3])
    by_key = {r["key"]: r for r in rows}

    # Pick the final passages first, read their text second. The other order
    # pulled three times as much out of MongoDB as it returned, because most
    # candidates lose to a duplicate before anyone reads them.
    chosen: list[tuple[str, dict[str, Any]]] = []
    seen_sha: set[str] = set()
    for key in ordered[: limit * 3]:
        row = by_key.get(key)
        if row is None:
            continue
        # One document, one result. The corpus holds the same order as both a
        # PDF and a DOCX conversion (1 665 `SAME_STEM_DIFFERENT_BYTES` pairs),
        # and returning both spends the answer's budget saying one thing twice.
        sha = row.get("sha256")
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        chosen.append((key, row))
        if len(chosen) >= limit:
            break

    wanted: list[str] = []
    for _, row in chosen:
        wanted.append(row["chunk_id"])
        if neighbours:
            wanted.extend(i for i in (row.get("prev_id"), row.get("next_id")) if i)
    texts = _chunk_texts(wanted)

    results: list[dict[str, Any]] = []
    spent = 0
    for key, row in chosen:
        body = texts.get(row["chunk_id"], {})
        text = (body.get("text") or "")[:TEXT_CHARS]
        context = {}
        if neighbours:
            for side in ("prev", "next"):
                cid = row.get(f"{side}_id")
                if cid and cid in texts:
                    context[side] = (texts[cid].get("text") or "")[:CONTEXT_CHARS]

        cost = len(text) + sum(len(v) for v in context.values())
        if results and spent + cost > budget_chars:
            # Stop rather than truncate: a passage cut in half reads as if the
            # document says less than it does.
            break
        spent += cost

        results.append(
            {
                "score": round(fused[key], 5),
                "text": text,
                "context": context,
                "citation": {
                    "sha256": row.get("sha256"),
                    "title": row.get("title"),
                    "section": row.get("section_path") or row.get("section_title"),
                    "paragraphs": row.get("paragraphs"),
                    "rel_path": row.get("rel_path"),
                    "collection": row.get("collection"),
                },
                "flags": _flags(row),
                "graph": {
                    "mentions_acts": [m for m in (row.get("mentions") or []) if m],
                    "defines_terms": [t for t in (row.get("defines_terms") or []) if t],
                },
            }
        )
    return results


def _flags(row: dict[str, Any]) -> dict[str, Any]:
    """What the caller must not omit when quoting this passage.

    Every document in the corpus is `unreviewed` and none has a verified legal
    status, so an answer that leaves this out is claiming more than the data
    supports.
    """
    flags: dict[str, Any] = {
        "review_state": row.get("review_state") or "unreviewed",
        "legal_status_verified": bool(row.get("legal_status_verified")),
    }
    if row.get("identity_warning"):
        flags["identity_warning"] = row["identity_warning"]
    return flags


# ---------------------------------------------------------------------------
# Documents, and the ones with no text of their own
# ---------------------------------------------------------------------------
_DOCUMENT_CYPHER = """
MATCH (d:Document {sha256: $sha})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH d, count(c) AS chunks
OPTIONAL MATCH (d)-[:READABLE_SIBLING]->(r:Document)
RETURN d.sha256 AS sha256, d.title AS title, d.rel_path AS rel_path,
       d.primary_collection AS collection, d.review_state AS review_state,
       d.legal_status_verified AS legal_status_verified,
       d.identity_warning AS identity_warning,
       d.extraction_quality AS extraction_quality,
       d.l1_uri AS l1_uri,
       chunks,
       r.sha256 AS readable_sha256, r.title AS readable_title,
       r.rel_path AS readable_rel_path
"""

NO_TEXT_NOTE = (
    "Bu hujjatning matni mavjud emas (konversiya bo'sh natija bergan). "
    "O'qiladigan nusxasi ko'rsatilmoqda."
)

NO_TEXT_NOTE_UNRESOLVED = (
    "Bu hujjatning matni mavjud emas va o'qiladigan nusxasi topilmadi. "
    "Asl fayl bazaga yuklanmagan."
)


def get_document(sha256: str) -> Optional[dict[str, Any]]:
    """A document card, with the redirect a textless document needs.

    361 of the 1 185 documents converted to nothing, so they carry no chunk and
    no search can reach their contents. Every one of them has a
    `READABLE_SIBLING` pointing at a copy that did convert, so the honest
    answer is that copy plus a note saying why -- not an empty result, and not
    the sibling's text passed off as this document's.
    """
    rows = read_query(_DOCUMENT_CYPHER, sha=sha256)
    if not rows:
        return None
    row = rows[0]

    card: dict[str, Any] = {
        "sha256": row["sha256"],
        "title": row.get("title"),
        "rel_path": row.get("rel_path"),
        "collection": row.get("collection"),
        "chunk_count": row.get("chunks") or 0,
        "extraction_quality": row.get("extraction_quality"),
        "flags": _flags(row),
        "has_text": bool(row.get("chunks")),
    }

    if card["has_text"]:
        return card

    if row.get("readable_sha256"):
        card["note"] = NO_TEXT_NOTE
        card["read_instead"] = {
            "sha256": row["readable_sha256"],
            "title": row.get("readable_title"),
            "rel_path": row.get("readable_rel_path"),
        }
    else:
        card["note"] = NO_TEXT_NOTE_UNRESOLVED
        card["original_file"] = row.get("l1_uri")
    return card


_DOCUMENT_SEARCH_CYPHER = f"""
CALL db.index.fulltext.queryNodes('{DOCUMENT_FULLTEXT_INDEX}', $q, {{limit: $k}})
YIELD node AS d, score
RETURN d.sha256 AS sha256, score ORDER BY score DESC
"""


def search_documents(query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """Find documents by title and filename, each resolved through `get_document`
    so a textless hit arrives with its readable copy attached."""
    q = normalize_query(query)
    if not q:
        return []
    rows = read_query(_DOCUMENT_SEARCH_CYPHER, q=q, k=limit * 2)
    out: list[dict[str, Any]] = []
    for row in rows:
        card = get_document(row["sha256"])
        if card is None:
            continue
        card["score"] = round(row["score"], 4)
        out.append(card)
        if len(out) >= limit:
            break
    return out
