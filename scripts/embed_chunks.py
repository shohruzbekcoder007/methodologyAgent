"""
Embed the corpus and write the vectors onto the Neo4j nodes.

Driven by `scripts/embed_chunks.sh`; runs inside the app container, which is
where the Neo4j and MongoDB drivers and the credentials already are.

What it embeds, and why from MongoDB:

  Chunk  ->  `chunks.text`, the original, in the script the document was
             written in. Neo4j's `search_text` is a complete copy but
             transliterated, and transliteration flattens exactly what a
             multilingual model is good at reading -- abbreviations and proper
             names most of all.
  Term   ->  name + definition. Small (1 308 of them), and a definition
             question is better served by matching a definition than by
             matching a passage that happens to contain the word.

Resumable by construction: a node is pending when it has no vector, or one
from a different model. Interrupt it and run it again -- it picks up where it
stopped. Change the model and it re-embeds everything, because vectors from
two models cannot be compared and silently mixing them degrades every search
afterwards with no error anywhere.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.knowledge import store  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="[embed] %(message)s", stream=sys.stdout
)
# On the first run `embedding` exists on no node yet, so the driver warns that
# the property is unknown for every query that mentions it. It is expected and
# it buries the progress output.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logger = logging.getLogger("embed")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
class Target:
    """One kind of node to embed: how to list it, read it and write it back."""

    def __init__(
        self,
        name: str,
        label: str,
        key_property: str,
        mongo_collection: str,
        vector_index: str,
    ) -> None:
        self.name = name
        self.label = label
        self.key_property = key_property
        self.mongo_collection = mongo_collection
        self.vector_index = vector_index

    def pending(self, model_key: str, limit: int | None) -> list[dict[str, Any]]:
        cypher = (
            f"MATCH (n:{self.label}) "
            "WHERE n.embedding IS NULL OR n.embedding_model <> $model "
            f"RETURN n.{self.key_property} AS key, n.mongo_id AS mongo_id "
            f"ORDER BY n.{self.key_property}"
        )
        if limit:
            cypher += f" LIMIT {int(limit)}"
        return store.read_query(cypher, model=model_key)

    def total(self) -> int:
        rows = store.read_query(f"MATCH (n:{self.label}) RETURN count(n) AS n")
        return rows[0]["n"] if rows else 0

    def embedded(self, model_key: str) -> int:
        rows = store.read_query(
            f"MATCH (n:{self.label}) WHERE n.embedding_model = $model "
            "RETURN count(n) AS n",
            model=model_key,
        )
        return rows[0]["n"] if rows else 0

    def texts(self, mongo_ids: list[str]) -> dict[str, str]:
        raise NotImplementedError

    def write(self, rows: list[dict[str, Any]], model_key: str) -> None:
        # `db.create.setNodeVectorProperty` stores the vector in Neo4j's own
        # float32 array form. A plain SET would store a list of doubles, which
        # the vector index refuses.
        store.write_query(
            "UNWIND $rows AS row "
            f"MATCH (n:{self.label} {{{self.key_property}: row.key}}) "
            "CALL db.create.setNodeVectorProperty(n, 'embedding', row.vec) "
            "SET n.embedding_model = $model",
            rows=rows,
            model=model_key,
        )

    def create_index(self, dim: int) -> None:
        # The dimension cannot be a query parameter -- index options are read
        # at planning time -- so it is interpolated, having been forced to int.
        store.write_query(
            f"CREATE VECTOR INDEX {self.vector_index} IF NOT EXISTS "
            f"FOR (n:{self.label}) ON (n.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(dim)}, "
            "`vector.similarity_function`: 'cosine'}}"
        )


class ChunkTarget(Target):
    def __init__(self) -> None:
        super().__init__("chunks", "Chunk", "chunk_id", "chunks", "nmc_chunk_vec")

    def texts(self, mongo_ids: list[str]) -> dict[str, str]:
        cursor = store.mongo_db().chunks.find(
            {"_id": {"$in": mongo_ids}}, {"text": 1, "section_title": 1}
        )
        out: dict[str, str] = {}
        for doc in cursor:
            text = (doc.get("text") or "").strip()
            title = (doc.get("section_title") or "").strip()
            # The heading travels with the passage: a chunk from the middle of
            # a table says little on its own, and the section it sits under is
            # often the only thing naming the indicator being described.
            out[doc["_id"]] = f"{title}\n\n{text}".strip() if title else text
        return out


class TermTarget(Target):
    def __init__(self) -> None:
        super().__init__("terms", "Term", "key", "terms", "nmc_term_vec")

    def texts(self, mongo_ids: list[str]) -> dict[str, str]:
        cursor = store.mongo_db().terms.find(
            {"_id": {"$in": mongo_ids}}, {"name": 1, "definitions_text": 1}
        )
        out: dict[str, str] = {}
        for doc in cursor:
            name = (doc.get("name") or "").strip()
            definition = (doc.get("definitions_text") or "").strip()
            out[doc["_id"]] = f"{name}: {definition}".strip(": ").strip()
        return out


TARGETS = {"chunks": ChunkTarget(), "terms": TermTarget()}


# ---------------------------------------------------------------------------
def _batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_target(
    target: Target, *, model_key: str, dim: int, batch: int, max_chars: int,
    limit: int | None, force: bool,
) -> dict[str, int]:
    if force:
        logger.info("%s: clearing existing vectors (--force)", target.name)
        store.write_query(
            f"MATCH (n:{target.label}) WHERE n.embedding IS NOT NULL "
            "REMOVE n.embedding, n.embedding_model"
        )

    total = target.total()
    done_already = target.embedded(model_key)
    pending = target.pending(model_key, limit)
    # `pending` is what this run will touch; with --limit that is a slice of
    # the outstanding work, not all of it, so the two are reported separately.
    logger.info(
        "%s: %d nodes, %d already embedded, %d outstanding, %d in this run",
        target.name, total, done_already, total - done_already, len(pending),
    )
    if not pending:
        return {"total": total, "embedded": 0, "skipped": 0}

    embedded = skipped = 0
    started = time.time()
    for group in _batched(pending, batch):
        ids = [row["mongo_id"] for row in group if row.get("mongo_id")]
        bodies = target.texts(ids)

        payload: list[dict[str, Any]] = []
        for row in group:
            text = (bodies.get(row.get("mongo_id")) or "").strip()
            if not text:
                # Nothing to embed. Not an error: 361 documents converted to
                # nothing, and their emptiness is a fact about the corpus.
                skipped += 1
                continue
            payload.append({"key": row["key"], "text": text[:max_chars]})

        if not payload:
            continue
        vectors = store.embed_texts([p["text"] for p in payload], input_type="document")
        target.write(
            [{"key": p["key"], "vec": v} for p, v in zip(payload, vectors)], model_key
        )
        embedded += len(payload)

        elapsed = time.time() - started
        rate = embedded / elapsed if elapsed > 0 else 0
        remaining = len(pending) - embedded - skipped
        logger.info(
            "%s: %d/%d  %.0f/s  ~%s qoldi",
            target.name, embedded + skipped, len(pending), rate,
            f"{remaining / rate:.0f}s" if rate > 0 else "?",
        )

    logger.info("%s: creating vector index %s (dim=%d)", target.name, target.vector_index, dim)
    target.create_index(dim)
    return {"total": total, "embedded": embedded, "skipped": skipped}


def check(query: str) -> None:
    """Run one real search through the retrieval layer."""
    from agents.knowledge import retrieval

    print(f"\n--- so'rov: {query!r}")
    hits = retrieval.search_chunks(query, limit=3)
    if not hits:
        print("    natija yo'q")
        return
    for i, hit in enumerate(hits, 1):
        cit = hit["citation"]
        print(f"  {i}. score={hit['score']}  {cit['title']}")
        print(f"     bo'lim : {cit['section']}")
        print(f"     manba  : {cit['sha256'][:16]}…  ({cit['collection']})")
        print(f"     holat  : {hit['flags']['review_state']}, "
              f"huquqiy holat tasdiqlangan: {hit['flags']['legal_status_verified']}")
        if hit["graph"]["mentions_acts"]:
            print(f"     havola : {', '.join(hit['graph']['mentions_acts'][:3])}")
        print(f"     matn   : {hit['text'][:160].replace(chr(10), ' ')}…")


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed the corpus into Neo4j vectors")
    ap.add_argument("--only", choices=["all", "chunks", "terms"], default="all")
    ap.add_argument("--batch", type=int, default=0, help="0 = servisning batch_max")
    ap.add_argument("--limit", type=int, default=0, help="faqat N ta node (sinov)")
    ap.add_argument("--force", action="store_true", help="hammasini qayta embed qilish")
    ap.add_argument("--check", metavar="QUERY", help="embed qilmay, bitta qidiruv sinovi")
    args = ap.parse_args()

    if args.check:
        check(args.check)
        return 0

    identity = store.embedding_identity()
    model_key = identity["index_key"]
    dim = int(identity["dim"])
    batch = args.batch or int(identity["batch_max"])
    max_chars = int(identity["max_chars"])
    logger.info(
        "embedding service: model=%s dim=%d device=%s batch=%d max_chars=%d",
        identity.get("model"), dim, identity.get("device"), batch, max_chars,
    )
    logger.info("index_key=%s", model_key)

    names = ["chunks", "terms"] if args.only == "all" else [args.only]
    summary = {}
    started = time.time()
    for name in names:
        summary[name] = run_target(
            TARGETS[name],
            model_key=model_key, dim=dim, batch=batch, max_chars=max_chars,
            limit=args.limit or None, force=args.force,
        )

    logger.info("-" * 58)
    for name, stat in summary.items():
        logger.info(
            "%s: %d ta embed qilindi, %d ta matnsiz o'tkazib yuborildi (jami %d)",
            name, stat["embedded"], stat["skipped"], stat["total"],
        )
    logger.info("jami vaqt: %.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        store.close()
