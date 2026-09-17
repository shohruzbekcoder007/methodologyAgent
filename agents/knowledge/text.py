"""Cyrillic -> Latin transliteration, matching what the ingest wrote.

Neo4j keeps only `search_text` on a Chunk: the corpus text transliterated to
Latin with apostrophes unified. The full-text index runs on that, so a query
has to go through the same transform or a Cyrillic query will not match a
Cyrillic document. Mirrors `latin()` in the ingest's
`scripts/export_neo4j_cypher.py` -- keep the two in step.

Vector search does not need this: the embedding model reads both scripts, and
it reads the original text, which is why `scripts/embed_chunks.py` embeds
`chunks.text` from MongoDB rather than `search_text` from here.
"""

from __future__ import annotations

_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": "'", "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ў": "o'", "қ": "q", "ғ": "g'",
    "ҳ": "h", "і": "i", "є": "ye",
}

_TRANS: dict[int, str] = {}
for _c, _l in _CYR.items():
    _TRANS[ord(_c)] = _l
    _TRANS[ord(_c.upper())] = (_l[:1].upper() + _l[1:]) if _l else ""
for _ap in "‘’ʻʼ`´ʹ′":
    _TRANS[ord(_ap)] = "'"


def latin(text: str) -> str:
    """Transliterate to Latin and unify apostrophes."""
    return (text or "").translate(_TRANS)


# Lucene syntax characters. A user's question is prose, not a query language,
# so a stray `-` or `:` in "PQ-4796" or "14:00" would otherwise be parsed as an
# operator and change or break the search.
_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def normalize_query(q: str) -> str:
    """Prepare a user's question for the full-text index."""
    text = latin(q)
    return "".join(("\\" + ch) if ch in _LUCENE_SPECIAL else ch for ch in text).strip()
