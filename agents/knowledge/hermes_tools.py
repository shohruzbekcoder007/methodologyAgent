"""The same four tools, registered the way the Hermes backend takes them.

`AIAgent` has no parameter for LangChain tools -- it builds its tool list from
its own registry, filtered by `enabled_toolsets`. So the LangChain wrappers in
`tools.py` reach only the `hermes_lite` fallback, and the real backend needs
this: the same functions registered into `tools.registry` under one toolset
name, `knowledge`, which then has to appear in `HERMES_ENABLED_TOOLSETS`.

Two adapters, one implementation. The logic stays in `retrieval.py` and
`graph.py`; both files here are thin enough that a change to either adapter
cannot change what a tool actually does.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from agents.knowledge import graph, retrieval

logger = logging.getLogger("knowledge.hermes_tools")

TOOLSET = "knowledge"

_lock = threading.Lock()
_registered = False


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _guard(name: str, fn: Callable[..., Any]) -> Callable[..., str]:
    """Turn an exception into a reported failure.

    A tool that raises ends the turn with a stack trace the user never asked
    for; a tool that returns its error lets the agent say the lookup failed
    and carry on, which is the answer a person would give.
    """

    def run(args: dict[str, Any], **_kw: Any) -> str:
        try:
            return _dump(fn(args or {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s failed: %s", name, exc, exc_info=True)
            return _dump({"error": f"{name} ishlamadi: {exc}"})

    return run


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _search_knowledge(args: dict[str, Any]) -> Any:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query bo'sh bo'lmasligi kerak"}
    limit = max(1, min(int(args.get("limit") or 5), 10))
    hits = retrieval.search_chunks(query, limit=limit)
    if not hits:
        return {"results": [], "note": "Bu so'rov bo'yicha hujjat topilmadi."}
    return {"results": hits, "count": len(hits)}


def _get_document(args: dict[str, Any]) -> Any:
    return graph.document(
        (args.get("sha256") or "").strip(),
        include_text=bool(args.get("include_text")),
        section=(args.get("section") or None),
        offset=max(0, int(args.get("offset") or 0)),
    )


def _find_related(args: dict[str, Any]) -> Any:
    direction = args.get("direction") or "both"
    return graph.find_related(
        sha256=(args.get("sha256") or None),
        key=(args.get("key") or None),
        relation=(args.get("relation") or None),
        direction=direction if direction in {"out", "in", "both"} else "both",
        limit=int(args.get("limit") or 20),
    )


def _count_documents(args: dict[str, Any]) -> Any:
    year = args.get("year")
    return graph.count_documents(
        collection=(args.get("collection") or None),
        year=int(year) if year not in (None, "") else None,
        kind=(args.get("kind") or None),
        folder=(str(args["folder"]) if args.get("folder") not in (None, "") else None),
        group_by=(args.get("group_by") or None),
    )


# ---------------------------------------------------------------------------
# Schemas -- the description is what the model reads to choose a tool, so each
# one says when to use it and, just as importantly, when not to.
# ---------------------------------------------------------------------------
SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge",
        "description": (
            "Milliy statistika qo'mitasi metodologik hujjatlaridan matn qidirish "
            "(so'z mosligi va ma'no bo'yicha birga).\n"
            "ISHLATING: ko'rsatkich qanday hisoblanishi, metodologik tartib, atama "
            "ta'rifi, hujjat mazmuni so'ralganda. Hujjatlarga taalluqli har qanday "
            "savolda birinchi navbatda shu tool.\n"
            "ISHLATMANG: salomlashish; javob suhbat tarixida yoki oldingi tool "
            "natijasida bo'lsa; aniq hujjatning to'liq matni kerak bo'lsa "
            "(get_document, include_text=true); sanoq savollarida (count_documents).\n"
            "Savolni foydalanuvchi bergan holicha, to'liq jumla sifatida uzating — "
            "kalit so'zlarga bo'lmang.\n"
            "Natijada matn, iqtibos (sha256, sarlavha, bo'lim) va flags bo'ladi. "
            "flags.review_state va flags.legal_status_verified ni javobda eslatib "
            "o'ting. citation.collection fayl qayerdan topilganini bildiradi, "
            "hujjatning kuchda ekanini EMAS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Foydalanuvchining savoli, to'liq jumla sifatida.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Nechta parcha qaytarilsin (1-10, standart 5).",
                },
            },
            "required": ["query"],
        },
        "handler": _search_knowledge,
        "emoji": "🔎",
    },
    {
        "name": "get_document",
        "description": (
            "Bitta hujjat: kartasi va (so'ralsa) matni.\n"
            "ISHLATING: qidiruv topgan hujjat haqida qo'shimcha ma'lumot kerak "
            "bo'lganda (include_text=false); hujjatning to'liq matni yoki bir bo'limi "
            "kerak bo'lganda (include_text=true); qidiruv bergan parcha yetarli "
            "bo'lmay davomi kerak bo'lganda.\n"
            "ISHLATMANG: qaysi hujjat kerakligi noma'lum bo'lsa — avval "
            "search_knowledge bilan toping; mavzu bo'yicha izlash uchun.\n"
            "sha256 ni qidiruv natijasidagi citation.sha256 dan oling, o'ylab topmang.\n"
            "section — bo'lim sarlavhasining bir qismi, masalan \"3-bob\"; faqat "
            "include_text=true bilan ishlaydi. has_more true bo'lsa, next_offset "
            "qiymati bilan qayta chaqiring.\n"
            "has_text false bo'lsa — hujjatning matni yo'q. Javobda note ni ayting va "
            "read_instead dagi o'qiladigan nusxaga yo'naltiring; o'sha nusxaning matnini "
            "asl hujjatniki sifatida ko'rsatmang."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sha256": {"type": "string", "description": "Hujjatning sha256 kaliti."},
                "include_text": {
                    "type": "boolean",
                    "description": "true = matni ham qaytarilsin (standart false).",
                },
                "section": {
                    "type": "string",
                    "description": "Bo'lim sarlavhasining bir qismi (include_text bilan).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Nechanchi parchadan boshlab o'qilsin (standart 0).",
                },
            },
            "required": ["sha256"],
        },
        "handler": _get_document,
        "emoji": "📄",
    },
    {
        "name": "find_related",
        "description": (
            "Hujjat yoki akt bilan bog'langan narsalarni topish (graf bo'ylab yurish).\n"
            "ISHLATING: \"nimaga tayanadi\", \"kim havola qiladi\", \"qaysi qonunga "
            "asoslangan\", \"boshqa nusxasi bormi\", \"asl fayli qaysi\" kabi "
            "bog'lanish savollarida.\n"
            "ISHLATMANG: hujjat mazmuni haqidagi savollarda (search_knowledge).\n"
            "IKKI QADAM: 1) avval relation'siz chaqiring — qanday bog'lanishlar borligi "
            "va nechtaligi qaytadi; 2) keyin kerakli relation nomi bilan qayta chaqiring.\n"
            "Ko'p uchraydiganlari: MENTIONS_ACT, MENTIONS_DOCUMENT, MENTIONS_LAW "
            "(nimaga havola qiladi), SAME_STEM_DIFFERENT_BYTES (boshqa nusxasi), "
            "READABLE_SIBLING (o'qiladigan nusxasi), HAS_SOURCE_FILE (asl fayllari), "
            "DEFINES_TERM (ta'riflangan atamalar).\n"
            "direction: \"in\" — kim bunga havola qiladi; \"out\" — bu nimaga havola "
            "qiladi; \"both\" — ikkalasi."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sha256": {"type": "string", "description": "Hujjat kaliti."},
                "key": {
                    "type": "string",
                    "description": "Akt yoki atama kaliti, masalan \"qaror:4796:2020-08-03\".",
                },
                "relation": {
                    "type": "string",
                    "description": "Bog'lanish turi. Bo'sh qoldirilsa — mavjudlari ro'yxati.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "description": "Bog'lanish yo'nalishi (standart both).",
                },
                "limit": {"type": "integer", "description": "Maksimal natija (standart 20)."},
            },
            "required": [],
        },
        "handler": _find_related,
        "emoji": "🔗",
    },
    {
        "name": "count_documents",
        "description": (
            "Hujjatlarni sanash: papka, yil, tur va to'plam bo'yicha.\n"
            "ISHLATING: \"nechta\", \"qancha\", \"qaysi yili eng ko'p\" kabi sanoq "
            "savollarida.\n"
            "ISHLATMANG: hujjat mazmuni kerak bo'lganda (search_knowledge).\n"
            "collection: amaldagi | bekor | docsarchive. year: 2010-2026. "
            "kind: buyruq | qaror | farmon | farmoyish | qonun | bayonnoma | unknown. "
            "folder: \"1\"-\"26\". group_by: collection | year | kind | folder.\n"
            "MUHIM: collection fayl qaysi papkadan topilganini bildiradi, hujjatning "
            "huquqiy holatini EMAS. \"bekor\" papkasidagi hujjat bekor qilingan degani "
            "emas — javobda shuni aytib o'ting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "amaldagi | bekor | docsarchive"},
                "year": {"type": "integer", "description": "2010-2026"},
                "kind": {"type": "string", "description": "buyruq | qaror | farmon | ..."},
                "folder": {"type": "string", "description": "Papka raqami, \"1\"-\"26\"."},
                "group_by": {
                    "type": "string",
                    "enum": ["collection", "year", "kind", "folder"],
                    "description": "Shu o'lchov bo'yicha guruhlab sanash.",
                },
            },
            "required": [],
        },
        "handler": _count_documents,
        "emoji": "🔢",
    },
]


def register_hermes_tools() -> list[str]:
    """Put the five tools into the Hermes registry. Idempotent.

    Must run before `AIAgent` is constructed: the agent reads the registry
    once while it builds its tool list, so anything registered afterwards is
    invisible for that agent's whole lifetime.
    """
    global _registered
    with _lock:
        if _registered:
            return [s["name"] for s in SCHEMAS]
        try:
            from tools.registry import registry  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hermes tool registry unavailable: %s", exc)
            return []

        names: list[str] = []
        for spec in SCHEMAS:
            schema = {k: spec[k] for k in ("name", "description", "parameters")}
            try:
                registry.register(
                    name=spec["name"],
                    toolset=TOOLSET,
                    schema=schema,
                    handler=_guard(spec["name"], spec["handler"]),
                    description=spec["description"].split("\n", 1)[0],
                    emoji=spec.get("emoji", ""),
                )
                names.append(spec["name"])
            except Exception as exc:  # noqa: BLE001
                logger.error("could not register %s: %s", spec["name"], exc)
        _registered = True
        logger.info("registered %d knowledge tools in toolset %r", len(names), TOOLSET)
        return names


def tool_names() -> list[str]:
    return [s["name"] for s in SCHEMAS]
