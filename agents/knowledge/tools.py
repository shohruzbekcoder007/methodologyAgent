"""The four tools the host agent gets.

Each docstring is a prompt: it is what the model reads to decide whether to
call the tool, so it says when to use it *and when not to*. The negative half
is what stops a 27B model reaching for search when the answer is already in
front of it.

Four, not eleven. The agent already carries the framework's own memory,
session-search and todo tools, and every extra name makes the choice harder.
The coverage that eleven would buy is here instead in one argument:
`find_related` takes any of the catalogue's 77 relation types rather than
giving each its own tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.tools import tool

from agents.knowledge import graph, retrieval

logger = logging.getLogger("knowledge.tools")


def _dump(payload: Any) -> str:
    """Tool results go back as JSON text: it survives the round trip through
    the model unchanged, and the model can quote fields out of it verbatim."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _fail(tool_name: str, exc: Exception) -> str:
    # A tool that raises ends the turn; a tool that reports its failure lets
    # the agent say the search did not work, which is the honest answer.
    logger.error("%s failed: %s", tool_name, exc, exc_info=True)
    return _dump({"error": f"{tool_name} ishlamadi: {exc}"})


@tool
def search_knowledge(query: str, limit: int = 5) -> str:
    """Milliy statistika qo'mitasi metodologik hujjatlaridan matn qidirish.

    QACHON ISHLATILADI: ko'rsatkich qanday hisoblanishi, metodologik tartib,
    atama ta'rifi, hujjat mazmuni so'ralganda. Foydalanuvchining savoli
    hujjatlarga taalluqli bo'lsa — birinchi navbatda shu tool.

    QACHON ISHLATILMAYDI: salomlashish va suhbat haqidagi savollar; javob
    allaqachon suhbat tarixida yoki oldingi tool natijasida bo'lsa; aniq
    hujjatning to'liq matni kerak bo'lsa (get_document, include_text=true);
    "nechta" degan sanash savollari (count_documents ishlating).

    Savolni foydalanuvchi bergan holicha, to'liq jumla sifatida uzating —
    kalit so'zlarga bo'lib tashlamang. Qidiruv ham so'z mosligi, ham ma'no
    bo'yicha ishlaydi, shuning uchun to'liq jumla aniqroq natija beradi.

    Har bir natijada: matn, oldingi/keyingi parcha, iqtibos (sha256, sarlavha,
    bo'lim yo'li, papka) va flags. `flags.review_state` va
    `flags.legal_status_verified` ni javobda albatta eslatib o'ting.
    `citation.collection` faylning qayerdan topilganini bildiradi, hujjatning
    kuchda ekanini EMAS.
    """
    try:
        hits = retrieval.search_chunks(query, limit=max(1, min(int(limit), 10)))
        if not hits:
            return _dump({"results": [], "note": "Bu so'rov bo'yicha hujjat topilmadi."})
        return _dump({"results": hits, "count": len(hits)})
    except Exception as exc:  # noqa: BLE001
        return _fail("search_knowledge", exc)


@tool
def get_document(
    sha256: str,
    include_text: bool = False,
    section: Optional[str] = None,
    offset: int = 0,
) -> str:
    """Bitta hujjat: kartasi va (so'ralsa) matni.

    QACHON ISHLATILADI: qidiruv topgan hujjat haqida qo'shimcha ma'lumot kerak
    bo'lganda (`include_text=false`); hujjatning to'liq matni yoki bir bo'limi
    kerak bo'lganda (`include_text=true`); qidiruv bergan parcha yetarli
    bo'lmay davomi kerak bo'lganda.

    QACHON ISHLATILMAYDI: qaysi hujjat kerakligi hali noma'lum bo'lsa — avval
    search_knowledge bilan toping; mavzu bo'yicha izlash uchun.

    `sha256` ni qidiruv natijasining `citation.sha256` maydonidan oling —
    o'zingiz o'ylab topmang.

    `section` — bo'lim sarlavhasining bir qismi, masalan "3-bob" yoki
    "Yakuniy qoidalar"; faqat `include_text=true` bilan ishlaydi. Javobda
    `has_more` true bo'lsa, davomini o'qish uchun `next_offset` qiymati bilan
    qayta chaqiring.

    Agar `has_text` false bo'lsa, hujjatning matni mavjud emas. Bunda javobda
    `note` ni aytib o'ting va `read_instead` dagi o'qiladigan nusxaga
    yo'naltiring. O'sha nusxaning matnini asl hujjatning matni sifatida
    ko'rsatmang.
    """
    try:
        return _dump(
            graph.document(
                (sha256 or "").strip(),
                include_text=bool(include_text),
                section=(section or None),
                offset=max(0, int(offset or 0)),
            )
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("get_document", exc)


@tool
def find_related(
    sha256: Optional[str] = None,
    key: Optional[str] = None,
    relation: Optional[str] = None,
    direction: str = "both",
    limit: int = 20,
) -> str:
    """Hujjat yoki akt bilan bog'langan narsalarni topish (graf bo'ylab yurish).

    QACHON ISHLATILADI: "bu hujjat nimaga tayanadi", "unga kim havola qiladi",
    "qaysi qonunga asoslangan", "bu hujjatning boshqa nusxasi bormi", "qaysi
    faylardan olingan" kabi bog'lanish savollari.

    QACHON ISHLATILMAYDI: hujjat mazmuni haqidagi savollar uchun
    (search_knowledge).

    IKKI QADAMDA ISHLATING:
      1. Avval `relation` siz chaqiring — shu tugunda qanday bog'lanishlar
         borligi va nechtaligi qaytadi.
      2. Keyin kerakli `relation` nomi bilan qayta chaqiring.

    Ko'p uchraydigan bog'lanishlar:
      MENTIONS_ACT, MENTIONS_DOCUMENT, MENTIONS_LAW  — nimaga havola qiladi
      SAME_STEM_DIFFERENT_BYTES                      — o'sha hujjatning boshqa nusxasi
      READABLE_SIBLING                               — o'qiladigan nusxasi
      HAS_SOURCE_FILE                                — asl fayllari
      DEFINES_TERM                                   — shu hujjatda ta'riflangan atamalar

    `direction`: "out" (bu hujjatdan chiquvchi), "in" (bunga kiruvchi, ya'ni
    kim unga havola qiladi), "both". Kim havola qilishini bilish uchun "in".

    `sha256` — hujjat uchun; `key` — akt yoki atama uchun (masalan
    "qaror:4796:2020-08-03").
    """
    try:
        return _dump(
            graph.find_related(
                sha256=(sha256 or None),
                key=(key or None),
                relation=(relation or None),
                direction=direction if direction in {"out", "in", "both"} else "both",
                limit=limit,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("find_related", exc)


@tool
def count_documents(
    collection: Optional[str] = None,
    year: Optional[int] = None,
    kind: Optional[str] = None,
    folder: Optional[str] = None,
    group_by: Optional[str] = None,
) -> str:
    """Hujjatlarni sanash: papka, yil, tur va to'plam bo'yicha.

    QACHON ISHLATILADI: "nechta", "qancha", "qaysi yili eng ko'p", "5-papkada
    2024 yilda nechta qaror bor" kabi sanoq savollari.

    QACHON ISHLATILMAYDI: hujjat mazmuni yoki ro'yxati kerak bo'lganda —
    qidiruv boshqa tool.

    Filtrlar (bo'sh qoldirsa hisobga olinmaydi):
      collection — amaldagi | bekor | docsarchive
      year       — 2010..2026
      kind       — buyruq | qaror | farmon | farmoyish | qonun | bayonnoma | unknown
      folder     — papka raqami, "1" dan "26" gacha
      group_by   — collection | year | kind | folder

    MUHIM: `collection` fayl qaysi papkadan topilganini bildiradi, hujjatning
    huquqiy holatini EMAS. "bekor" papkasidagi hujjat bekor qilingan degani
    emas. Javobda buni aynan shunday tushuntiring.
    """
    try:
        return _dump(
            graph.count_documents(
                collection=collection or None,
                year=year,
                kind=kind or None,
                folder=folder or None,
                group_by=group_by or None,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("count_documents", exc)


def as_langchain_tools() -> list[Any]:
    """Everything this package contributes to the host agent."""
    return [
        search_knowledge,
        get_document,
        find_related,
        count_documents,
    ]
