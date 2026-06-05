"""Export KB chunks from the existing ChromaDB (rag_db_v2/rag_db) to JSONL.

The old index already carries a `posture_class` tag per chunk, so we reuse it
instead of paying for 2842 Groq tagging calls (see src/rag/tagger.py for the
LLM-tagging path if richer tags are ever needed).

Outputs:
  data/kb_chunks_raw.jsonl     -> {"id", "text"}
  data/kb_chunks_tagged.jsonl  -> {"id", "text", "metadata": {...guide schema...}}

metadata schema (matches guide §6.1):
  applicable_postures: list from {correct_posture, slouching, neck_forward, lean}
  anatomical_region:   cervical_spine | thoracic_spine | lumbar_spine | shoulder | general
  content_type:        definition | cause | consequence | correction | exercise | background
  key_terms:           3-6 salient terms
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

SRC_DB = "rag_db_v2/rag_db/chroma.sqlite3"
ALL_POSTURES = ["correct_posture", "slouching", "neck_forward", "lean"]

# Map the legacy posture_class tag to the canonical applicable_postures list.
_POSTURE_MAP = {
    "correct": ["correct_posture"],
    "correct_posture": ["correct_posture"],
    "slouching": ["slouching"],
    "neck_forward": ["neck_forward"],
    "leaning": ["lean"],
    "lean": ["lean"],
    # topical buckets apply to every posture
    "general": list(ALL_POSTURES),
    "exercise": list(ALL_POSTURES),
    "breaks": list(ALL_POSTURES),
}

_STOP = {"about", "above", "after", "again", "against", "because", "before",
         "below", "between", "during", "should", "would", "could", "their",
         "there", "these", "those", "where", "which", "while", "posture",
         "muscle", "muscles", "body"}


def _applicable_postures(posture_class: str) -> List[str]:
    return _POSTURE_MAP.get((posture_class or "").strip().lower(), list(ALL_POSTURES))


def _anatomical_region(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("cervical", "neck", "head forward", "craniovertebral", "chin")):
        return "cervical_spine"
    if any(k in t for k in ("thoracic", "upper back", "kyphosis", "scapula", "scapular", "rounded")):
        return "thoracic_spine"
    if any(k in t for k in ("lumbar", "lower back", "low back", "pelvis", "pelvic")):
        return "lumbar_spine"
    if "shoulder" in t:
        return "shoulder"
    return "general"


def _content_type(text: str, posture_class: str) -> str:
    t = text.lower()
    pc = (posture_class or "").lower()
    if pc == "exercise" or any(k in t for k in ("stretch", "exercise", "repetition", "hold for", "reps")):
        return "exercise"
    if pc == "breaks":
        return "background"
    if any(k in t for k in ("is defined", "refers to", "definition", "is a condition", "is the")):
        return "definition"
    if any(k in t for k in ("caused by", "due to", "results from", "because of")):
        return "cause"
    if any(k in t for k in ("can lead to", "may cause", "increases the risk", "consequence", "results in pain")):
        return "consequence"
    if any(k in t for k in ("correct", "improve", "adjust", "to fix", "realign", "avoid")):
        return "correction"
    return "background"


def _key_terms(text: str, n: int = 5) -> List[str]:
    words = re.findall(r"[a-zA-Z]{5,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w in _STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return out


def _load_chunks(db_path: str) -> List[Dict]:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT id, key, string_value, int_value FROM embedding_metadata"
    ).fetchall()
    by_id: Dict[int, Dict] = {}
    for _id, key, sval, ival in rows:
        d = by_id.setdefault(_id, {})
        d[key] = sval if sval is not None else ival
    chunks = []
    for _id, d in by_id.items():
        text = d.get("chroma:document")
        if not text or not str(text).strip():
            continue
        chunks.append({
            "id": f"chunk_{_id:05d}",
            "text": str(text),
            "posture_class": d.get("posture_class", "general"),
            "page_num": d.get("page_num"),
            "source": d.get("source"),
        })
    con.close()
    return chunks


def export(db_path: str = SRC_DB,
           raw_out: str = "data/kb_chunks_raw.jsonl",
           tagged_out: str = "data/kb_chunks_tagged.jsonl") -> int:
    Path(raw_out).parent.mkdir(parents=True, exist_ok=True)
    chunks = _load_chunks(db_path)
    with open(raw_out, "w", encoding="utf-8") as fr, \
         open(tagged_out, "w", encoding="utf-8") as ft:
        for c in chunks:
            fr.write(json.dumps({"id": c["id"], "text": c["text"]}) + "\n")
            md = {
                "applicable_postures": _applicable_postures(c["posture_class"]),
                "anatomical_region": _anatomical_region(c["text"]),
                "content_type": _content_type(c["text"], c["posture_class"]),
                "key_terms": _key_terms(c["text"]),
                "legacy_posture_class": c["posture_class"],
                "source": c.get("source") or "",
                "page_num": c.get("page_num") if c.get("page_num") is not None else -1,
            }
            ft.write(json.dumps({"id": c["id"], "text": c["text"],
                                 "metadata": md}) + "\n")
    print(f"Exported {len(chunks)} chunks -> {raw_out} and {tagged_out}")
    return len(chunks)


if __name__ == "__main__":
    export()
