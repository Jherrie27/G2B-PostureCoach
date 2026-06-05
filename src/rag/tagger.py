"""Tag KB chunks with metadata using Groq for one-time bulk tagging.

NOTE: In this project we did NOT need to run this — the existing index already
carries a `posture_class` tag per chunk, so src/rag/export_chunks.py reuses those
(plus local heuristics) and avoids ~2842 Groq calls. This module is kept for the
case where richer LLM tags are wanted later.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict
from groq import Groq

CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODEL = "llama-3.1-8b-instant"

TAGGING_PROMPT = """You are tagging physiotherapy textbook chunks for a posture
correction system. Output ONLY valid JSON.

The system classifies posture as one of:
- correct_posture: aligned head, shoulders, hips, upright torso
- slouching: rounded upper back, shoulders rolled forward, spine flexion
- neck_forward: head protrudes forward (turtle neck), torso mostly upright
- lean: lateral tilt, left/right asymmetry

For the chunk below, output JSON with these fields:

{{
  "applicable_postures": ["slouching", ...],   // list, at least 1 entry, can be ["correct_posture"] if it's about ideal posture
  "anatomical_region": "cervical_spine" | "thoracic_spine" | "lumbar_spine" | "shoulder" | "general",
  "content_type": "definition" | "cause" | "consequence" | "correction" | "exercise" | "background",
  "key_terms": ["..."]                          // 3-6 important terms from the chunk
}}

CHUNK:
\"\"\"
{chunk}
\"\"\"

Output only the JSON, no prose:"""


def tag_chunk(text: str, retries: int = 3) -> Dict:
    for attempt in range(retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": TAGGING_PROMPT.format(chunk=text[:2000])}],
                temperature=0.1,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            data = json.loads(raw)
            assert "applicable_postures" in data
            return data
        except Exception as e:
            print(f"  tagging attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return {
        "applicable_postures": ["correct_posture", "slouching",
                                "neck_forward", "lean"],
        "anatomical_region": "general",
        "content_type": "background",
        "key_terms": [],
    }


def tag_corpus(chunks_jsonl_in: str, out_jsonl: str):
    """Each input line should be {'id': str, 'text': str}."""
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(chunks_jsonl_in) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            tags = tag_chunk(obj["text"])
            obj["metadata"] = tags
            fout.write(json.dumps(obj) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"  tagged {n} chunks")
    print(f"DONE - tagged {n} chunks -> {out_jsonl}")


if __name__ == "__main__":
    tag_corpus(chunks_jsonl_in="data/kb_chunks_raw.jsonl",
               out_jsonl="data/kb_chunks_tagged.jsonl")
