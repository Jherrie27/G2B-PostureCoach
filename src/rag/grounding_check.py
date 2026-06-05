"""Quick check: does the response reference at least N key terms from retrieved chunks?"""
import re
from typing import List, Dict


def _extract_terms(text: str) -> set:
    """Coarse: nouns/jargon = words 5+ chars, not stop words, lowercased."""
    STOP = {"about", "above", "after", "again", "against", "because",
            "before", "below", "between", "during", "should", "would", "could",
            "their", "there", "these", "those", "where", "which", "while"}
    words = re.findall(r"[a-zA-Z]{5,}", text.lower())
    return {w for w in words if w not in STOP}


def is_grounded(response: str, retrieved: List[Dict],
                min_overlap: int = 2) -> bool:
    chunk_terms = set()
    for r in retrieved:
        chunk_terms |= _extract_terms(r["text"])
    response_terms = _extract_terms(response)
    return len(chunk_terms & response_terms) >= min_overlap


def fallback_response(state) -> str:
    """When the response fails grounding, return a templated state-only answer."""
    if state.primary_issue == "none":
        return ("Your posture currently looks well aligned. Keep your head "
                "stacked over your shoulders and your shoulders over your hips.")
    issue = state.primary_issue.replace("_", " ")
    return (f"I don't have detailed references for your specific question, but "
            f"your current measurements show {issue} as the main deviation. "
            f"Try small adjustments and watch the indicator above.")
