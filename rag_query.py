"""Posture-aware RAG query (G2B v3). Replaces the previous generic flow.

The old LangChain-based flow is preserved in rag_query_v2_backup.py.
"""
import os
from groq import Groq
from dotenv import load_dotenv

from src.rag.retriever import PostureRetriever
from src.rag.prompt_builder import build_prompt
from src.rag.grounding_check import is_grounded, fallback_response
from src.state.posture_state import PostureState

# Load GROQ_API_KEY from a local .env (gitignored) or the environment.
load_dotenv()
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and put your Groq "
        "key in it, or set the GROQ_API_KEY environment variable."
    )
GROQ_CLIENT = Groq(api_key=_GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

# Module-level singleton so retriever (and its embedding model) loads once
_RETRIEVER: PostureRetriever = None


def get_retriever() -> PostureRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = PostureRetriever()
    return _RETRIEVER


def answer(user_question: str, state: PostureState) -> dict:
    """Returns dict with keys: text, retrieved, grounded."""
    retriever = get_retriever()
    # The classifier label is valid even when is_reliable=False (e.g. hips off
    # screen on a desk webcam): ears + shoulders are enough to tell lean /
    # slouch / neck-forward apart. So we pass the live posture through to the
    # retriever and prompt REGARDLESS of the reliability gate. This is what makes
    # the chatbot "see" the camera in real time.
    posture = state.posture_class if state else None

    # Expand query with current state for better retrieval.
    expanded = user_question
    if state and posture:
        conf_note = "" if state.is_reliable else " (partial view)"
        expanded = (f"{user_question} "
                    f"[posture: {state.posture_class}{conf_note}, "
                    f"issue: {state.primary_issue}]")

    retrieved = retriever.retrieve(expanded, current_posture=posture, top_k=5)
    messages = build_prompt(state, retrieved, user_question)

    resp = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=400,
    )
    text = resp.choices[0].message.content.strip()
    grounded = is_grounded(text, retrieved)
    # Only prepend the templated fallback when retrieval returned NOTHING at all.
    # Prepending it merely because the answer didn't echo reference jargon caused
    # the confusing two-part "I don't have references... but you're slouching"
    # replies. A grounded posture answer no longer gets the preamble.
    if not grounded and not retrieved:
        text = fallback_response(state) + "\n\n" + text
    return {"text": text, "retrieved": retrieved, "grounded": grounded}
