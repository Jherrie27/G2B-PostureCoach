"""
rag_query.py
Posture Correction Coach — RAG + LLM Integration
Retrieves relevant chunks from ChromaDB and calls Groq API.
"""

import os
import time
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── CONFIG ────────────────────────────────────────────────────────
DB_PATH      = "./rag_db_v2"
EMBED_MODEL  = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # set via .env / environment
ALERT_MODEL  = "llama-3.1-8b-instant"    # fast — for real-time alerts
SUMMARY_MODEL= "llama-3.3-70b-versatile" # quality — for session summary

# ── LOAD DB ONCE AT STARTUP ───────────────────────────────────────
print("Loading RAG knowledge base...")
_embeddings = HuggingFaceEmbeddings(
    model_name    = EMBED_MODEL,
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
)
_db = Chroma(
    persist_directory  = DB_PATH,
    embedding_function = _embeddings,
)
_client = Groq(api_key=GROQ_API_KEY)
print("RAG knowledge base loaded.")


# ── RETRIEVAL ─────────────────────────────────────────────────────
def retrieve_context(posture_label: str, k: int = 5) -> str:
    """
    Retrieves the top-k relevant chunks from ChromaDB
    filtered by posture class.
    """
    query = f"{posture_label} posture correction exercises advice"

    # try with filter first
    try:
        results = _db.similarity_search(
            query,
            k      = k,
            filter = {"posture_class": posture_label},
        )
    except Exception:
        results = []

    # fallback without filter if no results
    if not results:
        results = _db.similarity_search(query, k=k)

    if not results:
        return ""

    return "\n\n".join(r.page_content for r in results)


# ── ALERT COACHING CALL ───────────────────────────────────────────
def get_coaching_advice(
    posture_label:     str,
    streak_minutes:    float,
    session_total_min: float,
) -> str:
    """
    Called when hard alert fires (10 min bad posture streak).
    Returns coaching advice string from Llama 3.1-8B via Groq.
    """
    context = retrieve_context(posture_label)

    if not context:
        return (
            f"You have been in a {posture_label.replace('_',' ')} posture "
            f"for {streak_minutes:.0f} minutes. "
            "Please sit upright and take a short break."
        )

    prompt = f"""You are a posture correction coach assistant for a remote worker.
Use ONLY the context below from a physiotherapy textbook to give advice.
Be concise, friendly, and practical. Use plain language.

CONTEXT:
{context}

USER SESSION DATA:
- Detected posture: {posture_label.replace('_', ' ')}
- Current streak:   {streak_minutes:.1f} minutes
- Total poor posture today: {session_total_min:.1f} minutes

Based on the context and session data, provide:
1. What this posture does to the body (1-2 sentences)
2. Three immediate correction steps (numbered)
3. Two stretches to do right now
4. One long-term setup tip

Keep the total response under 200 words."""

    try:
        response = _client.chat.completions.create(
            model      = ALERT_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 350,
            temperature= 0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"You have been {posture_label.replace('_',' ')} for "
            f"{streak_minutes:.0f} minutes. Please correct your posture. "
            f"(LLM error: {str(e)})"
        )


# ── CHATBOT Q&A CALL ──────────────────────────────────────────────
def answer_user_question(
    question:      str,
    posture_label: str,
    stats:         dict,
) -> str:
    """
    Called when user types a question in the chatbot panel.
    Uses the same RAG + Groq pipeline.
    """
    context = retrieve_context(posture_label, k=5)

    session_summary = (
        f"Session duration: {stats.get('session_duration_sec', 0)/60:.1f} min | "
        f"Poor posture: {stats.get('poor_posture_pct', 0):.1f}% | "
        f"Current: {posture_label.replace('_', ' ')}"
    )

    prompt = f"""You are a posture correction coach assistant.
Use ONLY the context below from a physiotherapy textbook to answer the question.
If the answer is not in the context, say so clearly.
Be concise and practical. Keep response under 150 words.

CONTEXT:
{context}

USER SESSION: {session_summary}

USER QUESTION: {question}

Answer:"""

    try:
        response = _client.chat.completions.create(
            model      = ALERT_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 250,
            temperature= 0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, could not get a response. ({str(e)})"


# ── END OF SESSION SUMMARY ────────────────────────────────────────
def get_session_summary(stats: dict) -> str:
    """
    Called when user ends the session.
    Generates a full session report using the 70B model.
    """
    streak_log = stats.get("streak_log", [])
    streak_text = ""
    if streak_log:
        streak_text = "\n".join(
            f"  - {s['label'].replace('_',' ')}: {s['duration_sec']:.0f} sec"
            for s in streak_log[-5:]   # last 5 streaks
        )

    # get general advice for all posture types seen
    labels_seen = list(set(s["label"] for s in streak_log))
    context     = retrieve_context(
        labels_seen[0] if labels_seen else "correct", k=4)

    prompt = f"""You are a posture correction coach.
Generate a friendly end-of-session report for a remote worker.
Use the context below for advice. Keep it under 250 words.

CONTEXT:
{context}

SESSION DATA:
- Total duration:       {stats.get('session_duration_sec',0)/60:.1f} minutes
- Poor posture time:    {stats.get('total_poor_sec',0)/60:.1f} minutes ({stats.get('poor_posture_pct',0):.1f}%)
- Number of streaks:    {stats.get('num_streaks',0)}
- Recent bad streaks:
{streak_text if streak_text else '  None — great session!'}

Write:
1. A one-sentence summary of how well they did
2. Their biggest posture issue today
3. Three specific action items for tomorrow
4. An encouraging closing line"""

    try:
        response = _client.chat.completions.create(
            model      = SUMMARY_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 400,
            temperature= 0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"Session complete. You spent {stats.get('poor_posture_pct',0):.1f}% "
            f"of your session in poor posture. "
            f"Keep working on it! (LLM error: {str(e)})"
        )
