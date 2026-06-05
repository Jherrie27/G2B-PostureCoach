"""
app.py
Posture Correction Coach — Streamlit Dashboard
Run alongside main.py for the full UI experience.

    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time
import os
import json
import joblib
import numpy as np
import threading
import queue

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "Posture Coach",
    page_icon  = "🧍",
    layout     = "wide",
)

# ── SESSION STATE INIT ────────────────────────────────────────────
if "tracker"        not in st.session_state:
    from session import SessionTracker
    st.session_state.tracker = SessionTracker(
        soft_alert_sec = 120,
        hard_alert_sec = 600,
    )
if "llm_advice"     not in st.session_state: st.session_state.llm_advice    = ""
if "llm_loading"    not in st.session_state: st.session_state.llm_loading   = False
if "chat_history"   not in st.session_state: st.session_state.chat_history  = []
if "session_active" not in st.session_state: st.session_state.session_active= False
if "label"          not in st.session_state: st.session_state.label         = "correct"
if "confidence"     not in st.session_state: st.session_state.confidence    = 0.0

LABEL_COLORS = {
    "correct":      "#22C55E",
    "slouching":    "#EF4444",
    "neck_forward": "#EAB308",
    "leaning":      "#A855F7",
}

# ── HEADER ────────────────────────────────────────────────────────
st.title("🧍 Posture Correction Coach")
st.caption("Medusa Systems · CSS181 · Mapúa University")
st.divider()

# ── LAYOUT: 3 COLUMNS ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1.5, 1.3])

# ══════════════════════════════════════════════════════════════════
# COLUMN 1 — LIVE STATUS
# ══════════════════════════════════════════════════════════════════
with col1:
    st.subheader("📷 Live Status")

    stats = st.session_state.tracker.get_stats()
    label = stats["current_label"]
    color = LABEL_COLORS.get(label, "#888888")

    # posture badge
    st.markdown(
        f"""<div style="
            background:{color}22;
            border:2px solid {color};
            border-radius:12px;
            padding:16px;
            text-align:center;
            margin-bottom:12px;">
            <span style="font-size:2rem;font-weight:700;color:{color}">
                {label.upper().replace('_',' ')}
            </span>
        </div>""",
        unsafe_allow_html=True
    )

    # confidence
    st.caption(f"Confidence: {st.session_state.confidence:.0%}")

    # streak timer
    streak = stats["current_streak_sec"]
    streak_str = f"{int(streak)//60:02d}:{int(streak)%60:02d}"
    st.metric("Current Streak", streak_str,
              help="How long you've been in current bad posture")

    st.divider()

    # session metrics
    st.subheader("📊 Session Stats")

    session_min = stats["session_duration_sec"] / 60
    poor_min    = stats["total_poor_sec"] / 60
    poor_pct    = stats["poor_posture_pct"]

    m1, m2 = st.columns(2)
    m1.metric("Session", f"{session_min:.1f} min")
    m2.metric("Poor Posture", f"{poor_pct:.1f}%")

    m3, m4 = st.columns(2)
    m3.metric("Poor Time", f"{poor_min:.1f} min")
    m4.metric("Streaks", stats["num_streaks"])

    # posture bar
    if session_min > 0:
        good_pct = max(0, 100 - poor_pct)
        st.write("Session breakdown:")
        st.progress(good_pct / 100,
                    text=f"Good {good_pct:.0f}% · Poor {poor_pct:.0f}%")

    st.divider()

    # streak history
    if stats["streak_log"]:
        st.subheader("📋 Streak Log")
        log_data = [
            {
                "Posture": s["label"].replace("_", " "),
                "Duration": f"{s['duration_sec']:.0f}s"
            }
            for s in reversed(stats["streak_log"][-8:])
        ]
        st.dataframe(
            pd.DataFrame(log_data),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════
# COLUMN 2 — LLM COACHING PANEL
# ══════════════════════════════════════════════════════════════════
with col2:
    st.subheader("💬 Coaching Advice")

    if st.session_state.llm_loading:
        st.info("⏳ Getting personalized coaching advice...")
        with st.spinner("Consulting knowledge base..."):
            time.sleep(0.5)

    if st.session_state.llm_advice:
        st.success("Latest advice from your posture coach:")
        st.markdown(
            f"""<div style="
                background:var(--background-color);
                border:0.5px solid #333;
                border-radius:10px;
                padding:16px;
                font-size:14px;
                line-height:1.7;">
                {st.session_state.llm_advice.replace(chr(10), '<br>')}
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.info(
            "Coaching advice will appear here automatically when "
            "bad posture is detected for 10 minutes. "
            "You can also ask a question below."
        )

    st.divider()

    # ── CHATBOT ───────────────────────────────────────────────────
    st.subheader("🤖 Ask Your Coach")

    # chat history display
    for msg in st.session_state.chat_history[-6:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # input
    user_input = st.chat_input(
        "Ask about your posture, pain, exercises...")

    if user_input:
        # add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # get RAG response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from rag_query import answer_user_question
                stats    = st.session_state.tracker.get_stats()
                response = answer_user_question(
                    user_input,
                    stats["current_label"],
                    stats,
                )
            st.write(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

# ══════════════════════════════════════════════════════════════════
# COLUMN 3 — CONTROLS + SUMMARY
# ══════════════════════════════════════════════════════════════════
with col3:
    st.subheader("⚙️ Controls")

    # manual advice trigger
    if st.button("💡 Get Advice Now", use_container_width=True):
        stats = st.session_state.tracker.get_stats()
        label = stats["current_label"]
        if label == "correct":
            st.toast("Your posture looks good! Keep it up.")
        else:
            with st.spinner("Getting advice..."):
                from rag_query import get_coaching_advice
                streak_min = stats["current_streak_sec"] / 60
                total_min  = stats["total_poor_sec"] / 60
                advice = get_coaching_advice(label, streak_min, total_min)
                st.session_state.llm_advice = advice
            st.rerun()

    # session summary
    if st.button("📄 End Session + Summary",
                 use_container_width=True,
                 type="primary"):
        with st.spinner("Generating session summary..."):
            from rag_query import get_session_summary
            stats   = st.session_state.tracker.get_stats()
            summary = get_session_summary(stats)
        st.session_state.session_summary = summary
        st.rerun()

    # reset session
    if st.button("🔄 Reset Session", use_container_width=True):
        from session import SessionTracker
        st.session_state.tracker      = SessionTracker()
        st.session_state.llm_advice   = ""
        st.session_state.chat_history = []
        st.rerun()

    # ── GROQ API KEY INPUT ────────────────────────────────────────
    st.divider()
    st.subheader("🔑 Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("API key set for this session.")

    # ── POSTURE GUIDE ─────────────────────────────────────────────
    st.divider()
    st.subheader("📖 Posture Guide")
    guide = {
        "🟢 Correct":       "Ears over shoulders, neutral spine",
        "🔴 Slouching":     "Rounded back, hunched shoulders",
        "🟡 Neck Forward":  "Head pushed toward screen",
        "🟣 Leaning":       "Body tilted to one side",
    }
    for posture, desc in guide.items():
        st.markdown(f"**{posture}** — {desc}")

    # ── SESSION SUMMARY DISPLAY ───────────────────────────────────
    if hasattr(st.session_state, "session_summary"):
        st.divider()
        st.subheader("📋 Session Summary")
        st.markdown(st.session_state.session_summary)

# ── AUTO REFRESH ──────────────────────────────────────────────────
# refreshes every 2 seconds to update live stats
time.sleep(2)
st.rerun()
