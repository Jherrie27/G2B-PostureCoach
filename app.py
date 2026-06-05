"""G2B Posture Correction Coach — live Streamlit app.

Previous (v2) Streamlit UI is preserved in app_v2_backup.py.
"""
import time
import streamlit as st

from src.workers.shared_state import SharedPostureState
from src.workers.cv_worker import CVWorker
from src.workers.chat_worker import ChatWorker
from src.utils.config import mediapipe_complexity, target_fps, camera_resolution

st.set_page_config(page_title="G2B Posture Coach", layout="wide")

st.markdown("# G2B Posture Correction Coach")
st.caption("Live posture analysis + AI coaching — built for Raspberry Pi 5")

# === Init singletons (survive across reruns) ===
if "shared" not in st.session_state:
    w, h = camera_resolution()
    st.session_state.shared = SharedPostureState()
    st.session_state.cv_worker = CVWorker(
        st.session_state.shared,
        width=w, height=h,
        model_complexity=mediapipe_complexity(),
        target_fps=target_fps(),
    )
    st.session_state.cv_worker.start()
    st.session_state.chat_worker = ChatWorker(st.session_state.shared)
    st.session_state.chat_history = []

# === Session summary bar ===
summary_slot = st.empty()

# === Layout: two columns ===
left, right = st.columns([1, 1])

with left:
    st.subheader("Live view")
    frame_slot = st.empty()
    state_slot = st.empty()
    metrics_slot = st.empty()

with right:
    st.subheader("Chat with your coach")
    chat_area = st.container()
    user_input = st.chat_input("Ask about your posture...")

    with chat_area:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with chat_area:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.chat_worker.respond(user_input)
                st.write(result["text"])
                st.session_state.chat_history.append(("assistant", result["text"]))

# === Continuously refresh the live view (5 Hz is plenty for UI) ===
LABEL_EMOJI = {
    "correct_posture": "🟢",
    "slouching":       "🟠",
    "neck_forward":    "🟡",
    "lean":            "🟣",
}
for _ in range(20):  # ~4 seconds of refresh, then st.rerun()
    frame = st.session_state.cv_worker.latest_frame()
    state = st.session_state.shared.snapshot()
    if frame is not None:
        frame_slot.image(frame, channels="BGR", use_column_width=True)
    if state is not None:
        emoji = LABEL_EMOJI.get(state.posture_class, "")
        state_slot.markdown(
            f"### {emoji} **{state.posture_class.replace('_', ' ').title()}**  "
            f"`conf={state.confidence:.2f}`  "
            f"_(holding for {state.posture_duration_sec:.0f}s)_"
        )
        cols = metrics_slot.columns(3)
        cols[0].metric("Forward head", f"{state.ear_shoulder_offset_x:+.2f}",
                       delta=f"{state.feature_deviations['forward_head']:.2f}")
        cols[1].metric("Shoulder roll", f"{state.shoulder_roll_z:+.2f}",
                       delta=f"{state.feature_deviations['shoulder_roll']:.2f}")
        cols[2].metric("Tilt", f"{state.shoulder_tilt_angle:+.1f}°",
                       delta=f"{state.feature_deviations['shoulder_tilt']:.1f}")

        scols = summary_slot.columns(4)
        scols[0].metric("Session", f"{state.session_duration_sec/60:.1f} min")
        scols[1].metric("Correct posture",
                        f"{state.posture_distribution.get('correct_posture', 0)*100:.0f}%")
        scols[2].metric("Corrections", state.correction_events)
        scols[3].metric("Worst streak", f"{state.longest_bad_posture_streak_sec:.0f}s")
    time.sleep(0.2)

st.rerun()
