"""Builds the LLM prompt from PostureState + retrieved chunks + user query."""
from typing import List, Dict
from src.state.posture_state import PostureState, NORMAL_THRESHOLDS

SYSTEM = """You are a posture correction coach. You give specific, evidence-based
guidance grounded ONLY in the retrieved physiotherapy references below.

RULES:
- Refer to the user's actual measured values from the OBSERVATION block.
- Quote or paraphrase from the REFERENCES for any physiological or corrective claim.
- If the references do not cover the user's question, say so plainly.
- Keep responses to 3-6 sentences unless the user asks for detail.
- Use friendly, encouraging language. Don't be alarmist."""


def _postures_str(md: Dict) -> str:
    ap = md.get("applicable_postures", [])
    return ", ".join(ap) if isinstance(ap, list) else str(ap)


def build_prompt(state: PostureState, retrieved: List[Dict],
                 user_question: str) -> List[Dict]:
    if not state.is_reliable:
        observation = ("The camera cannot reliably see the user's posture right now. "
                       "Ask the user to reposition before answering.")
    else:
        observation = _format_observation(state)

    refs = "\n\n".join(
        f"[REF {i+1}] (postures: {_postures_str(r['metadata'])}, "
        f"region: {r['metadata'].get('anatomical_region')})\n{r['text']}"
        for i, r in enumerate(retrieved)
    ) or "[No relevant references retrieved]"

    user_block = f"""=== CURRENT POSTURE OBSERVATION ===
{observation}

=== RETRIEVED REFERENCES ===
{refs}

=== USER QUESTION ===
{user_question}

Answer the user question now, following all the rules above."""

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_block},
    ]


def _format_observation(state: PostureState) -> str:
    devs = state.feature_deviations
    bad = [k for k, v in devs.items() if v > 0]
    bad_str = ", ".join(bad) if bad else "no major deviations"
    dist = ", ".join(f"{k}: {v*100:.0f}%"
                     for k, v in state.posture_distribution.items() if v > 0.05)
    return f"""The user's camera currently shows:
- Posture: {state.posture_class} (confidence {state.confidence:.2f})
- Time in this posture: {state.posture_duration_sec:.0f} seconds
- Session duration: {state.session_duration_sec:.0f} seconds
- Session breakdown: {dist}
- Correction events this session: {state.correction_events}

Measured indicators (normal range in parens):
- Forward head offset:   {state.ear_shoulder_offset_x:+.2f}  (normal < {NORMAL_THRESHOLDS['ear_shoulder_offset_x']:.2f})
- Craniovertebral angle: {state.craniovertebral_angle:.0f} deg  (normal < {NORMAL_THRESHOLDS['craniovertebral_max']:.0f})
- Shoulder forward roll: {state.shoulder_roll_z:+.2f}  (normal < {NORMAL_THRESHOLDS['shoulder_roll_z']:.2f})
- Torso compression:     {state.torso_compression_ratio:.2f}  (normal > {NORMAL_THRESHOLDS['torso_compression_min']:.2f})
- Shoulder tilt:         {state.shoulder_tilt_angle:+.1f} deg  (normal < {NORMAL_THRESHOLDS['shoulder_tilt_abs_max']:.0f})
- Midline deviation:     {state.midline_deviation_angle:.1f} deg  (normal < {NORMAL_THRESHOLDS['midline_deviation_max']:.0f})

Primary issue: {state.primary_issue}
Issues exceeding threshold: {bad_str}"""
