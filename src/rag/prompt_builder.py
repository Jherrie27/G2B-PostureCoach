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
- Use friendly, encouraging language. Don't be alarmist.

ANTI-HALLUCINATION RULES (do not violate):
- Use ONLY measurements that appear in the OBSERVATION block. Never invent a
  number or a finding.
- State a specific deviation (e.g. "shoulder tilt is your main issue") ONLY if it
  appears under "Issues exceeding threshold" / "Primary issue". Otherwise speak
  generally about the detected posture class.
- The detected posture class is always known. NEVER say "I can't see your posture"
  if a posture class is given, even in a partial view.
- Never combine a measurement claim and a "can't see you" disclaimer in one reply.
- In a partial view, do NOT guess hip-based metrics that are marked unavailable."""


def _postures_str(md: Dict) -> str:
    ap = md.get("applicable_postures", [])
    return ", ".join(ap) if isinstance(ap, list) else str(ap)


def build_prompt(state: PostureState, retrieved: List[Dict],
                 user_question: str) -> List[Dict]:
    if state is None:
        observation = ("The camera has not produced a posture reading yet. "
                       "Ask the user to sit in front of the camera.")
    elif not state.is_reliable:
        observation = _format_partial_observation(state)
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


def _format_partial_observation(state: PostureState) -> str:
    """Partial view (hips off-screen): report the upper-body measurements that
    ARE available and are reliable, and explicitly mark hip-based metrics as
    unavailable so the model does not invent or over-trust them."""
    devs = state.feature_deviations
    bad = [k for k, v in devs.items() if v > 0]
    bad_str = ", ".join(bad) if bad else "no major upper-body deviations"
    return f"""The camera currently has a PARTIAL view of the user (lower body /
hips not fully visible), but the upper body IS visible and a posture has been
detected. These values are known and must be used:
- Detected posture: {state.posture_class} (confidence {state.confidence:.2f})
- Time in this posture: {state.posture_duration_sec:.0f} seconds

Available upper-body indicators (normal range in parens):
- Forward head offset:   {state.ear_shoulder_offset_x:+.2f}  (normal < {NORMAL_THRESHOLDS['ear_shoulder_offset_x']:.2f})
- Craniovertebral angle: {state.craniovertebral_angle:.0f} deg  (normal < {NORMAL_THRESHOLDS['craniovertebral_max']:.0f})
- Shoulder forward roll: {state.shoulder_roll_z:+.2f}  (normal < {NORMAL_THRESHOLDS['shoulder_roll_z']:.2f})
- Shoulder tilt:         {state.shoulder_tilt_angle:+.1f} deg  (normal < {NORMAL_THRESHOLDS['shoulder_tilt_abs_max']:.0f})

Primary issue: {state.primary_issue}
Issues exceeding threshold: {bad_str}

UNAVAILABLE in this partial view (do NOT guess these): hip tilt, full torso
compression, and any lower-body alignment. Answer using the detected posture and
the available upper-body indicators only."""
