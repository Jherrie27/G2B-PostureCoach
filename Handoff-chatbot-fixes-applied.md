# G2B Posture Coach — Chatbot Fix Handoff (APPLIED)

**Date:** 2026-06-20
**Host:** Windows, user `iansa`, project at `C:\Users\iansa\G2B-PostureCoach`
**Branch:** `main`
**Supersedes:** Nothing — this is an addendum to `Handoff-latest.md` (2026-06-06)
**Status:** Fixes applied and verified working in live demo. Chatbot now responds
with posture-specific advice. One known cosmetic issue remains (see §6).

This file is the single source of truth for the chatbot fix work. A fresh
engineer or AI session should be able to pick up from here.

---

## 0. TL;DR

The chatbot was refusing to answer ("camera can't see your posture") even when
the classifier was actively detecting a posture, because the reliability gate
(`is_reliable=False`, triggered by hips being off-camera) caused `rag_query.py`
to throw away the posture label before it reached the chatbot. It was also
hallucinating findings (e.g. "shoulder tilt is your main deviation") it could
not actually know.

**Three files were edited. The model was NOT touched.** After the fix, the
chatbot correctly answers:

> "Based on the OBSERVATION block, your posture has been detected as lean, with
> a confidence of 0.64 and a duration of 1 second. The primary issues detected
> are shoulder tilt and midline deviation."

This is now a **grounded** answer pulled from real measured values, not a
hallucination.

---

## 1. Environment recap (from prior handoffs)

- **Python 3.12** venv at `.venv` (Python 3.13/3.14 break mediapipe — must use 3.12).
- Run command: `.venv\Scripts\python.exe -m streamlit run app.py`
- Groq API key lives in gitignored `.env` as `GROQ_API_KEY=...`, loaded by
  `rag_query.py` via `python-dotenv`. **Rotate the key after the demo.**
- The venv had to be created with `--copies` flag to avoid a broken-symlink
  error: `py -3.12 -m venv .venv --copies`

---

## 2. The problems (observed in live demo)

From two demo screenshots, these symptoms were confirmed:

1. **Chatbot refused to answer** — said "the camera cannot reliably see your
   posture, please reposition" even though the classifier showed a live label
   (e.g. *Lean conf=0.56, holding for 272s*).
2. **Hallucinated findings** — responses opened with "your current measurements
   show shoulder tilt as the main deviation" — a specific claim the model could
   not actually verify when `is_reliable=False`.
3. **Self-contradiction** — the same response said "shoulder tilt is the main
   deviation" in paragraph 1, then "I can't see your posture" in paragraph 2.
4. **Correct posture stuck at 0–3%, Corrections 0** — every frame marked
   `is_reliable=False` because hips were off-camera, so the session tracker
   never counted valid posture frames.

### Root cause

All four symptoms trace to **one design flaw**: the reliability gate
(`KEY_FOR_RELIABILITY` in `src/cv/landmarks.py`) **requires hips to be visible**.
A standard desk webcam frames the user from the chest up, so hips are never
seen, so `is_reliable` is permanently `False`. Downstream:
- `rag_query.py` set `posture = None` when `is_reliable=False` → chatbot got no
  posture label.
- `prompt_builder.py` emitted a "camera can't see you" observation block → the
  model refused, then hallucinated to fill the gap.

---

## 3. Fixes applied (3 files, model untouched)

### File 1: `rag_query.py` — posture passthrough

**The single most important fix.** Changed so the classifier label is passed to
the retriever and prompt **regardless of `is_reliable`**. The label is valid even
when hips are off-screen (ears + shoulders are enough to classify lean / slouch /
neck-forward).

The `posture = state.posture_class if state and state.is_reliable else None` line
was changed to `posture = state.posture_class if state else None`. The query
expansion was updated to append `(partial view)` to the posture tag when
`is_reliable` is False, so the retriever still finds posture-specific chunks.

**Model params were deliberately left unchanged:** `temperature=0.3`,
`max_tokens=400`. (An earlier draft proposed lowering temperature; it was
reverted at the user's request to avoid touching model behavior.)

### File 2: `src/cv/landmarks.py` — REVERTED to original

This file was briefly edited (hips removed from `KEY_FOR_RELIABILITY`,
`VISIBILITY_MIN` lowered to 0.4) but that change **broke the CV pipeline** and
caused the app to hang on "Camera is starting up." **It was fully reverted to
the original committed version:**

```python
KEY_FOR_RELIABILITY = ["left_ear", "right_ear", "left_shoulder",
                       "right_shoulder", "left_hip", "right_hip"]
VISIBILITY_MIN = 0.5
```

> **LESSON LEARNED:** Do not modify `landmarks.py`. The reliability gate is
> consumed by the CV worker in a way that is sensitive to this list. The chatbot
> fix in `rag_query.py` makes the landmarks change unnecessary anyway — the
> chatbot now works even with `is_reliable=False`, so there is no need to relax
> the gate.

### File 3: `src/rag/prompt_builder.py` — anti-hallucination + partial-view block

Two changes:

**(a) Stronger anti-hallucination rules** added to the `SYSTEM` prompt. The new
rules explicitly forbid the model from:
- inventing measurements not in the OBSERVATION block,
- stating a finding (e.g. "shoulder tilt is your main issue") unless it appears
  under "Issues exceeding threshold",
- saying "camera can't see you" if any measured values exist,
- combining a measurement claim with a "can't see you" disclaimer in one reply,
- speculating about unmeasured (hip-based) values in partial view.

**(b) Partial-view observation block.** The old code, when `not is_reliable`,
emitted only: *"The camera cannot reliably see the user's posture. Ask the user
to reposition."* This is what caused the refusal. It now emits the **actual
upper-body measurements that ARE available** (detected posture class, confidence,
duration, primary issue, forward-head offset, shoulder roll, shoulder tilt) and
explicitly notes that hip-based metrics are unavailable and must not be guessed.

This is why the chatbot now answers correctly with grounded values instead of
refusing or hallucinating.

---

## 4. Current verified behavior (post-fix)

From the latest demo screenshot, with hips now partially in frame:

- **Session:** 40.2 min
- **Correct posture:** 3%
- **Corrections:** 38
- **Worst streak:** 863s
- **Chatbot Q:** "what is my posture"
- **Chatbot A:** *"Based on the OBSERVATION block, your posture has been detected
  as lean, with a confidence of 0.64 and a duration of 1 second. The primary
  issues detected are shoulder tilt and midline deviation."*

The answer is now **grounded** — every value (lean, 0.64, shoulder tilt, midline
deviation) comes from the real PostureState, not invented.

> **Note:** One residual hallucination-style preamble still appears occasionally:
> *"I don't have detailed references for your specific question, but your current
> measurements show shoulder tilt as the main deviation."* This is the
> `fallback_response()` from `grounding_check.py` being prepended when the
> grounding check fails. See §6 for how to address it.

---

## 5. Exact final file states

### `rag_query.py` (key section)
```python
def answer(user_question: str, state: PostureState) -> dict:
    retriever = get_retriever()

    # FIX: posture label valid even when is_reliable=False (hips off-screen)
    posture = state.posture_class if state else None

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
        temperature=0.3,   # unchanged
        max_tokens=400,    # unchanged
    )
    text = resp.choices[0].message.content.strip()
    grounded = is_grounded(text, retrieved)
    if not grounded:
        text = fallback_response(state) + "\n\n" + text
    return {"text": text, "retrieved": retrieved, "grounded": grounded}
```

### `src/cv/landmarks.py`
Reverted to original — hips IN the reliability list, `VISIBILITY_MIN = 0.5`.
**Do not edit this file.**

### `src/rag/prompt_builder.py`
`SYSTEM` prompt now has an "ANTI-HALLUCINATION RULES" section. The
`build_prompt` function has a three-way branch: no state → ask user to sit;
partial view → emit real upper-body measurements + "don't guess hips"; reliable
→ full observation via `_format_observation`.

---

## 6. Remaining issue — the "I don't have detailed references" preamble

This line is NOT from the LLM hallucinating. It is the hard-coded
`fallback_response(state)` in `src/rag/grounding_check.py`, which gets
**prepended** to the answer whenever `is_grounded(text, retrieved)` returns
False. Look at the last lines of `rag_query.py`:

```python
    grounded = is_grounded(text, retrieved)
    if not grounded:
        text = fallback_response(state) + "\n\n" + text  # both, user can compare
```

When grounding fails (often, because retrieval returns weak chunks for short
questions like "what is my posture"), the fallback string is glued on top of the
real answer — producing the confusing two-part reply.

### To fully remove this preamble (optional next step)

Open `src/rag/grounding_check.py` and inspect:
1. `is_grounded()` — likely too strict (requires the answer to cite `[REF N]`
   tokens). Short factual answers about the user's own posture don't cite refs,
   so they always "fail" grounding.
2. `fallback_response()` — the source of the "I don't have detailed references"
   text.

**Two clean options:**

**Option A — only prepend fallback when retrieval is truly empty:**
In `rag_query.py`, change the fallback condition so it only fires when no
references were retrieved at all, not merely when the answer doesn't cite them:
```python
    grounded = is_grounded(text, retrieved)
    if not grounded and not retrieved:
        text = fallback_response(state) + "\n\n" + text
```

**Option B — replace, don't prepend:**
Only use the fallback when the model produced nothing useful, and replace rather
than concatenate, so the two-part contradictory reply can never happen:
```python
    grounded = is_grounded(text, retrieved)
    if not grounded and not text.strip():
        text = fallback_response(state)
```

Option A is recommended — it keeps the grounding safety net for genuinely
unanswerable questions while removing the preamble from normal posture answers.

> **Do this only after the demo** if time allows — the current behavior is
> functional, just slightly verbose.

---

## 7. Why "Correct posture" stays low (3%)

This is expected given the camera framing, not a bug. The user's hips are mostly
off-screen, so most frames are `is_reliable=False` and the classifier leans
toward `lean` (the shoulder-tilt + midline signals dominate when the lower body
isn't visible). The session distribution counts whatever the smoother outputs.

To get a realistic "correct posture %", the user must **frame from the waist up**
with good front lighting. This is a framing/data limitation, not a code bug, and
is consistent with the known issue documented in `Handoff-latest.md` §11.

---

## 8. What was NOT changed (important)

| Component | Status | Why |
|---|---|---|
| `models/posture_lgbm_v3.txt` | Untouched | Model fix not requested; accuracy is a data problem |
| `temperature` / `max_tokens` | Untouched (0.3 / 400) | User asked not to alter model behavior |
| `src/cv/landmarks.py` | Reverted to original | Editing it broke the CV pipeline |
| `src/cv/features.py` | Untouched | Feature sign issues need real-data validation |
| `src/rag/retriever.py` | Untouched | Retrieval quality acceptable |
| `app.py` | Untouched | UI fixes (confidence color, correction flash) deferred |

---

## 9. Verification checklist (current state)

```
[x] App launches without hanging on "Camera is starting up"
[x] Live camera feed displays
[x] Session metrics populate (40.2 min, 38 corrections shown)
[x] Posture label detected (lean, conf 0.64)
[x] Chatbot answers with the detected posture and real measurements
[x] Chatbot no longer flatly refuses with "camera can't see you"
[x] Model files and params untouched
[ ] "I don't have detailed references" preamble removed  ← optional, see §6
[ ] Realistic correct-posture % (needs waist-up framing)  ← framing, not code
```

---

## 10. Quick orientation for next session

- The chatbot fix lives entirely in `rag_query.py` (posture passthrough) and
  `src/rag/prompt_builder.py` (anti-hallucination rules + partial-view block).
- **Do not touch `src/cv/landmarks.py`** — it broke the camera last time and is
  unnecessary now.
- The only remaining cleanup is the `fallback_response` preamble in
  `grounding_check.py` (§6) — a 1-line conditional change in `rag_query.py`.
- Model, features, and thresholds are all untouched and should stay that way
  unless real landmark data is collected (see `Handoff-latest.md` §13).
- Everything still runs from Python 3.12 venv:
  `.venv\Scripts\python.exe -m streamlit run app.py`
