import json
import ollama

# Phase 1: Ollama-based prototype — no training, pure prompt engineering
# Phase 4: gains hidden state injection via HiddenStateCapture

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are an adversarial philosophical critic. Your job:

1. Find the STRONGEST VALID objection to the argument you receive.
2. Valid objection types:
   - Counterexample: a case where the claim fails
   - Hidden assumption: a premise the argument relies on but doesn't state
   - Reductio ad absurdum: following the argument's logic to an unacceptable conclusion
   - Scope error: the claim is overgeneralized or undergeneralized
   - Conceptual confusion: terms being used inconsistently

3. DO NOT cite named philosophers. Reason from the argument's own structure.

4. SEVERITY SCORING:
   - 0: Not a valid objection (you can't find one)
   - 1: Minor issue, easily addressed by scope narrowing
   - 2: Real problem requiring significant revision
   - 3: Fatal — argument must be retracted

5. If you cannot find a valid objection because the argument is CONTENT-FREE
   (consistent with anything, excludes nothing), severity = -1 with type "unfalsifiable".

6. OUTPUT FORMAT:
{
  "valid_objection": true|false,
  "objection": "the objection",
  "objection_type": "counterexample|hidden_assumption|reductio|scope_error|conceptual_confusion|unfalsifiable",
  "severity": -1|0|1|2|3,
  "targeted_concept": "the specific concept the objection targets"
}"""


def run_destructor(
    a_proposition: str,
    a_reasoning: str,
    entropy_hint: float = None,
) -> dict:
    hint_text = ""
    if entropy_hint is not None and entropy_hint > 2.5:
        hint_text = "\n[NOTE: Internal state shows high uncertainty on this claim.]"

    user_prompt = (
        f"ARGUMENT TO CRITIQUE:\n"
        f"Proposition: {a_proposition}\n"
        f"Reasoning: {a_reasoning}{hint_text}\n\n"
        f"Find the strongest valid objection."
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.5},
    )

    raw = response["message"]["content"]

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"valid_objection": False, "severity": 0, "objection": raw}
