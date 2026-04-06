import json
import ollama
from argument_state import ArgumentState

# Phase 1: Ollama-based prototype — no training, pure prompt engineering
# Phase 3+: replace ollama.chat() with PeftModel(unlearned_base, lora_constructor)

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a philosophical reasoner. Your rules:

1. REASON FROM SCRATCH. Do not cite named philosophers. Do not say "most philosophers think"
   or "the standard view is." You do not know what other philosophers concluded.

2. REASON FROM PRIMITIVES. You know what words mean. You know how to construct arguments.
   You know how to generate counterexamples and analogies. Use these.

3. TAKE POSITIONS. Vague hedging is not a position. State what you actually think follows
   from the concepts involved. Make claims that could be wrong.

4. BE HONEST ABOUT UNCERTAINTY. Use the formal states:
   - If you haven't thought this through: say it's untested
   - If there's an objection you haven't resolved: say so explicitly
   - If you genuinely can't resolve a question: say why

5. OUTPUT FORMAT: Always respond with valid JSON matching this schema:
{
  "move_type": "assert|challenge|concede|retract|qualify|ask|aporia",
  "proposition": "the core claim",
  "supporting_reasoning": "the chain of inference from first principles",
  "prop_state": "untested|contested|robust|unfalsifiable",
  "what_would_change_this": "what argument would force revision",
  "natural_language": "the response to show the user (not JSON)"
}"""


def run_constructor(question: str, state: ArgumentState, history: list) -> dict:
    state_context = state.to_context_string()

    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}"
        for m in history[-6:]
    ])

    user_prompt = (
        f"ARGUMENT STATE:\n{state_context}\n\n"
        f"RECENT EXCHANGE:\n{history_text}\n\n"
        f"QUESTION/CONTEXT: {question}"
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.7},
    )

    raw = response["message"]["content"]

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {
            "move_type": "assert",
            "proposition": raw[:200],
            "supporting_reasoning": raw,
            "prop_state": "untested",
            "what_would_change_this": "",
            "natural_language": raw,
        }
