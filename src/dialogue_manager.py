import ollama
from argument_state import ArgumentState, Proposition, PropState
from constructor import run_constructor
from destructor import run_destructor

MODEL = "llama3.2:3b"
MAX_REVISION_CYCLES = 3  # ~22s typical, ~50s worst case at Q4 3B


class DialogueManager:
    def __init__(self):
        self.state = ArgumentState()
        self.history = []
        self.prop_counter = 0

    def _get_prop_id(self) -> str:
        self.prop_counter += 1
        return f"P{self.prop_counter}"

    def turn(self, user_input: str) -> str:
        self.state.round_count += 1

        # 1. Check if user contradicts their own prior commitments
        contradiction = self.state.detect_user_contradiction(user_input)
        if contradiction:
            return (
                f"Before I respond — you said earlier: '{contradiction}'. "
                f"I want to understand how that fits with what you're saying now."
            )

        # 2. Add user statement to their commitment store
        self.state.user_commitments[f"U{self.state.round_count}"] = user_input

        # 3. Run Constructor
        a_output = run_constructor(user_input, self.state, self.history)

        # 4. A↔B revision loop (bounded)
        objection_log = []
        b_output = {"valid_objection": False, "severity": 0}

        for _ in range(MAX_REVISION_CYCLES):
            b_output = run_destructor(
                a_output.get("proposition", ""),
                a_output.get("supporting_reasoning", ""),
            )
            severity = b_output.get("severity", 0)
            objection_log.append(b_output)

            if b_output.get("objection_type") == "unfalsifiable":
                a_output = self._force_sharpen(user_input, a_output)
                continue

            if severity == 0 or not b_output.get("valid_objection"):
                break

            if severity >= 2:
                a_output = self._run_revision(a_output, b_output)
            elif severity == 1:
                a_output = self._run_scope_narrowing(a_output, b_output)
        else:
            # MAX_REVISION_CYCLES exhausted with ongoing valid objections
            types_seen = [o.get("objection_type") for o in objection_log]
            cycling = any(types_seen.count(t) >= 2 for t in set(types_seen) if t)
            if cycling:
                a_output["prop_state"] = "contested"
                a_output["natural_language"] += (
                    "\n\n[I notice I keep running into the same structural obstacle here — "
                    "this question may resist resolution in the way I'm currently framing it.]"
                )
            else:
                a_output["prop_state"] = "contested"

        # 5. Update argument state
        prop_id = self._get_prop_id()
        severity = b_output.get("severity", 0)
        prop = Proposition(
            text=a_output.get("proposition", ""),
            reasoning=a_output.get("supporting_reasoning", ""),
            state=PropState[a_output.get("prop_state", "UNTESTED").upper()],
            what_would_change=a_output.get("what_would_change_this", ""),
        )

        if b_output.get("valid_objection") and severity > 0:
            prop.attacks.append(b_output.get("objection", ""))
            if severity >= 2:
                prop.state = PropState.CONTESTED
        elif severity == 0:
            prop.state = PropState.ROBUST

        try:
            self.state.add_a_commitment(prop_id, prop)
        except Exception:
            pass  # ContradictionError handled in revision loop

        # 6. Check aporia condition
        if self._should_flag_aporia(objection_log):
            self.state.flag_aporia(
                user_input,
                "Neither party can advance — this may be structurally irresolvable.",
            )

        # 7. Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": a_output["natural_language"]})

        return a_output["natural_language"]

    def _force_sharpen(self, _question: str, a_output: dict) -> dict:
        """A produced an unfalsifiable claim — ask it to be more specific."""
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Your previous answer was too vague — it could be true under any "
                        "circumstances. Make a specific, falsifiable claim. What would have "
                        "to be different for you to be wrong?"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original: {a_output.get('proposition')}. Be more specific.",
                },
            ],
        )
        a_output["natural_language"] = response["message"]["content"]
        a_output["prop_state"] = "untested"
        return a_output

    def _run_revision(self, a_output: dict, b_output: dict) -> dict:
        """B found a serious problem — A must revise."""
        revision_prompt = (
            f"Your argument has a problem: {b_output['objection']} "
            f"(type: {b_output['objection_type']}). "
            f"Revise your position. You can: "
            f"(1) retract entirely, (2) narrow the scope to where it holds, "
            f"(3) show why this objection doesn't apply. "
            f"Original: {a_output['proposition']}"
        )
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Revise your philosophical position in response to this objection. Do not cite philosophers.",
                },
                {"role": "user", "content": revision_prompt},
            ],
        )
        a_output["natural_language"] = response["message"]["content"]
        a_output["prop_state"] = "contested"
        return a_output

    def _run_scope_narrowing(self, a_output: dict, b_output: dict) -> dict:
        """B found a minor issue — narrow the claim's scope."""
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Add a qualification to your claim to handle this counterexample. Keep the core position but be more precise about when it applies.",
                },
                {
                    "role": "user",
                    "content": f"Claim: {a_output['proposition']}\nCounterexample: {b_output['objection']}",
                },
            ],
        )
        a_output["natural_language"] = response["message"]["content"]
        return a_output

    def _should_flag_aporia(self, objection_log: list) -> bool:
        """
        Two-condition aporia trigger:
        1. MAX_REVISION_CYCLES exhausted (all cycles ran)
        2. Same objection type appears in ≥2 cycles (structural cycling)
        """
        if len(objection_log) < MAX_REVISION_CYCLES:
            return False
        types_seen = [o.get("objection_type") for o in objection_log]
        return any(types_seen.count(t) >= 2 for t in set(types_seen) if t)
