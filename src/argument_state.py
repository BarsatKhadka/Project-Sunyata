from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class PropState(Enum):
    UNTESTED = "untested"           # B hasn't attacked yet
    CONTESTED = "contested"         # B attacked, A hasn't fully resolved
    ROBUST = "robust"               # B attacked, A resolved all attacks
    UNFALSIFIABLE = "unfalsifiable" # B can't attack — content too thin


@dataclass
class Proposition:
    text: str
    reasoning: str
    state: PropState = PropState.UNTESTED
    attacks: list = field(default_factory=list)      # list of B's objections
    resolutions: list = field(default_factory=list)  # A's responses to each attack
    what_would_change: str = ""


@dataclass
class ArgumentState:
    # What A has asserted
    a_commitments: dict = field(default_factory=dict)  # prop_id -> Proposition
    a_retractions: dict = field(default_factory=dict)  # prop_id -> what forced it

    # What the user has committed to
    user_commitments: dict = field(default_factory=dict)

    # Questions neither side has resolved
    contested_questions: list = field(default_factory=list)
    aporic_questions: list = field(default_factory=list)

    # Full history for paper analysis
    revision_history: list = field(default_factory=list)
    round_count: int = 0

    def add_a_commitment(self, prop_id: str, proposition: Proposition):
        for pid, prior in self.a_commitments.items():
            if self._contradicts(proposition.text, prior.text):
                raise ContradictionError(
                    f"New proposition '{proposition.text}' contradicts prior '{prior.text}'. "
                    f"Must retract {pid} first."
                )
        self.a_commitments[prop_id] = proposition

    def detect_user_contradiction(self, new_statement: str) -> Optional[str]:
        for prior in self.user_commitments.values():
            if self._contradicts(new_statement, prior):
                return prior
        return None

    def flag_aporia(self, question: str, reason: str):
        self.aporic_questions.append({
            'question': question,
            'reason': reason,
            'round': self.round_count
        })

    def _contradicts(self, _prop_a: str, _prop_b: str) -> bool:
        # Phase 1: placeholder — always False
        # Phase 4+: replace with NLI-based entailment check
        return False

    def to_context_string(self) -> str:
        """Serialize state for injection into model prompts."""
        committed = [
            f"- {p.text} [state: {p.state.value}]"
            for p in self.a_commitments.values()
        ]
        contested = [f"- {q}" for q in self.contested_questions]
        if not committed:
            return "No commitments yet."
        parts = ["MY CURRENT COMMITMENTS:", "\n".join(committed)]
        if contested:
            parts += ["\nOPEN QUESTIONS:", "\n".join(contested)]
        return "\n".join(parts)


class ContradictionError(Exception):
    pass
