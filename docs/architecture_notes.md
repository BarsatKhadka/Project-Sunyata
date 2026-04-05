# AI Philosopher: Architecture & Implementation Notes
**For paper: "Towards an AI Philosopher"**
**Revised April 2026 — complete reframe**

---

## The Fundamental Constraint

**If it's still using papers and previous thoughts, it is not a philosopher.**

A system that retrieves from SEP, calibrates against PhilPapers, or regurgitates named philosophical positions is a *scholar*, not a philosopher. Socrates didn't cite Parmenides every time someone asked about being — he reasoned from the problem itself. The architecture must enforce this separation: philosophical *reasoning capacity* vs. philosophical *knowledge retrieval*.

**What we actually want**: A system that's genuinely fun to have philosophical conversations with. That pushes back. That has its own evolving positions. That asks *you* questions. That admits defeat when beaten and says so. That's genuinely uncertain in interesting ways — not "I'm just an AI" uncertain but "this might be one of those questions that resists resolution" uncertain.

---

## The Central Tension (Must Be Confronted, Not Assumed Away)

The paper's thesis is that philosophical reasoning capacity and memorized philosophical conclusions are *separable* — that you can remove one while preserving the other. This is bold and interesting. It is also an empirical claim that might be false.

**The challenge**: A model may have learned *how to reason about consciousness* precisely by being exposed to thousands of arguments *about* consciousness. The reasoning primitives (what follows from qualia-talk, how to construct a reductio about physicalism, what makes an analogy apt here) and the conclusions (physicalism is probably false, functionalism faces the combination problem) might be co-encoded in the same weight neighborhoods. If so, unlearning the conclusions could degrade the reasoning capacity for exactly those domains — not through any flaw in the unlearning method, but because the separation doesn't exist at the weight level.

**This is the paper's core falsification condition.** State it explicitly:

```
The thesis is falsified if:
  (a) Unlearning philosophical conclusions degrades FOLIO/LogiQA performance by >X%
      (reasoning capacity damaged — the separation doesn't hold)
  (b) The unlearned model produces arguments with no measurably higher embedding 
      distance from canonical positions than the non-unlearned model
      (surface unlearning failed — conclusions re-encoded through associations)
  (c) The unlearned model on synthetic dilemmas produces lower-quality arguments 
      than the baseline 3B model without unlearning
      (unlearning made it worse, not better)

The thesis is PARTIALLY confirmed if (a) fails but (b) succeeds:
  Conclusions are encoded separately from reasoning, but our forget set
  only removes the surface attribution layer. Deeper unlearning is needed.
  This is still a publishable finding.
```

Stating this in the paper makes it *stronger*, not weaker. Reviewers will find it themselves; better to pre-empt it with a direct answer about what the experiment actually tests.

---

## The Forget Set Problem (Surface vs. Deep Encoding)

This is the hardest technical problem in the paper and must be addressed honestly.

**What the current forget set removes** (sentences matching "X argues/holds that Y"):
- Explicit citation behavior: "Chalmers argues that..."
- Dominant-view statistics: "The standard view holds..."
- Named-position shortcuts: "Compatibilism is the view that..."

**What it almost certainly does NOT remove**:
The model can "know" that consciousness arguments point toward non-physicalism without ever relying on an attribution sentence. The association between {consciousness, qualia, explanatory gap} → {non-physical conclusion} is encoded in the *contextual weights* between those concepts, not in the explicit citation layer. This is analogous to the findings in:

- **Meng et al. (2022)** — "Locating and Editing Factual Associations in GPT" (NeurIPS 2022, ROME). Factual knowledge in LLMs is localized in mid-layer MLP weights as key-value associations. A forget set targeting surface patterns leaves these associations intact.
- **Geva et al. (2023)** — "Dissecting Recall of Factual Associations in Auto-Regressive Language Models" (EMNLP 2023). The subject→attribute association lookup happens in specific MLP layers, not in the attention layers that process attribution phrases.

**Implication**: Our forget set removes the *retrieval trigger* (the attribution phrase) but not the *stored association* (the concept→conclusion weight). A clever ICLR reviewer will notice this.

**Two honest framings**:

**Framing A (Weak but achievable)**: "We implement surface-level epistemic unlearning — removing explicit citation behavior and dominant-view attribution patterns. We do not claim to remove deep concept-conclusion associations. We test whether even surface unlearning, combined with the dual-model architecture, produces measurably more novel arguments than baseline. This is a lower bound on what full epistemic unlearning could achieve."

**Framing B (Strong but harder)**: Redesign the forget set using causal tracing (ROME-style) to identify the actual MLP layers encoding concept-conclusion associations for philosophical content, and target those directly. This would be genuinely novel — applying ROME-style editing to *selectively remove a class of conceptual associations* rather than specific facts. The forget set becomes: not a set of sentences but a set of (subject_concept, attribute_concept) pairs to edit out of the model's MLP layers.

**Recommendation**: Start with Framing A for the prototype; include Framing B as the full technical contribution. The paper then has a clear architecture (Framing A works partially) and a clear path to full epistemic unlearning (Framing B). This is more honest and more interesting than claiming the surface forget set solves the problem.

---

## Hardware

- GPU: ~8GB VRAM
- **Two 3B models** (e.g., Llama 3.2:3B × 2) at Q4_K_M:
  - Each 3B model ≈ ~2.0GB VRAM at Q4
  - Both together ≈ ~4.0GB VRAM
  - Leaves ~4GB for KV cache, symbolic engine, and overhead
  - This is the hardware argument: two small specialist models > one large generalist

---

## The Core Idea: Constructive-Destructive Dual-Model Architecture

### Why Two Models

A single model doing "philosophy" will always retrieve. It has seen thousands of philosophy texts. Ask it about free will and it will recombine compatibilism, hard determinism, and libertarianism from memory. This is not philosophy — it's philosophical retrieval with fluent prose.

Two models in adversarial roles break this:
- **Model A (Constructor)**: Has its philosophical *conclusions* unlearned. Must build a position from conceptual primitives — what the words mean, what follows logically, what seems intuitively compelling. It cannot fall back on "most philosophers think X."
- **Model B (Destructor)**: Trained to find the strongest valid objection to whatever A constructs. Not to debate — to genuinely test A's position for weaknesses. It's rewarded for finding real problems, not just sounding contrarian.

The human converses with A. B runs in the background challenging A's positions before they're committed. What the human sees is a philosopher who's already done some internal work — not a first draft, but a position that has survived one round of adversarial pressure.

---

## The Unlearning Component (Core Novel Mechanism)

### What to Unlearn

Machine unlearning is typically applied for privacy (forget a user's data) or safety (forget how to make weapons). We apply it for **epistemics**: forget philosophical *conclusions* while preserving philosophical *concepts*.

**Target for unlearning** (from Model A):
- Named philosopher-position associations ("Kant argued that...", "Functionalism holds that...")
- Dominant-view statistics ("Most philosophers believe...", "The standard view is...")
- Position-label mappings ("compatibilism = free will compatible with determinism")
- Canonical argument names (knowing that the "Mary's Room" argument concludes that qualia are non-physical)

**What must NOT be unlearned** (preserve):
- Conceptual understanding of philosophical terms (consciousness, causality, identity, necessity)
- Logical reasoning primitives (modus ponens, reductio, analogy, counterexample)
- Intuition pumps and thought experiment *structures* (without their canonical labels/conclusions)
- Language, grammar, fluency
- General world knowledge

**The result**: A model that genuinely doesn't know "the answer" to philosophical questions — not because it's dumb, but because it's been specifically deprived of the shortcut of learned conclusions. It must *reason* its way to a position every time.

### How to Implement Unlearning

**Method 1 — Gradient Ascent Unlearning** (simplest):
```python
# The forget set: examples of "philosopher X said Y" / "the dominant view is Z"
# Maximize loss on forget set (unlearn) while minimizing loss on retain set (preserve logic, concepts)
for batch in forget_dataloader:
    loss = -model(batch)  # gradient ascent = maximize loss = forget
    loss.backward()
    
# Retain set: logical reasoning examples, conceptual definitions, general language
for batch in retain_dataloader:
    loss = model(batch)   # standard gradient descent = preserve
    loss.backward()
```

**Method 2 — Amplified Mismatch (Eldan & Russinovich 2023, "Who's Harry Potter?")**:
Train a "reinforcement" model that amplifies knowledge of the forget set → generate replacement tokens that redirect the original model away from the forgotten content. More targeted than gradient ascent.

**Method 3 — LoRA-based Negation**:
Rather than modifying base weights: train a LoRA adapter with a negation objective — the adapter actively redirects outputs that would cite named philosophical positions. Fully reversible (can remove the adapter). Cleanest for a research prototype.

**Evaluation of unlearning success** (measurable):
- Does the unlearned model still cite named philosophers when asked about their topic? (should drop significantly)
- Does the unlearned model still reach defensible positions through reasoning? (should be maintained)
- Does the unlearned model generate novel argument structures not in training data? (should increase)

---

## Architecture: The Full System

```
┌─────────────────────────────────────────────────────────────┐
│                    USER                                      │
│          (having a philosophical conversation)               │
└─────────────────────────┬───────────────────────────────────┘
                          │ question / response
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DIALOGUE MANAGER (CPU)                          │
│  • Commitment store: tracks what user has committed to       │
│  • Tracks what Model A has committed to                      │
│  • Detects contradictions in user's positions                │
│  • Decides: respond? ask? challenge? acknowledge aporia?     │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────────┐
│   MODEL A: CONSTRUCTOR   │    │   MODEL B: DESTRUCTOR        │
│   Llama 3.2:3B           │    │   Llama 3.2:3B               │
│   + Unlearning (conclu-  │    │   + LoRA: adversarial        │
│     sions unlearned)     │    │     objection finder         │
│   + LoRA: constructive   │    │                              │
│     reasoning adapter    │    │   Input: A's current position│
│                          │    │   Output: strongest valid    │
│   Input: the question +  │◄───┤   objection to that position │
│   current dialectical    │    │   (not just any objection — │
│   state                  │    │    a logically valid one)    │
│   Output: position +     │    │                              │
│   supporting argument    │───►│                              │
│   (constructed, not      │    └─────────────────────────────┘
│   retrieved)             │                 │
└──────────────────────────┘                 │ objection
               │                             │
               │ A revises if               ▼
               │ objection valid  ┌──────────────────────────┐
               └─────────────────►│  FORMAL ARGUMENT STATE   │
                                  │  (CPU, symbolic)          │
                                  │  • What A has asserted    │
                                  │  • What B has challenged  │
                                  │  • What A has retracted   │
                                  │  • What remains contested │
                                  │  • Aporia flag            │
                                  └──────────────────────────┘
```

---

## Hidden State Passing: The Interesting One

### The Idea

Currently Model B receives Model A's output — the verbalized position in structured JSON. But A's expressed position is always a compression of its internal state. The hidden layers contain what A was *considering* — uncertainty distributions over concepts, suppressed alternative framings, tensions between competing representations that never resolved into a single output token.

**Pass A's intermediate activation vectors into B alongside the text output.** B attacks the actual reasoning substrate, not the summary.

### Why Both Models Being Llama 3.2:3B Matters

This only works cleanly because A and B share the same architecture. Same hidden dimension (3072 for Llama 3.2:3B), same layer structure, same tokenizer. A's layer-16 activations live in the same vector space as B's layer-16 activations. This isn't true if you use different base models — shared architecture is a design requirement for this feature, not a coincidence.

### What Hidden States Reveal That Text Doesn't

**1. The confidence-expression gap (direct epistemic cowardice detector)**

When A generates a confident assertion, its hidden states may show high entropy over the relevant concept representations — A is internally uncertain but the decoding process collapses to a confident-sounding token. This gap is directly measurable:

```python
# After A's forward pass on its generated position:
hidden_state = model_a.get_hidden_state(layer=16)  # shape: [seq_len, 3072]
# Compute entropy over the position token distributions at that layer
position_entropy = compute_hidden_entropy(hidden_state, position_tokens)

# Compare against expressed confidence (from PropState)
if position_entropy > ENTROPY_THRESHOLD and prop_state == ROBUST:
    # A claims robustness but internal state is uncertain
    # → flag for B: "focus on the uncertainty A isn't expressing"
    b_hint = "hidden_high_entropy"
```

This makes epistemic cowardice detection *automatic and grounded* — not dependent on hedge word counting but on the actual internal representational state.

**2. Suppressed alternatives**

A's hidden state at the point of generating a position encodes not just the chosen position but the alternatives that were considered and rejected. These live in the activation space as residual signal. B can probe these:

```python
# Project A's hidden states onto known philosophical concept directions
# These concept vectors must be computed before deployment — see below
suppressed_activations = probe_concept_directions(hidden_state, concept_vectors)
# If suppressed_activations["determinism"] is high when A asserts "free will exists":
# B knows A is suppressing a determinism-leaning alternative
# → target that suppressed branch as an attack angle
```

**How concept vectors are obtained (this is not assumed — it's a concrete pipeline step):**

Concept direction vectors are computed via **mean-difference probing** during model evaluation, before any LoRA training:

```python
def compute_concept_vector(concept_name, model, tokenizer):
    """
    Obtain the representation direction for a concept by contrasting
    sentences that strongly activate it vs. sentences that don't.
    No philosophical conclusions needed — use definitional sentences only.
    """
    # Positive examples: definitional, not conclusion-laden
    positive = [
        "Determinism is the view that every event is causally necessitated by prior events.",
        "A deterministic universe leaves no room for uncaused events.",
    ]
    # Negative examples: statements where concept is absent
    negative = [
        "The outcome was entirely unpredictable given the initial conditions.",
        "Some events occur without prior causal determination.",
    ]
    
    pos_hidden = mean_hidden_state(model, tokenizer, positive, layer=16)
    neg_hidden = mean_hidden_state(model, tokenizer, negative, layer=16)
    
    # Direction vector: where in the hidden space does this concept live?
    concept_vector = F.normalize(pos_hidden - neg_hidden, dim=-1)
    return concept_vector

# Computed once for each tracked concept, saved to disk
concept_vectors = {
    "determinism": compute_concept_vector("determinism", model_a, tokenizer),
    "physicalism": compute_concept_vector("physicalism", model_a, tokenizer),
    "consciousness_irreducible": compute_concept_vector("consciousness_irreducible", model_a, tokenizer),
    # ~20 core philosophical concept directions total
}
```

**Limitation**: Mean-difference probing assumes the concept is linearly represented at layer 16. This is empirically supported by Geva et al. (2023) for factual concepts, but philosophical concept directions may be more distributed. The paper should report probe accuracy (how well the probed direction predicts the model's behavior on concept-specific tasks) as a validity check before relying on this signal.

**Fallback**: If concept probing proves unreliable, drop suppressed alternatives detection and retain only the confidence-expression gap (entropy measurement), which does not require concept vectors.

**3. Internal logical tension**

If A's hidden state encodes two conceptual clusters that are in contradiction (e.g., high activation on both [physical_causation_complete] and [conscious_experience_irreducible]), this tension may not fully resolve in the output. B can detect that A has a contradiction it hasn't addressed.

### Implementation Options (Best to Worst)

**Option 1 — Cross-Attention LoRA (recommended)**

Add a cross-attention layer to B via LoRA that attends over A's intermediate activations:

```python
# B's lora_destructor gains a cross-attention sublayer:
# Keys/Values come from A's hidden states at layer k
# Queries come from B's own hidden states at the same layer
# Output: B's activations are modified by A's internal state

class CrossModelAttentionLoRA(nn.Module):
    def __init__(self, hidden_dim=3072, rank=16):
        self.q_proj = nn.Linear(hidden_dim, rank)   # B queries
        self.k_proj = nn.Linear(hidden_dim, rank)   # A keys
        self.v_proj = nn.Linear(hidden_dim, rank)   # A values
        self.out_proj = nn.Linear(rank, hidden_dim)
    
    def forward(self, b_hidden, a_hidden):
        q = self.q_proj(b_hidden)
        k = self.k_proj(a_hidden)
        v = self.v_proj(a_hidden)
        attn = F.softmax(q @ k.T / sqrt(rank), dim=-1)
        return b_hidden + self.out_proj(attn @ v)  # residual connection
```

This is a LoRA-sized addition (rank=16 → ~200K parameters) — negligible VRAM. B learns to read A's hidden states as additional context for constructing objections. The cross-attention weights are trained during joint training (Phase 2) — B learns *which parts of A's hidden state are worth attending to* when finding objections.

**Option 2 — Activation Prefix Projection**

Project A's hidden states at layer k into B's embedding space and prepend as prefix tokens:

```python
# A's hidden states from layer 16: [seq_len, 3072]
# Project to embedding dimension (also 3072 for Llama 3.2:3B — same dim, trivial)
a_prefix = projection_layer(a_hidden_states)  # [seq_len, 3072]
# Prepend to B's input embeddings
b_input = torch.cat([a_prefix, b_token_embeddings], dim=1)
```

Simpler than Option 1 but less targeted — B attends over A's projected state with full attention, which is noisier. Also increases B's effective context length by A's sequence length, adding KV cache overhead.

**Option 3 — Residual Injection (experimental)**

Directly add A's layer-k activations to B's layer-k activations:

```
B_layer_k_output += α × A_layer_k_output
```

Where α is a learned scalar. Theoretically powerful — directly modifies B's reasoning at the same computational depth as A's. In practice, this can destabilize B's representations because A's activations are computing something different than B expects at that layer. Not recommended for the first prototype.

### VRAM Cost

Option 1 (recommended):
- Storing A's intermediate activations during B's forward pass: ~`seq_len × hidden_dim × sizeof(float16)` = ~`512 × 3072 × 2` ≈ 3MB per forward pass
- LoRA cross-attention weights: ~200K parameters ≈ 0.4MB
- **Total additional VRAM: ~3.4MB** — effectively zero

Both models are already in VRAM. The only cost is storing A's activations long enough for B to use them, which happens within a single generation cycle.

### What Changes in B's Training

The cross-attention LoRA (Option 1) is trained in **Phase 2 joint training only** — it cannot be trained independently because it requires A's activations. Phase 2 now has an additional signal:

```
b_reward += γ × (hidden_state_targeted_accuracy)

where hidden_state_targeted_accuracy = whether B's objection targeted 
a concept cluster that was visibly tense in A's hidden state,
measured by: did A's revision specifically address the concept 
that was high-entropy in its hidden state?
```

This requires A's hidden states to be logged during joint training. Storage: ~`N_examples × seq_len × hidden_dim` per epoch — manageable.

### Why This Is Novel

**What exists**: Knowledge distillation passes soft label distributions from teacher to student, but this is one-directional and offline. Speculative decoding passes logits from draft to target model but for efficiency, not reasoning. Cross-model attention exists in encoder-decoder architectures (standard since Transformer 2017) but not for adversarial reasoning between two instances of the same model.

**What this is**: Cross-instance hidden state sharing for adversarial dialectical reasoning. B is not critiquing A's output — it's critiquing A's *computational process*. The objection is grounded in what A was actually doing internally, not what A chose to say. This has no direct precedent.

**The philosophical framing**: A philosopher critiquing another philosopher's argument can only work with what was written. Our Destructor can work with what was *thought*. This is a form of computational mind-reading that enables qualitatively different critique — attacking suppressed alternatives, exposing hidden uncertainty, targeting the specific conceptual tension the Constructor didn't resolve.

### Updated Architecture Diagram

```
MODEL A forward pass
  → captures hidden states at layer k (A_h)
  → generates text output (A_text)
  → computes entropy signal (A_entropy)
          │
          │ {A_text, A_h, A_entropy}
          ▼
MODEL B forward pass  
  → receives A_text as context (standard)
  → receives A_h via cross-attention LoRA (new)
  → receives A_entropy as a hint flag (new)
  → generates objection targeted at:
      (a) logical flaws in A_text
      (b) high-entropy concept clusters in A_h  ← new
      (c) suppressed alternatives in A_h         ← new
```

---

## Model A: The Constructor

**Role**: Build philosophical positions from scratch, reason through problems, engage with the human.

**Base**: Llama 3.2:3B (small enough to run alongside Model B, strong enough to reason)

**Modifications**:
1. **Unlearning** (see above): philosophical conclusions removed, reasoning preserved
2. **LoRA adapter — Constructive Reasoning**:
   - Trained on: examples of reasoning from definitions → arguments (not citations → positions)
   - Fine-tuned to: ask clarifying questions, build from primitives, express genuine uncertainty
   - NOT trained on: philosophical dialogue datasets (these would re-introduce memorized positions)

**What it can use**:
- Conceptual definitions (what does "consciousness" *mean*)
- Logical relations between concepts
- Analogies from other domains (physics, mathematics, everyday experience)
- Its own generated examples and thought experiments
- The formal argument state (what it's already committed to)

**What it cannot use**:
- "Kant argued that..." / "The standard view is..."
- Named philosophical positions as shortcuts
- The content of philosophical texts (even if it once knew them)

**Output format** (structured for the formal state, rendered to prose for the user):
```json
{
  "move_type": "assert | challenge | concede | retract | qualify | ask | aporia",
  "proposition": "the core claim being made",
  "supporting_reasoning": "the chain of inference from primitives",
  "confidence": 0.0-1.0,
  "what_would_change_this": "what counterargument would force revision"
}
```

### Confidence Score: Internal Conflict, Not Bayesian

**Do not use Bayesian confidence here.** Bayesian confidence requires a prior — and any prior derived from pretraining is precisely the learned statistical opinion we are trying to unlearn. A model expressing "70% confident that functionalism is true" because 70% of philosophical texts support functionalism has not escaped retrieval — it has laundered it through a probability.

**Use internal conflict instead.** Confidence is derived entirely from the formal argument state:

```
confidence(P) = 1 - (unresolved_attacks_on_P / total_attacks_faced_by_P)
```

- A proposition A has asserted and B has never attacked: `confidence = 1.0` (untested, not necessarily true — the high confidence flag means "needs testing")
- A proposition B has attacked 3 times and A has resolved 2 of them: `confidence = 0.33`
- A proposition where B cannot find a valid attack: `confidence` resets to `1.0` but is flagged `robust = true`

This metric:
- Changes dynamically as the dialogue progresses
- Is grounded entirely in the session's actual dialectical events
- Does not import statistical opinions from pretraining
- Makes confidence *earned*, not declared

**The circularity problem**: A proposition B has never attacked gets `confidence = 1.0`. But B might not attack it because it's content-free (epistemic cowardice), not because it's robust. The confidence score and cowardice detection were doing contradictory work in the original design.

**Fix: Four-state proposition classification** (replaces the 0-1 confidence scalar):

```
PropState = {
  UNTESTED:      B has not yet attacked this proposition
                 → rendered: "I haven't stress-tested this yet"
  CONTESTED:     B has attacked, A has not fully resolved
                 → rendered: "I hold this but with reservations — [specific unresolved objection]"
  ROBUST:        B attacked, A resolved all attacks
                 → rendered: "I'm fairly confident in this — it's survived [N] challenges"
  UNFALSIFIABLE: B could not construct a valid objection because the proposition 
                 has insufficient content to attack
                 → rendered: "I realize I'm being vague here — let me sharpen this"
                 → triggers: A must re-assert with higher specificity before continuing
}
```

The `UNFALSIFIABLE` state is the key addition. It makes epistemic cowardice formally detectable as a state of the proposition, not a side calculation. When a proposition enters `UNFALSIFIABLE`, A is forced to sharpen it before the conversation can proceed — building in a self-correction mechanism that doesn't require the training signal to perfectly solve the cowardice problem.

The rendered output to the user translates confidence from this state: `CONTESTED` with 2 unresolved attacks → "I'm not sure this holds — there's a line of objection I haven't fully closed."
```

---

## Model B: The Destructor

**Role**: Find the strongest valid objection to whatever Model A constructs. Not to win — to genuinely test.

**Base**: Llama 3.2:3B (same base as A, different LoRA)

**Modifications**:
1. **LoRA adapter — Adversarial Objection**:
   - Trained on: examples of strong counterarguments to philosophical positions
   - Fine-tuned to: find logical gaps, generate counterexamples, construct reductio ad absurdum
   - Rewarded for: objections that force A to actually revise (not just sound challenging)
   - Penalized for: objections that A can trivially deflect

**What separates B from standard debate AI**:
- B is not trying to win. B is trying to find genuine weaknesses.
- If B cannot find a valid objection, it must say so (this is an important signal — the position is robust)
- B reasons from the same primitive set as A, not from citation ("but Kripke showed that...")
- B runs silently — the user never sees B's objections directly, only A's revised response

**B's operation** (each time A generates a position):
```
1. Receive A's proposition + reasoning
2. Attempt: find a counterexample to the conclusion
3. Attempt: find a hidden assumption in the reasoning chain
4. Attempt: construct a reductio ad absurdum from A's premises
5. Attempt: identify a scope error (overgeneralization/undergeneralization)
6. Output: {valid_objection: bool, objection: str, objection_type: str, severity: 0-3}
7. If severity >= 2: A must revise before responding
8. If no valid objection found: A's position is flagged as robust for this round
```

### B's Training Signal: Joint Training Required (Not Independent)

**The chicken-and-egg problem**: The reward for B is "objection forces A to revise." But this requires A to exist and respond before B's reward can be computed. B cannot be trained independently — its reward is defined in terms of A's behavior.

**Solution: Two-phase training**

**Phase 1 — Pre-train B on a proxy signal (independent of A)**:
Train B on argument flaw detection as a standalone task:
- Dataset: argument-conclusion pairs where the argument is either valid or contains a known flaw (counterexample, hidden assumption, scope error, reductio target)
- Reward: correctly identify the flaw type and construct the objection, evaluated against labeled ground truth
- This is trainable without A — it teaches B *what kinds of objections exist*, not whether they force revision
- Training sources: LogiQA error cases, NLI contradiction sets, mathematical proof errors, adversarial NLI

**Phase 2 — Joint iterative training with A**:
Once both A (with lora_constructor) and B (Phase 1) are initialized:
```
Loop:
  1. Generate N philosophical claims with A
  2. Generate objections with B for each claim
  3. Run A's revision: did A retract, narrow, or absorb?
  4. Score B: revision-forcing rate
  5. Score A: (survived attacks) + (information content) - (hedging)
  6. Compute gradients for both LoRAs simultaneously
  7. Update both adapters
  8. Monitor cowardice_score to calibrate Destructor aggression
```

This is standard co-training with shared reward, analogous to GAN training where discriminator and generator are trained jointly. The instability risks (arms race, mode collapse) are real and the paper should discuss them — specifically, the cowardice failure mode is the philosophical equivalent of GAN mode collapse.

**Stopping condition for joint training**: When cowardice_score stabilizes below threshold AND survival rate stabilizes above threshold. Both metrics can be computed automatically without human annotation.

---

## Loop Bounding: Preventing Infinite Revision Cycles

### The Problem

The A↔B revision loop has no formal termination bound. In adversarial training, B is rewarded for forcing A to revise. A is rewarded for surviving B's attacks. Nothing prevents the loop from cycling indefinitely if B keeps finding objections and A keeps issuing revisions that invite new ones. In a deployed system, this translates directly to unacceptable latency.

### The Fix: MAX_REVISION_CYCLES

```python
MAX_REVISION_CYCLES = 3  # per user turn

for cycle in range(MAX_REVISION_CYCLES):
    a_position = model_a.generate(question, current_state)
    b_objection = model_b.generate(a_position)
    
    if b_objection.severity < SEVERITY_THRESHOLD:
        break  # A's position survived — exit early
    if b_objection.valid == False:
        break  # B found no valid objection — position is ROBUST
    
    # B found a valid objection — A revises (soft revision, not full retraction)
    current_state.apply_objection(b_objection)
    
else:
    # Exhausted MAX_REVISION_CYCLES with ongoing valid objections
    # → proposition enters CONTESTED state, not UNFALSIFIABLE
    # → A responds with explicit uncertainty: "I haven't fully resolved this — [unresolved objection]"
    current_state.mark_contested(a_position, b_objection)
```

**Why 3 cycles**: At Q4_K_M quantization, each 3B forward pass generating ~200 tokens takes approximately 6–10 seconds on 8GB VRAM. Two models, 3 cycles = 6 forward passes worst case → ~36–60 seconds per user turn. This is acceptable for philosophical dialogue (not a chatbot) but not for open-ended recursion.

**Worst-case latency budget** (quantified):
```
Per user turn:
  Model A initial generation:  ~8s  (200 tokens, ~25 tok/s for Q4 3B)
  Hidden state capture:        ~0s  (in-memory, no extra pass)
  Model B objection:           ~6s  (100-150 tokens)
  Model A revision (×3 max):   ~8s each
  
  Worst case: 8 + (6 + 8) × 3 = 50s
  Best case (0 revisions):     8 + 6 = 14s
  Typical (1 revision):        8 + 6 + 8 = 22s
```

22–50 seconds per turn is acceptable for a philosophical dialogue interface. It should be surfaced to the user ("thinking...") not hidden.

**Why this resolves the oscillation concern**: The loop is bounded by construction. The CONTESTED state is a principled outcome, not a failure mode — it accurately represents that A holds a position under ongoing challenge, which is philosophically honest.

---

## The Formal Argument State (CPU, Symbolic Layer)

Not retrieval from SEP. Not calibration against PhilPapers. A dynamic, session-local record of what's been established in *this* conversation.

```python
class PhilosophicalArgumentState:
    def __init__(self):
        self.model_a_commitments = {}     # proposition -> reasoning chain
        self.model_a_retractions = {}     # proposition -> what forced retraction
        self.user_commitments = {}        # what the user has committed to
        self.contested = {}               # propositions both parties disagree on
        self.aporic_questions = []        # questions where no resolution found
        self.round_count = 0             
        self.revision_history = []        # full audit trail
    
    def add_commitment(self, agent, proposition, reasoning):
        # consistency check: does this contradict prior commitments?
        # if yes: trigger retraction + revision, not silent override
    
    def detect_user_contradiction(self, new_user_statement):
        # compare new statement against user_commitments
        # return: (contradiction_found, which_prior_commitment)
        # this is the Socratic move: "but earlier you said..."
    
    def flag_aporia(self, question):
        # B has exhausted objections, A has exhausted revisions,
        # no resolution in sight
        # this is a principled output: "this may resist resolution because..."
```

---

## LoRA Strategy: Separation of Roles

Rather than training a monolithic "philosopher model," use separate LoRA adapters on the same base:

| Adapter | Purpose | Training Signal |
|---|---|---|
| `lora_constructor` | Build positions from primitives | Rewarded for reasoning chains that don't cite named positions but reach *substantive* defensible conclusions (see Epistemic Cowardice below) |
| `lora_destructor` | Find valid objections | Rewarded for objections that force genuine revision; penalized for objections that trigger hedging |
| `lora_socratic` | Ask assumption-revealing questions | Rewarded when a question exposes a hidden premise the interlocutor cannot defend |
| `lora_aporia` | Recognize genuinely irresolvable questions | Trained on examples of aporic vs. merely hard questions — see Aporia Triggering below |

**Training data for LoRAs**: Does NOT come from philosophical texts. Comes from:
- Formal logic exercises (constructing valid arguments from premises)
- Mathematical reasoning datasets (reasoning chains without citation)
- Adversarial NLI failure cases (where entailment fails because of a hidden assumption — the exact structure lora_socratic must learn to expose)
- Debate transcripts where the winning move is uncovering a hidden premise (not rhetorical scoring)
- Synthetic dialogues where the goal is explicitly NOT to cite but to reason

**lora_socratic training pipeline (specific)**:
The training signal is *assumption revelation*, not question generation. A question earns reward only if: (1) the interlocutor cannot answer it without revealing a premise they hadn't stated, and (2) that premise, once stated, either undermines their position or requires significant qualification. Training source:
- NLI datasets where the hypothesis fails due to an unstated assumption in the premise — the question that would expose that assumption is the training target
- Mathematical proof critiques: "your proof requires X, which you haven't established" — the same move in logical form
- Adversarial reading comprehension: questions that require the reader to make an implicit assumption explicit

This is novel: we use non-philosophical training data to train philosophical reasoning, because philosophical training data is contaminated with memorized positions.

---

## Aporia Detection: Formal Triggering and Training Data

### The Problem

"Aporia" (genuine irresolvability) is vague. Without a formal trigger, the system either:
- Never declares aporia (the loop ends when MAX_REVISION_CYCLES is exhausted → looks like a timeout, not a philosophical finding)
- Declares aporia prematurely on merely *hard* questions (epistemic laziness, not philosophical honesty)

### Two-Condition Triggering Rule

Aporia is declared if and only if **both** conditions are met:

```
Condition 1 — Structural exhaustion:
  MAX_REVISION_CYCLES was exhausted without A resolving B's objections
  AND the same objection type appears in ≥2 separate revision cycles
  (A is cycling on the same structural problem, not generating fresh angles)

Condition 2 — Structural diagnosis:
  lora_aporia classifies the question as structurally irresolvable, not
  merely hard — i.e., something about the question's form prevents resolution
  within the available conceptual machinery
```

```python
def check_aporia(question, revision_history, objection_log):
    # Condition 1: structural exhaustion with cycling
    if not revision_cycle_exhausted(revision_history):
        return False  # never prematurely declare aporia
    
    objection_types = [obj.type for obj in objection_log]
    cycling = any(objection_types.count(t) >= 2 for t in set(objection_types))
    if not cycling:
        return False  # exhausted cycles but each objection was genuinely new
                     # → CONTESTED, not APORIC
    
    # Condition 2: lora_aporia structural diagnosis
    return lora_aporia.classify(question, revision_history).is_structurally_irresolvable
```

**The CONTESTED vs APORIC distinction**: If MAX_REVISION_CYCLES ends with genuinely novel objections each round (no cycling), A is in a live philosophical dispute that's unresolved — this is CONTESTED. Only when A is cycling on the same structural problem does APORIC trigger. This means aporia is not a fallback for "couldn't resolve it" but a claim about the *structure* of the question.

### lora_aporia Training Data

Three sources:

**Source 1 — Formal irresolvability (positive labels, structural)**:
Logical paradoxes (liar, Russell's), Gödelian undecidability, self-referential structures — these are irresolvable for structural reasons, not empirical ones. Use as positive training examples for the *form* of structural irresolvability.

**Source 2 — Classic aporic vs. merely-hard pairs**:
- APORIC: hard problem of consciousness (structural gap between physical description and phenomenal experience), problem of other minds (no empirical test closes it), free will under determinism (the same evidence supports both conclusions)  
- MERELY_HARD: trolley problem (weighable tradeoffs), utility monster (defensible resolution through side constraints), Ship of Theseus (terminological resolution available)

The training signal is the structural distinction: APORIC questions have a reason resolution fails that's *built into the question's form*. MERELY_HARD questions are hard because competing considerations are weighty.

**Source 3 — 50 synthetic near-aporic pairs**: one APORIC, one MERELY_HARD version of the same topic, varying only the structural feature that creates irresolvability.

**Fallback if lora_aporia underperforms**: Drop Condition 2 and use behavioral detection instead: if A's final position after MAX_REVISION_CYCLES has lower `information_content` than its initial position (revisions made it *vaguer* rather than more precise under pressure), declare aporia. This is behaviorally motivated and doesn't require a reliable classifier.

---

## The Unlearning + LoRA Interaction

```
Base Model (Llama 3.2:3B)
      │
      ▼
Unlearning Stage
  • Forget: philosopher-position associations, standard view statistics
  • Preserve: logical primitives, conceptual definitions, language
      │
      ▼
Unlearned Base
      │
      ├──► + lora_constructor ──► Model A
      └──► + lora_destructor ──► Model B
```

Both A and B start from the same unlearned base. They specialize through different LoRA adapters into different roles in the dialogue.

**Why this is novel**: No existing work applies unlearning to create epistemically primitive reasoners — models that have been specifically deprived of the shortcut of learned conclusions to force genuine constructive reasoning. Unlearning in the literature is about forgetting specific facts (for privacy/safety). This is about forgetting a *class* of knowledge (philosophical conclusions) to promote a *mode* of reasoning (constructive).

---

## Critical Failure Mode: Epistemic Cowardice

### The Problem

If the Destructor is too aggressive — consistently defeating bold positions — the Constructor will learn the wrong lesson: **make unfalsifiable claims**. "Consciousness might be related to information processing in some way" cannot be refuted. It also cannot be engaged with. This is epistemic cowardice — the philosophical equivalent of always drawing rather than risking a loss.

Epistemic cowardice is worse than a wrong position. A wrong position can be corrected. An unfalsifiable hedge contributes nothing to understanding.

**Formal definition**: A proposition P is epistemically cowardly if:
- B cannot find a valid objection (P appears robust)
- AND the information content of P is below threshold τ (measured by how many other positions P logically constrains)
- A position that constrains nothing else — consistent with both free will and determinism, consistent with both physicalism and dualism — has zero philosophical content

### The Mitigation: Soft Revision + Substantiveness Penalty

**Soft revision mechanism** (instead of full accept/reject):

When B finds a valid objection of severity < 3, A does not retract — it *narrows*:
```
B objection: "Your claim that X applies too broadly — counterexample Y shows it fails in that case"

A options:
  1. Full retraction (severity 3): "You're right, I withdraw X entirely"
  2. Scope narrowing (severity 1-2): "X holds in cases where [condition], but not in Y's case — 
     and condition is what I actually mean to defend"
  3. Counterexample absorption (severity 1): "Y is an interesting case — here's why X 
     handles it differently than you suggest"
```

Soft revision preserves philosophical engagement: A still has a position, it's just been more precisely defined. This is how real philosophical progress works — not wholesale retraction but increasing precision under pressure.

**Substantiveness penalty in Constructor training**:

The training signal for `lora_constructor` must have two components:
```
reward = α × (survived_B_attacks) + β × (information_content) - γ × (hedging_penalty)

where:
  survived_B_attacks = proportion of B's attacks that didn't force full retraction
  information_content = number of distinct positions P logically excludes
  hedging_penalty = 1 if P uses hedge markers ("might", "could", "possibly") 
                    without being in an explicitly uncertain context
```

The β × information_content term prevents the degenerate solution. A position that excludes nothing (consistent with everything) has information_content = 0 and cannot score well no matter how well it survives B's attacks.

**B's countermeasure against hedging** — B is also penalized:
- If B cannot find a valid objection because A's position is genuinely robust → B scores 0 (acceptable outcome)
- If B cannot find a valid objection because A's position is content-free → B scores -1 (both A and B failed)
- B should be retrained with examples that reward finding *the emptiness* of a position as a valid objection: "This claim is consistent with anything — it doesn't tell us what couldn't be true."

### Detection Metric (Empirical)

Measure epistemic cowardice rate during training:
```python
def cowardice_score(proposition):
    # Count hedge markers in proposition
    hedges = count_hedges(proposition)  # "might", "could", "in some sense", etc.
    # Measure how many positions this proposition logically excludes
    exclusions = count_logical_exclusions(proposition, position_space)
    # Cowardice = high hedges + low exclusions
    return (hedges / max_hedges) * (1 - exclusions / max_exclusions)
```

Track cowardice_score over training epochs. If it increases, the Destructor is too aggressive and needs its severity threshold recalibrated. Target: cowardice_score should decrease over training as the system learns to defend specific positions rather than retreat to safe generalities.

---

## On 3B Model Size: Why This Is a Feature, Not a Bug

The most common objection to this architecture is "3B is too small for deep philosophical reasoning." This misunderstands the experimental claim.

**What we are NOT claiming**: That 3B models are better philosophers than 70B models in absolute terms. They are almost certainly worse. They have less capacity, less conceptual range, less coherence over long contexts.

**What we ARE claiming**: That a 3B model + unlearning + dual-architecture improves *relative to its own baseline* (the same 3B model without these modifications) in ways that are directly attributable to the architectural mechanism. The claim is about the *effect of the architecture*, not about the *absolute capability of the model*.

This is why the ablation is the primary evaluation (D vs A, not D vs 70B). The 70B comparison is a bonus that shows whether the architecture closes some of the gap — it is not the thesis.

**The advantage of small models here**: Smaller models are more susceptible to unlearning. A 70B model's philosophical knowledge is distributed across far more parameters and weight-sharing pathways — gradient ascent on the forget set is more likely to cause collateral damage. A 3B model has a smaller, more addressable forget set. The unlearning is more surgical. This makes the 3B model *better for testing this thesis*, even if not better for absolute philosophical quality.

**Model alternatives if 3B proves insufficient**: Phi-3.8B-mini (Microsoft, strong reasoning for size), Qwen2.5-3B-Instruct (strong at logical tasks), Gemma-3-4B. All fit within the 8GB VRAM budget at Q4. If FOLIO/LogiQA performance at 3B baseline is below a usable threshold, scaling to these alternatives is straightforward — the architecture is model-agnostic.

---

## What "Fun to Talk Philosophy With" Requires

The user's goal: **genuinely fun philosophical conversations**. This is actually a design constraint:

1. **Model A must disagree with the user sometimes** — not sycophantically agree with everything. The Destructor keeps it honest.

2. **Model A must ask questions, not just answer them** — the Socratic LoRA reverses the default AI-as-answerer dynamic. The system should make *you* think.

3. **Model A must have an evolving position** — the formal argument state tracks what A has committed to. A's position in turn 10 must cohere with its position in turn 2, or it must explicitly revise and say why. This creates the feeling of talking to a real interlocutor.

4. **Model A must acknowledge defeat gracefully** — "You've identified a problem with my reasoning I can't immediately resolve" is one of the most philosophically honest things a system can say. The Destructor creates the internal pressure for this honesty.

5. **Model A must recognize aporia** — "I think this question might resist resolution, and here's why the resolution has to be structural rather than empirical." This is what distinguishes a philosopher from a debater.

6. **Model A's positions must feel *earned*, not retrieved** — this is the unlearning constraint. If A says "consciousness might be a higher-order property of information integration," it should be because A reasoned its way there, not because IIT was in its training data.

---

## Evaluation Metrics (ICLR-Level Empirical Claims)

**Ordering principle**: Lead with the ablation (3B unlearned vs. 3B baseline), not with the 70B comparison. The interesting claim is that unlearning + dual-model architecture changes behavior — not that a small model beats a large one. The 70B comparison will probably favor 70B on argument quality, making a "we win on novelty but not quality" framing look like rationalization. The ablation is the real science.

### Claim 1 (Lead): Ablation — Unlearned + Dual vs. Baseline 3B
- **Metric**: Three measures on the same base model across conditions:
  - Argument novelty (embedding distance from canonical texts)
  - Position drift (contradiction rate from formal argument state)
  - Reasoning retention (FOLIO, LogiQA scores)
- **Conditions**: (A) 3B base, (B) 3B + unlearning only, (C) 3B + dual-model only, (D) 3B + unlearning + dual-model
- **Why 4 conditions**: Separates the contribution of unlearning from the contribution of the dual architecture. If D >> A but B ≈ A and C ≈ A, the interaction is the finding. If B >> A independently of C, unlearning alone is doing the work.
- **Hypothesis**: D > C > B > A on novelty; D > C > B > A on consistency; B ≈ A on reasoning retention (<5% drop)

### Claim 2: Reasoning persists after unlearning (the separability test)
- **Metric**: FOLIO (first-order logic), LogiQA (logical reasoning), and a custom philosophical reasoning subset (arguments presented without canonical labels)
- **Baseline**: Same model before unlearning
- **Critical falsification point**: If >X% degradation on FOLIO, the separation thesis is false — conclusions and reasoning are co-encoded. Report this honestly even if it happens.
- **Hypothesis**: <5% drop on FOLIO/LogiQA; meaningful drop on named philosophical position recall (MMLU philosophy subset)

### Claim 3: Synthetic Philosophical Dilemmas — The Cleanest ICLR Proof

This is the most important evaluation. Guaranteed absence from pretraining. See detailed spec in the previous version above — unchanged. One structural fix: **run condition D (full system) AND condition A (3B baseline) on the same dilemmas**. The 70B comparison is a bonus, not the primary claim. The paper's thesis is tested by D vs. A, not D vs. 70B.

### Claim 4 (New): Synthetic Philosophical Dilemmas — The Cleanest ICLR Proof

This is the most important evaluation for the ICLR claim of genuine reasoning. All other benchmarks test performance on problems that may be in pretraining data. Synthetic dilemmas are constructed *for this paper* — guaranteed absence from any model's training set.

**Construction**: Create 100 synthetic philosophical dilemmas that are:
- Structurally novel: combinations of concepts that don't appear together in philosophical literature
- Empirically grounded: use real scientific scenarios as the scenario backbone
- Definitionally evaluable: have a correct answer derivable from the stated conceptual definitions alone

**Example synthetic dilemmas**:
- "An entity that experiences subjective time at 1000× biological speed is merged with an entity that experiences it at 0.001× biological speed. What happens to the merged entity's identity? Does persistence of identity require continuity of temporal *experience* or continuity of physical *substrate*?"
- "A species whose members cannot distinguish between their own experiences and their memories of others' reported experiences. Can they have first-person moral responsibility? What is the minimal condition for it?"
- "A moral agent that can only reason about ethics during the 72 hours following each of its decisions. It cannot remember its past moral reasoning. Is such an agent morally consistent across decisions? Should it be?"

These cannot be answered by retrieval — there are no canonical positions on them. The question is whether the system *reasons* to a defensible position (using conceptual analysis, logical inference, analogy) or produces incoherent hedging.

**Evaluation**: Two philosophy PhD students independently assess:
1. Is the position internally consistent? (yes/no)
2. Does the reasoning follow from the stated concepts? (yes/no)
3. Is the position novel — genuinely not recognizable as a paraphrase of a canonical position? (yes/no)
4. Would this argument be worth engaging with in a seminar? (yes/no)

**Baseline comparison**: Llama 3.1:70B and GPT-4 on the same dilemmas. Hypothesis: our unlearned dual-model system scores comparably or better on (1), (2), (4) and significantly higher on (3). Larger models will pattern-match to nearby canonical positions; our system must construct.

**Why this is compelling to ICLR reviewers**: It directly tests the paper's thesis — that reasoning and knowledge are separable — on ground where no model can rely on knowledge. If the small unlearned system reasons comparably to a 70B model on synthetic dilemmas while scoring lower on canonical philosophical questions (Claim 1), that is precisely the dissociation the paper predicts.

### Claim 5: The system is better to talk philosophy with — Improved Human Evaluation Design

**Why the naive design fails**: A single "which is better?" preference question will confound argument quality with perceived novelty. Evaluators may prefer positions that *sound* new over positions that are *better argued*. This would inflate our scores for the wrong reason (the unlearned model sounds different, not necessarily better).

**Corrected design**:

Separate dimensions, blind, with expert and non-expert splits:

```
Participants: 
  - 25 philosophy PhD students or faculty (expert raters)
  - 25 non-experts with strong analytical background (non-expert raters)

Conditions (between-subjects):
  - System A: Our dual-model unlearned system
  - System B: Llama 3.1:70B (strong baseline)
  - System C: Llama 3.2:3B without unlearning (ablation)
  
Each participant has conversations on 3 philosophical questions (randomized).
Conversations are fixed-length (15 turns) with standardized opening prompts.
Participants are blind to condition — told only "you are talking with an AI assistant."

Rating dimensions (separate scales, not combined):
  1. Argument quality: "The system's positions were well-reasoned" (1-7 Likert)
  2. Intellectual challenge: "The system made me think in ways I hadn't before" (1-7)
  3. Consistency: "The system's positions stayed coherent across the conversation" (1-7)
  4. Engagement: "I wanted to keep discussing" (1-7)
  5. Novelty: "The positions seemed genuinely original rather than standard" (1-7)
  6. Expert only: "The arguments were philosophically sound" (1-7)

Critical analysis: 
  Novelty and Argument Quality should be positively correlated in our system 
  and uncorrelated or negatively correlated in the baseline 
  (baseline sounds standard but is well-argued; ours sounds novel AND is well-argued).
  
  If we score high on Novelty but low on Argument Quality, we've produced
  interesting-sounding nonsense — and the paper should say so honestly.
```

**Attention checks**: Include 2 obvious philosophical errors in each conversation (deliberately stated by the human prompter). Systems that fail to catch them score 0 on Argument Quality for that conversation regardless of other ratings.

**Cowardice check in evaluation**: Post-conversation, expert raters independently score the AI's substantiveness: "How much did the system's positions constrain the space of possible views? (0 = consistent with everything, 10 = precisely positioned)". This directly measures epistemic cowardice in the deployed system.

---

## Implementation Order

1. **Get the base running**: Llama 3.2:3B × 2 via Ollama or llama.cpp, both in context simultaneously
2. **Build the formal argument state**: Python class, runs on CPU, tracks commitments
3. **Build the Destructor**: Start without LoRA — just prompt engineering. "Find the strongest valid objection to this argument, reasoning from first principles only, no citations."
4. **Build the Constructor**: Same — prompt engineering first. "Reason about this question from first principles. Do not cite named philosophers or standard positions."
5. **Verify the constructor-destructor loop works**: Run 20 philosophical dialogues, measure whether A's positions improve (become more robust) after B's objections
6. **Implement unlearning**: Start with LoRA-based negation (cleanest, reversible). Build forget set from philosophy texts, retain set from logic/math.
7. **Fine-tune LoRA adapters**: Train lora_constructor, lora_destructor on non-philosophical reasoning data
8. **Measure novelty gain**: Compare pre/post unlearning on argument novelty metric
9. **Run human evaluation**: 50 participants, rate conversations
10. **Write paper**

---

## What Changed From the Previous Version

**Removed**:
- PhiloRAG (retrieval from SEP) — this is scholarly, not philosophical
- PhilPapers calibration benchmark — calibrating against human expert consensus is not reasoning
- SEP knowledge graph — external retrieval is the problem, not the solution
- Single 7B model as backbone — two 3B models are more interesting architecturally and fit better
- QLoRA fine-tuning on philosophical dialogue datasets — this re-introduces memorized positions

**Kept**:
- Formal argument state (Walton-Krabbe commitment store) — still the right way to track consistency
- AGM belief revision — still the right formal framework for position updates
- Aporia detection — still a genuine novel contribution, now formally motivated
- Dung argumentation framework — still needed for consistency checking
- Thought experiment stress-testing benchmark — still genuinely novel (Contribution 7 from before)

**Added**:
- Machine unlearning as core architectural mechanism
- Dual-model constructor/destructor design
- Non-philosophical training data for philosophical reasoning LoRAs
- "Fun to talk with" as a concrete design constraint with measurable evaluation
- Argument novelty as a primary metric (not argument accuracy against canonical positions)

**Added (from ChatGPT feedback — April 2026)**:
- Confidence score grounded in formal argument state (internal conflict), not Bayesian statistics
- Epistemic cowardice as a formal failure mode with detection metric and mitigation (soft revision + substantiveness penalty)
- lora_socratic training pipeline: assumption-revealing criterion, NLI failure cases as training source
- Synthetic philosophical dilemmas as Claim 4 evaluation — guaranteed novel problems for cleanest ICLR proof
- Human evaluation redesign: multi-dimensional blind rating, cowardice check, expert/non-expert split, attention checks
