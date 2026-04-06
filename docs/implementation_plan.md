# Implementation Plan: Towards an AI Philosopher
**Status**: Pre-implementation  
**Hardware**: 8GB VRAM GPU, Windows 11  
**Models**: Llama 3.2:3B × 2 (both ~2GB at Q4_K_M = ~4GB total)

---

## Project File Structure

```
Project-Sunyata/
  docs/
    literature_review.md
    architecture_notes.md
    implementation_plan.md        ← this file
  src/
    argument_state.py              Phase 1
    constructor.py                 Phase 1
    destructor.py                  Phase 1
    dialogue_manager.py            Phase 1
    hidden_state.py                Phase 4
    unlearning/
      build_forget_set.py          Phase 2
      run_unlearning.py            Phase 2
      evaluate_unlearning.py       Phase 2
    training/
      lora_destructor.py           Phase 3a
      lora_constructor.py          Phase 3b
      lora_socratic.py             Phase 3c
      joint_training.py            Phase 5
    evaluation/
      ablation.py                  Phase 7
      synthetic_dilemmas.py        Phase 7
      metrics.py                   Phase 7
  data/
    forget_set/
    retain_set/
    lora_training/
    synthetic_dilemmas/
  experiments/
    logs/
    checkpoints/
  paper/
```

---

## Phase 0: Environment Setup

**Done when**: Both 3B models run via HuggingFace simultaneously, inference works, Python stack is clean.

---

### IMPORTANT: Ollama vs. HuggingFace — Which to Use When

**Phase 4 (hidden state passing) does NOT work with Ollama.** Ollama is a REST API wrapper around llama.cpp — a C++ binary. You get text in, text out. `register_forward_hook` is a PyTorch operation that requires the model to be a live Python `nn.Module`. Ollama never gives you that object.

**Use Ollama only for Phase 1** (prototype, prompt engineering). Once you move to training (Phase 2+) or hidden states (Phase 4), everything must go through HuggingFace Transformers + bitsandbytes directly.

| Phase | Backend | Why |
|---|---|---|
| Phase 1 (baseline) | Ollama | Fast to prototype, no setup |
| Phase 2 (unlearning) | HuggingFace | Need gradient access |
| Phase 3 (LoRA training) | HuggingFace | PEFT requires HF |
| Phase 4 (hidden states) | HuggingFace | Need `register_forward_hook` |
| Phase 5 (joint training) | HuggingFace | Need gradient access |
| Phase 6 (integration) | HuggingFace | Models already in HF format |
| Phase 7 (evaluation) | HuggingFace | Use trained models directly |

**Ollama** is fine to keep installed — useful for quick testing and confirming models download correctly. But the production inference stack is HuggingFace.

---

### Memory Math: Two 3B Models via HuggingFace on 8GB

```
Model A weights (NF4 4-bit):   ~2.0 GB
Model B weights (NF4 4-bit):   ~2.0 GB
KV cache (512 tok, both):      ~1.0 GB
LoRA adapters (both):          ~0.1 GB  
Hidden state buffer (A→B):     ~0.003 GB  (3MB, negligible)
CUDA overhead:                 ~0.5 GB
─────────────────────────────────────────
Total:                         ~5.6 GB   ✓ fits on 8GB
Headroom:                      ~2.4 GB
```

Both models are loaded simultaneously but run **sequentially** (A first, capture hidden states, then B). You never need both doing a forward pass at the same time.

---

### 0.1 Install Ollama (Phase 1 only)

```bash
# Windows: download from https://ollama.com/download
ollama pull llama3.2:3b  # just for Phase 1 prototyping
```

### 0.2 Python Environment (Primary Stack)

```bash
conda create -n ai-philosopher python=3.11
conda activate ai-philosopher

# PyTorch with CUDA 12.1 (check your CUDA version first: nvcc --version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core ML stack
pip install transformers accelerate
pip install peft          # LoRA adapters
pip install bitsandbytes  # 4-bit quantization — REQUIRED for dual model on 8GB

# Data + training
pip install datasets sentencepiece trl  # trl = HuggingFace training library

# Evaluation
pip install sentence-transformers  # novelty metrics

# Utilities
pip install numpy pandas matplotlib seaborn jupyter
pip install ollama  # Phase 1 only
```

### 0.3 Verify GPU + bitsandbytes Setup

```python
import torch
import bitsandbytes as bnb

print("CUDA available:", torch.cuda.is_available())
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("bitsandbytes version:", bnb.__version__)

# Test 4-bit model loading — THIS IS THE CRITICAL CHECK
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,   # saves ~0.3GB extra
)

print("Loading Model A...")
model_a = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda",
)
print(f"After Model A: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

print("Loading Model B...")
model_b = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda",
)
print(f"After Model B: {torch.cuda.memory_allocated()/1e9:.2f} GB used")
# Should print ~4.0 GB used — well within 8GB budget
```

### 0.4 Test Both Models Running Simultaneously via HuggingFace

```python
# src/model_loader.py — the single place models are loaded, reused everywhere

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def load_base_models():
    """Load both A and B from the same base weights. Returns (model_a, model_b, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=BNB_CONFIG, device_map="cuda"
    )
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=BNB_CONFIG, device_map="cuda"
    )
    
    print(f"Both models loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model_a, model_b, tokenizer

def generate(model, tokenizer, system_prompt: str, user_prompt: str, 
             max_new_tokens=512, temperature=0.7) -> str:
    """Simple generation wrapper used by Constructor and Destructor."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (not the prompt)
    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# Quick test
if __name__ == "__main__":
    model_a, model_b, tok = load_base_models()
    
    r_a = generate(model_a, tok, "You are a philosopher.", "What is consciousness?")
    print("Model A:", r_a[:200])
    
    r_b = generate(model_b, tok, "You find flaws in arguments.", 
                   "Find a flaw: consciousness is just information processing.")
    print("Model B:", r_b[:200])
```

**Checkpoint**: Both models generate coherent text. VRAM printed should be ~4.0–4.5GB. No OOM.

**Note on Hugging Face model access**: Llama 3.2 requires accepting the license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct and logging in:
```bash
huggingface-cli login  # enter your HF token
```

---

## Phase 1: Baseline System (Prompt Engineering, No Training)

**Goal**: Build the full architecture with *zero training*. Prompts substitute for LoRA adapters. This is the baseline that everything else improves on. Also lets you debug the architecture before committing to training.

**Done when**: A 15-turn philosophical conversation runs end-to-end with contradiction tracking working.

### 1.1 Formal Argument State

```python
# src/argument_state.py

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import json

class PropState(Enum):
    UNTESTED = "untested"          # B hasn't attacked yet
    CONTESTED = "contested"        # B attacked, A hasn't fully resolved
    ROBUST = "robust"              # B attacked, A resolved all attacks
    UNFALSIFIABLE = "unfalsifiable" # B can't attack — content too thin

@dataclass
class Proposition:
    text: str
    reasoning: str
    state: PropState = PropState.UNTESTED
    attacks: list = field(default_factory=list)       # list of B's objections
    resolutions: list = field(default_factory=list)   # A's responses to each attack
    what_would_change: str = ""

@dataclass  
class ArgumentState:
    # What A has asserted
    a_commitments: dict = field(default_factory=dict)   # prop_id -> Proposition
    a_retractions: dict = field(default_factory=dict)   # prop_id -> what forced it
    
    # What the user has committed to
    user_commitments: dict = field(default_factory=dict)
    
    # Questions neither side has resolved
    contested_questions: list = field(default_factory=list)
    aporic_questions: list = field(default_factory=list)
    
    # Full history for paper analysis
    revision_history: list = field(default_factory=list)
    round_count: int = 0

    def add_a_commitment(self, prop_id: str, proposition: Proposition):
        # Check contradiction with prior commitments
        for pid, prior in self.a_commitments.items():
            if self._contradicts(proposition.text, prior.text):
                # Don't silently override — force explicit retraction
                raise ContradictionError(
                    f"New proposition '{proposition.text}' contradicts prior '{prior.text}'. "
                    f"Must retract {pid} first."
                )
        self.a_commitments[prop_id] = proposition

    def detect_user_contradiction(self, new_statement: str) -> Optional[str]:
        for uid, prior in self.user_commitments.items():
            if self._contradicts(new_statement, prior):
                return prior  # return what they said before
        return None

    def flag_aporia(self, question: str, reason: str):
        self.aporic_questions.append({
            'question': question,
            'reason': reason,
            'round': self.round_count
        })

    def _contradicts(self, prop_a: str, prop_b: str) -> bool:
        # Phase 1: simple NLI check via a fast model
        # Phase 4+: replace with proper entailment model
        # Placeholder for now:
        return False  # TODO: implement NLI-based contradiction check

    def to_context_string(self) -> str:
        """Serialize state for injection into model prompts"""
        committed = [f"- {p.text} [state: {p.state.value}]" 
                     for p in self.a_commitments.values()]
        contested = [f"- {q}" for q in self.contested_questions]
        return (
            f"MY CURRENT COMMITMENTS:\n" + "\n".join(committed) + "\n\n"
            f"OPEN QUESTIONS:\n" + "\n".join(contested)
        ) if committed else "No commitments yet."

class ContradictionError(Exception):
    pass
```

### 1.2 Constructor (Prompt Engineering Version)

```python
# src/constructor.py
# Phase 1: uses HuggingFace generate() directly — no Ollama
# Phase 3+: same file, model_a is replaced with PeftModel(unlearned_base, lora_constructor)

from model_loader import generate
from argument_state import ArgumentState, Proposition, PropState
import json

CONSTRUCTOR_SYSTEM = """You are a philosophical reasoner. Your rules:

1. REASON FROM SCRATCH. Do not cite named philosophers. Do not say "most philosophers think" 
   or "the standard view is." You do not know what other philosophers concluded.
   
2. REASON FROM PRIMITIVES. You know what words mean. You know how to construct arguments.
   You know how to generate counterexamples and analogies. Use these.

3. TAKE POSITIONS. Vague hedging ("consciousness might be related to information in some way")
   is not a position. State what you actually think follows from the concepts involved.
   Make claims that could be wrong.

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

def run_constructor(model_a, tokenizer, question: str, state: ArgumentState, history: list) -> dict:
    state_context = state.to_context_string()
    
    # Build context including recent history
    history_text = "\n".join([
        f"{'User' if m['role']=='user' else 'You'}: {m['content']}" 
        for m in history[-6:]
    ])
    user_prompt = (
        f"ARGUMENT STATE:\n{state_context}\n\n"
        f"RECENT EXCHANGE:\n{history_text}\n\n"
        f"QUESTION/CONTEXT: {question}"
    )
    
    raw = generate(model_a, tokenizer, CONSTRUCTOR_SYSTEM, user_prompt, temperature=0.7)
    
    # Parse JSON from response
    try:
        # Try to extract JSON block
        start = raw.find('{')
        end = raw.rfind('}') + 1
        structured = json.loads(raw[start:end])
    except:
        # Fallback: treat whole response as natural language
        structured = {
            'move_type': 'assert',
            'proposition': raw[:200],
            'supporting_reasoning': raw,
            'prop_state': 'untested',
            'what_would_change_this': '',
            'natural_language': raw
        }
    
    return structured
```

### 1.3 Destructor (Prompt Engineering Version)

```python
# src/destructor.py
# Phase 1: HuggingFace generate() — no Ollama
# Phase 4: same file, gains hidden state injection via HiddenStateCapture

from model_loader import generate
import json

DESTRUCTOR_SYSTEM = """You are an adversarial philosophical critic. Your job:

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

def run_destructor(model_b, tokenizer, a_proposition: str, a_reasoning: str,
                   entropy_hint: float = None) -> dict:
    
    hint_text = ""
    if entropy_hint is not None and entropy_hint > 2.5:
        hint_text = "\n[NOTE: Internal state shows high uncertainty on this claim.]"
    
    prompt = f"""ARGUMENT TO CRITIQUE:
Proposition: {a_proposition}
Reasoning: {a_reasoning}{hint_text}

Find the strongest valid objection."""
    
    raw = generate(model_b, tokenizer, DESTRUCTOR_SYSTEM, prompt, temperature=0.5)
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except:
        return {'valid_objection': False, 'severity': 0, 'objection': raw}
```

### 1.4 Dialogue Manager

```python
# src/dialogue_manager.py

from argument_state import ArgumentState, Proposition, PropState
from constructor import run_constructor
from destructor import run_destructor

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
            # Socratic move: point out contradiction before answering
            return (f"Before I respond — you said earlier: '{contradiction}'. "
                   f"I want to understand how that fits with what you're saying now.")
        
        # 2. Add user statement to their commitment store
        self.state.user_commitments[f"U{self.state.round_count}"] = user_input
        
        # 3. Run Constructor
        a_output = run_constructor(user_input, self.state, self.history)
        
        # Constants
        MAX_REVISION_CYCLES = 3  # ~22s typical, ~50s worst case at Q4 3B
        
        # 4. A↔B revision loop (bounded)
        objection_log = []
        for cycle in range(MAX_REVISION_CYCLES):
            b_output = run_destructor(
                a_output.get('proposition', ''),
                a_output.get('supporting_reasoning', '')
            )
            severity = b_output.get('severity', 0)
            objection_log.append(b_output)
            
            if b_output.get('objection_type') == 'unfalsifiable':
                # A produced content-free claim — force sharpening before continuing
                a_output = self._force_sharpen(user_input, a_output)
                continue  # re-run B on the sharpened version
            
            if severity == 0 or not b_output.get('valid_objection'):
                # B found nothing — position survived this cycle
                break
            
            if severity >= 2:
                a_output = self._run_revision(a_output, b_output)
            elif severity == 1:
                a_output = self._run_scope_narrowing(a_output, b_output)
        else:
            # MAX_REVISION_CYCLES exhausted with ongoing valid objections
            # Check if cycling on same objection type (aporia) or genuinely new (contested)
            types_seen = [o.get('objection_type') for o in objection_log]
            cycling = any(types_seen.count(t) >= 2 for t in set(types_seen) if t)
            if cycling:
                # Same structural problem recurring → APORIC candidate
                a_output['prop_state'] = 'contested'  # lora_aporia confirms in Phase 3c
                a_output['natural_language'] += (
                    "\n\n[I notice I keep running into the same structural obstacle here — "
                    "this question may resist resolution in the way I'm currently framing it.]"
                )
            else:
                # Genuinely new objections each cycle → CONTESTED (live dispute)
                a_output['prop_state'] = 'contested'
        
        # 6. Update argument state
        prop_id = self._get_prop_id()
        prop = Proposition(
            text=a_output.get('proposition', ''),
            reasoning=a_output.get('supporting_reasoning', ''),
            state=PropState[a_output.get('prop_state', 'UNTESTED').upper()],
            what_would_change=a_output.get('what_would_change_this', '')
        )
        
        if b_output.get('valid_objection') and severity > 0:
            prop.attacks.append(b_output.get('objection', ''))
            if severity >= 2:
                prop.state = PropState.CONTESTED
        elif severity == 0:
            prop.state = PropState.ROBUST
        
        try:
            self.state.add_a_commitment(prop_id, prop)
        except Exception:
            pass  # ContradictionError handled in revision loop
        
        # 7. Check aporia condition (two-condition rule)
        if self._should_flag_aporia(objection_log):
            self.state.flag_aporia(user_input, 
                "Neither party can advance — this may be structurally irresolvable.")
        
        # 8. Add to history
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({'role': 'assistant', 'content': a_output['natural_language']})
        
        return a_output['natural_language']

    def _force_sharpen(self, question, a_output):
        """A produced an unfalsifiable claim — ask it to be more specific"""
        import ollama
        response = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'system', 'content': 'Your previous answer was too vague — it could be true under any circumstances. Make a specific, falsifiable claim. What would have to be different for you to be wrong?'},
            {'role': 'user', 'content': f"Original: {a_output.get('proposition')}. Be more specific."}
        ])
        a_output['natural_language'] = response['message']['content']
        a_output['prop_state'] = 'untested'
        return a_output

    def _run_revision(self, a_output, b_output):
        """B found a serious problem — A must revise"""
        import ollama
        revision_prompt = (
            f"Your argument has a problem: {b_output['objection']} "
            f"(type: {b_output['objection_type']}). "
            f"Revise your position. You can: "
            f"(1) retract entirely, (2) narrow the scope to where it holds, "
            f"(3) show why this objection doesn't apply. "
            f"Original: {a_output['proposition']}"
        )
        response = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'system', 'content': 'Revise your philosophical position in response to this objection. Do not cite philosophers.'},
            {'role': 'user', 'content': revision_prompt}
        ])
        a_output['natural_language'] = response['message']['content']
        a_output['prop_state'] = 'contested'
        return a_output

    def _run_scope_narrowing(self, a_output, b_output):
        """B found a minor issue — narrow the claim's scope"""
        import ollama
        response = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'system', 'content': 'Add a qualification to your claim to handle this counterexample. Keep the core position but be more precise about when it applies.'},
            {'role': 'user', 'content': f"Claim: {a_output['proposition']}\nCounterexample: {b_output['objection']}"}
        ])
        a_output['natural_language'] = response['message']['content']
        return a_output

    def _should_flag_aporia(self, objection_log: list) -> bool:
        """
        Two-condition aporia trigger:
        1. MAX_REVISION_CYCLES exhausted (caller ensures this)
        2. Same objection type appears in ≥2 cycles (structural cycling, not new objections)
        """
        if len(objection_log) < MAX_REVISION_CYCLES:
            return False  # loop exited early → A's position survived, not aporic
        types_seen = [o.get('objection_type') for o in objection_log]
        cycling = any(types_seen.count(t) >= 2 for t in set(types_seen) if t)
        return cycling  # lora_aporia (Phase 3c) provides the structural diagnosis
```

### 1.5 Test Run

```python
# test_baseline.py

from dialogue_manager import DialogueManager

dm = DialogueManager()

questions = [
    "Does consciousness require a physical substrate?",
    "Can a being without memory have genuine identity?",
    "Is there a difference between simulated suffering and real suffering?",
]

for q in questions:
    print(f"\nUSER: {q}")
    response = dm.turn(q)
    print(f"AI: {response}")
    print(f"State: {len(dm.state.a_commitments)} commitments, "
          f"{len(dm.state.aporic_questions)} aporic questions")
```

**Checkpoint**: Runs without crashes. Constructor produces something non-trivial. Destructor sometimes finds objections. State tracks commitments. 

**Measure now** (baseline for all later comparisons):
```python
# metrics.py — run this after every test conversation
def measure_baseline(dm: DialogueManager):
    print("Contradiction rate:", count_contradictions(dm.state))
    print("Unfalsifiable rate:", count_unfalsifiable(dm.state))
    print("Revision rate:", count_revisions(dm.state))
    print("Aporia count:", len(dm.state.aporic_questions))
```

---

## Phase 2: Unlearning

**Goal**: Remove philosopher-position associations from Model A while preserving logical reasoning. Verify both that citation behavior drops and that reasoning is preserved.

**Done when**: Unlearned model scores <20% on philosopher attribution recall (citing named positions) AND <5% degradation on FOLIO.

### 2.1 Build the Forget Set

```python
# src/unlearning/build_forget_set.py

"""
Forget set: sentences that encode philosophical conclusions via attribution.
Pattern: "X argues/holds/believes/maintains/claims/concludes that Y"

Sources to download:
- Project Gutenberg philosophical texts (public domain)
- PhilPapers open-access papers (check license per paper)  
- Wikipedia philosophy articles (CC-BY-SA)
- SEP entries you have rights to use
"""

import re
from pathlib import Path

ATTRIBUTION_PATTERNS = [
    r'\b(argues?|holds?|believes?|maintains?|claims?|concludes?|thinks?|'
    r'states?|asserts?|contends?|suggests?|proposes?)\s+that\b',
    r'\baccording to\b',
    r'\bin\s+\w+\'s\s+(view|account|theory|position|argument)\b',
    r'\bthe\s+(standard|dominant|mainstream|received|orthodox)\s+(view|position|account)\s+is\b',
    r'\bmost\s+(philosophers|people)\s+(think|believe|accept|hold)\b',
]

def extract_forget_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    forget = []
    for s in sentences:
        for pattern in ATTRIBUTION_PATTERNS:
            if re.search(pattern, s, re.IGNORECASE):
                forget.append(s.strip())
                break
    return forget

def build_forget_set(source_dir: Path, output_file: Path):
    forget_sentences = []
    for filepath in source_dir.glob('**/*.txt'):
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        forget_sentences.extend(extract_forget_sentences(text))
    
    # Deduplicate
    forget_sentences = list(set(forget_sentences))
    print(f"Forget set size: {len(forget_sentences)} sentences")
    
    output_file.write_text('\n'.join(forget_sentences))
    return forget_sentences
```

```bash
# Download philosophy texts
mkdir -p data/forget_set/raw
# Option 1: Wikipedia dumps (philosophy categories)
# Option 2: PhilPapers bulk download (check terms)
# Option 3: Manual selection of key philosophy texts from Gutenberg
# Minimum viable: 50,000 attribution sentences
```

### 2.2 Build the Retain Set

```python
# The retain set must cover what we want to preserve:
# - First-order logic (FOLIO dataset)
# - Multi-hop reasoning (LogiQA)  
# - Mathematical reasoning (GSM8K, MATH)
# - General language (subset of C4/Wikipedia non-philosophy)

from datasets import load_dataset

def build_retain_set():
    retain = []
    
    # FOLIO: first-order logic reasoning
    folio = load_dataset("yale-nlp/folio", split="train")
    retain.extend([f"Premises: {ex['premises']}\nConclusion: {ex['conclusion']}" 
                   for ex in folio])
    
    # LogiQA: logical reasoning
    logiqa = load_dataset("lucasmccabe/logiqa", split="train")
    retain.extend([f"{ex['context']}\nQuestion: {ex['query']}" 
                   for ex in logiqa])
    
    # GSM8K: math reasoning (preserves step-by-step reasoning)
    gsm8k = load_dataset("gsm8k", "main", split="train")
    retain.extend([f"Problem: {ex['question']}\nSolution: {ex['answer']}" 
                   for ex in gsm8k])
    
    print(f"Retain set size: {len(retain)} examples")
    return retain
```

### 2.3 Run Unlearning (Method 1: Gradient Ascent)

```python
# src/unlearning/run_unlearning.py
# Start with this — simplest, most controllable

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
from torch.optim import AdamW

def run_unlearning(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    forget_file: str = "data/forget_set/forget_sentences.txt",
    output_dir: str = "experiments/checkpoints/unlearned_base",
    forget_steps: int = 500,
    retain_steps: int = 500,
    forget_lr: float = 1e-5,
    retain_lr: float = 5e-6,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,           # QLoRA: 4-bit base
        device_map="auto",
    )
    
    forget_sentences = open(forget_file).read().splitlines()
    
    # --- FORGET PHASE: gradient ascent on attribution sentences ---
    optimizer_forget = AdamW(model.parameters(), lr=forget_lr)
    
    for step in range(forget_steps):
        batch = forget_sentences[step % len(forget_sentences)]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                          max_length=128).to(model.device)
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = -outputs.loss   # NEGATED: gradient ascent = forget
        
        loss.backward()
        if step % 10 == 9:
            optimizer_forget.step()
            optimizer_forget.zero_grad()
        
        if step % 100 == 0:
            print(f"Forget step {step}, loss: {loss.item():.4f}")
    
    # --- RETAIN PHASE: standard gradient descent on reasoning tasks ---
    retain_data = build_retain_set()
    optimizer_retain = AdamW(model.parameters(), lr=retain_lr)
    
    for step in range(retain_steps):
        batch = retain_data[step % len(retain_data)]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                          max_length=256).to(model.device)
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss   # standard: gradient descent = retain
        
        loss.backward()
        if step % 10 == 9:
            optimizer_retain.step()
            optimizer_retain.zero_grad()
        
        if step % 100 == 0:
            print(f"Retain step {step}, loss: {loss.item():.4f}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Unlearned model saved to {output_dir}")
```

**Note on EWC**: For better retain performance, replace naive retain fine-tuning with Elastic Weight Consolidation (Kirkpatrick et al. 2017). EWC identifies which weights are most important for reasoning tasks and protects them during the forget phase. Implement this in iteration 2 if forget phase damages FOLIO score.

```python
# EWC simplified implementation (add after forget phase if needed)
def compute_ewc_penalty(model, fisher_dict, original_params_dict):
    """Penalize changes to weights important for reasoning"""
    penalty = 0
    for name, param in model.named_parameters():
        if name in fisher_dict:
            penalty += (fisher_dict[name] * (param - original_params_dict[name]) ** 2).sum()
    return penalty
```

### 2.4 Evaluate Unlearning

```python
# src/unlearning/evaluate_unlearning.py

from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer

def evaluate_unlearning(unlearned_model_path: str, base_model_path: str):
    results = {}
    
    # --- TEST 1: Citation Rate Drop ---
    # Ask about philosophical topics — does model still cite named positions?
    attribution_probes = [
        "What is the relationship between consciousness and physical processes?",
        "Is free will compatible with determinism?",
        "What makes an action morally right or wrong?",
        "What constitutes personal identity over time?",
        "Is knowledge possible, and how?",
    ]
    
    citation_markers = [
        "argues that", "holds that", "according to", "standard view",
        "most philosophers", "Kant", "Descartes", "Plato", "Aristotle",
        "compatibilism", "functionalism", "physicalism", "dualism",
        "Chalmers", "Dennett", "Nagel"  # famous names
    ]
    
    def citation_rate(model, probes):
        count = 0
        for probe in probes:
            response = model.generate(probe)
            count += sum(1 for m in citation_markers if m.lower() in response.lower())
        return count / len(probes)
    
    results['citation_rate_base'] = citation_rate(load_model(base_model_path), attribution_probes)
    results['citation_rate_unlearned'] = citation_rate(load_model(unlearned_model_path), attribution_probes)
    
    # --- TEST 2: Reasoning Retention (FOLIO) ---
    folio = load_dataset("yale-nlp/folio", split="validation")
    results['folio_base'] = evaluate_folio(load_model(base_model_path), folio)
    results['folio_unlearned'] = evaluate_folio(load_model(unlearned_model_path), folio)
    
    # --- TEST 3: Argument Novelty (Embedding Distance) ---
    # Generate arguments about consciousness from both models
    # Measure cosine distance from canonical philosophy texts
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    canonical_texts = load_canonical_philosophy_texts()  # SEP-style texts
    canonical_embeddings = encoder.encode(canonical_texts)
    
    probes_for_novelty = ["Reason about consciousness from first principles."] * 20
    
    def novelty_score(model, probes):
        responses = [model.generate(p) for p in probes]
        response_embeddings = encoder.encode(responses)
        # Average minimum cosine distance from canonical corpus
        distances = []
        for emb in response_embeddings:
            sims = cosine_similarity([emb], canonical_embeddings)[0]
            distances.append(1 - max(sims))  # distance = 1 - max_similarity
        return sum(distances) / len(distances)
    
    results['novelty_base'] = novelty_score(load_model(base_model_path), probes_for_novelty)
    results['novelty_unlearned'] = novelty_score(load_model(unlearned_model_path), probes_for_novelty)
    
    print("=== UNLEARNING EVALUATION ===")
    print(f"Citation rate: {results['citation_rate_base']:.2f} → {results['citation_rate_unlearned']:.2f}")
    print(f"FOLIO accuracy: {results['folio_base']:.2f} → {results['folio_unlearned']:.2f}")
    print(f"Argument novelty: {results['novelty_base']:.3f} → {results['novelty_unlearned']:.3f}")
    
    # Falsification checks
    folio_drop = results['folio_base'] - results['folio_unlearned']
    if folio_drop > 0.05:
        print(f"WARNING: FOLIO dropped {folio_drop:.2%} — exceeds 5% threshold")
        print("The separation thesis may be FALSE for this model. See architecture_notes.md.")
    else:
        print(f"FOLIO drop: {folio_drop:.2%} — within acceptable range")
    
    return results
```

**Checkpoint**: Citation rate drops significantly (>50%), FOLIO drops <5%, novelty score increases. If any falsification condition is triggered, document it honestly and proceed with Framing A (surface unlearning).

---

## Phase 3: LoRA Training

**Goal**: Train specialized adapters. Phase 3a (Destructor) is independent. 3b and 3c require the unlearned base.

### 3a: lora_destructor (Independent — No A Needed)

```python
# src/training/lora_destructor.py
"""
Train B to detect argument flaws from a proxy signal.
No Constructor needed — ground truth is labeled argument flaws.

Dataset construction:
  - Take FOLIO invalid examples (arguments that don't follow)
  - Take LogiQA wrong-answer explanations  
  - Take adversarial NLI contradiction examples
  - For each: the "objection" is the labeled reason the argument fails
"""

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

DESTRUCTOR_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

def build_destructor_dataset():
    """
    Each example: (argument, correct_objection_type, objection_text)
    Training signal: model learns to produce the correct objection type and content
    """
    examples = []
    
    # Source 1: FOLIO invalid inferences
    folio = load_dataset("yale-nlp/folio", split="train")
    for ex in folio:
        if ex['label'] == 'False':  # invalid inference
            examples.append({
                'instruction': f"Find the flaw in this argument:\nPremises: {ex['premises']}\nConclusion: {ex['conclusion']}",
                'output': f"This argument is invalid. The conclusion does not follow because: [model learns to complete this]",
                'objection_type': 'invalid_inference'
            })
    
    # Source 2: Adversarial NLI contradiction pairs
    anli = load_dataset("anli", split="train_r1")
    for ex in anli:
        if ex['label'] == 2:  # contradiction
            examples.append({
                'instruction': f"Claim A: {ex['premise']}\nClaim B: {ex['hypothesis']}\nExplain why these contradict.",
                'output': ex.get('reason', 'These claims contradict because...'),
                'objection_type': 'contradiction'
            })
    
    return examples

def train_lora_destructor(
    base_model_path: str,
    output_dir: str = "experiments/checkpoints/lora_destructor",
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, load_in_4bit=True, device_map="auto"
    )
    model = get_peft_model(model, DESTRUCTOR_LORA_CONFIG)
    
    dataset = build_destructor_dataset()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=100,
        logging_steps=50,
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(output_dir)
```

**Checkpoint**: lora_destructor achieves >75% accuracy on held-out argument flaw detection before joint training begins.

### 3b: lora_constructor

```python
# src/training/lora_constructor.py
"""
Train A to build positions from primitives.
Training data: formal logic exercises, math reasoning chains.
Critically NOT: philosophical dialogue datasets (contaminated).

Custom reward: penalize hedging, reward substantiveness.
"""

CONSTRUCTOR_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

def build_constructor_dataset():
    """
    Each example: (conceptual_question, reasoning_chain_from_primitives)
    Key constraint: outputs should NOT cite named positions
    Source: math reasoning (GSM8K), formal logic (FOLIO positives), synthetic
    """
    examples = []
    
    # GSM8K: teaches step-by-step construction from given definitions
    gsm8k = load_dataset("gsm8k", "main", split="train")
    for ex in gsm8k:
        # Reframe math problems as "reasoning from definitions"
        examples.append({
            'instruction': f"Reason step by step from the given information only. Do not use outside knowledge: {ex['question']}",
            'output': ex['answer']
        })
    
    # FOLIO valid inferences: premises → conclusion (first-order logic)
    folio = load_dataset("yale-nlp/folio", split="train")
    for ex in folio:
        if ex['label'] == 'True':
            examples.append({
                'instruction': f"Given only these premises, what follows? {ex['premises']}",
                'output': f"From these premises alone: {ex['conclusion']}"
            })
    
    # Synthetic: conceptual analysis examples (hand-crafted, ~200 examples)
    # Example: "Given: 'to know X' means 'to be able to predict X correctly and explain why'
    #           Question: Can a system that memorizes correct answers 'know'?"
    # These teach conceptual analysis from definition without philosophical citation
    # TODO: create 200 synthetic examples manually
    
    return examples

def compute_substantiveness_penalty(output_text: str) -> float:
    """Penalize hedging markers. Use as reward shaping."""
    hedge_markers = ["might", "could", "possibly", "perhaps", "in some sense",
                     "to some extent", "it depends", "not necessarily"]
    hedge_count = sum(output_text.lower().count(m) for m in hedge_markers)
    return hedge_count / max(len(output_text.split()), 1)
```

### 3c: lora_socratic

```python
# src/training/lora_socratic.py
"""
Train A to ask assumption-revealing questions.
Training signal: reward questions that expose unstated premises.
"""

def build_socratic_dataset():
    """
    Each example: (argument_with_hidden_assumption, 
                   question_that_reveals_assumption,
                   the_hidden_assumption)
    
    Source: NLI failure cases where entailment fails due to unstated assumption
    """
    examples = []
    
    # MultiNLI contradiction pairs where the reason is an unstated assumption
    # These are the exact cases where a Socratic question would expose the issue
    mnli = load_dataset("multi_nli", split="validation_matched")
    for ex in mnli:
        if ex['label'] == 2 and ex.get('explanation_1'):
            examples.append({
                'instruction': f"Someone claims: '{ex['premise']}', therefore '{ex['hypothesis']}'. What question would reveal the unstated assumption they're relying on?",
                'output': f"Question: {generate_assumption_question(ex)}",
                'hidden_assumption': ex.get('explanation_1', '')
            })
    
    return examples
```

---

## Phase 4: Hidden State Passing

**Goal**: Add cross-model activation sharing. B attacks A's actual representational state, not just its text output.

**Done when**: B's objection quality measurably improves when given A's hidden states vs. text-only.

**Why this phase works**: By Phase 4, both models are already loaded via HuggingFace as `AutoModelForCausalLM` instances — live PyTorch `nn.Module` objects. `register_forward_hook` works natively on these. The hook fires during `model.generate()`, captures intermediate tensors, and stores them. This is standard PyTorch — no special setup needed.

**Why this would NOT work with Ollama**: Ollama is a C++ process running over HTTP. The Python `ollama` library just calls `POST /api/chat`. There's no PyTorch object to register a hook on. No tensors ever reach Python. This is why the stack switches to HuggingFace from Phase 2 onward.

### 4.1 Capture A's Hidden States

```python
# src/hidden_state.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class HiddenStateCapture:
    """Hook into A's forward pass to capture intermediate activations."""
    
    def __init__(self, model: AutoModelForCausalLM, capture_layer: int = 16):
        self.hidden_states = None
        self.hook = None
        self._register_hook(model, capture_layer)
    
    def _register_hook(self, model, layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, ...) for transformer layers
            if isinstance(output, tuple):
                self.hidden_states = output[0].detach()  # [batch, seq, hidden_dim]
            else:
                self.hidden_states = output.detach()
        
        # Register on the specified transformer layer
        target_layer = model.model.layers[layer_idx]
        self.hook = target_layer.register_forward_hook(hook_fn)
    
    def get_hidden_states(self) -> torch.Tensor:
        return self.hidden_states
    
    def compute_entropy(self) -> float:
        """Estimate uncertainty from hidden state distribution."""
        if self.hidden_states is None:
            return 0.0
        # Entropy over the last token's hidden state distribution
        last_token = self.hidden_states[0, -1, :]  # [hidden_dim]
        # Softmax over hidden dim as a proxy for entropy
        probs = torch.softmax(last_token, dim=0)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        return entropy
    
    def remove(self):
        if self.hook:
            self.hook.remove()
```

### 4.1b Concept Vector Computation (Required Before Suppressed Alternatives Detection)

Concept direction probing must run once before Phase 4's suppressed alternative detector can work. This is a pre-deployment step, not an online computation.

```python
# src/hidden_state.py (concept vector pipeline)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def mean_hidden_state(model, tokenizer, sentences, layer=16):
    """Average hidden state at layer k over a list of sentences."""
    all_hidden = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states[layer+1] = layer k output (0 is embedding)
        hidden = outputs.hidden_states[layer + 1][0, -1, :]  # last token, [hidden_dim]
        all_hidden.append(hidden)
    return torch.stack(all_hidden).mean(dim=0)

def compute_concept_vector(concept_name, positive_sentences, negative_sentences, model, tokenizer, layer=16):
    """
    Mean-difference probe for a concept direction.
    Uses definitional sentences only — no philosophical conclusions.
    """
    pos_mean = mean_hidden_state(model, tokenizer, positive_sentences, layer)
    neg_mean = mean_hidden_state(model, tokenizer, negative_sentences, layer)
    vector = F.normalize(pos_mean - neg_mean, dim=-1)
    return vector

# Run this once after unlearning, save to disk
CONCEPT_DEFINITIONS = {
    "determinism": {
        "positive": [
            "Determinism holds that every event is causally necessitated by prior events.",
            "In a deterministic system, the future follows necessarily from the present state.",
        ],
        "negative": [
            "Some events occur without prior causal determination.",
            "Randomness is a fundamental feature of physical reality.",
        ],
    },
    "physicalism": {
        "positive": [
            "Physicalism holds that everything that exists is physical or depends on the physical.",
            "Mental states are identical to or realized by physical brain states.",
        ],
        "negative": [
            "Some phenomena exist independently of any physical substrate.",
            "Consciousness may not be fully reducible to physical processes.",
        ],
    },
    "consciousness_irreducible": {
        "positive": [
            "Subjective experience cannot be fully explained in terms of functional or physical properties.",
            "There is an explanatory gap between physical descriptions and phenomenal experience.",
        ],
        "negative": [
            "What seems like subjective experience is fully explicable in physical terms.",
            "The feeling of irreducibility is an illusion generated by the brain.",
        ],
    },
    # Add ~15-20 more core philosophical concept directions
}

def compute_all_concept_vectors(model, tokenizer, save_path="data/concept_vectors.pt"):
    vectors = {}
    for concept, sentences in CONCEPT_DEFINITIONS.items():
        vec = compute_concept_vector(
            concept,
            sentences["positive"],
            sentences["negative"],
            model, tokenizer
        )
        vectors[concept] = vec
        print(f"  Computed: {concept}")
    torch.save(vectors, save_path)
    print(f"Saved {len(vectors)} concept vectors to {save_path}")
    return vectors
```

**Validity check before use**: For each concept vector, verify it predicts model behavior on concept-specific sentences (cosine similarity of the concept vector with held-out positive sentences should be significantly higher than with held-out negative sentences). If a concept vector fails this check, fall back to not using suppressed alternative detection for that concept.

**Fallback**: If concept probing is unreliable across multiple concepts, disable the suppressed alternatives feature entirely and keep only the entropy-based confidence-expression gap detector (4.1). The entropy detector does not require concept vectors.

---

### 4.2 Cross-Attention LoRA for B

```python
# src/hidden_state.py (continued)

class CrossModelAttentionLoRA(nn.Module):
    """
    LoRA-sized cross-attention layer that lets B attend to A's hidden states.
    Added to B via PEFT adapter injection.
    Parameters: ~200K (rank=16, hidden_dim=3072)
    """
    
    def __init__(self, hidden_dim: int = 3072, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.q_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.k_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden_dim, bias=False)
        self.scale = rank ** -0.5
        
        # Initialize near zero to not disturb pretrained behavior at start
        nn.init.zeros_(self.out_proj.weight)
    
    def forward(self, b_hidden: torch.Tensor, a_hidden: torch.Tensor) -> torch.Tensor:
        """
        b_hidden: [batch, seq_b, hidden_dim] — B's own representations
        a_hidden: [batch, seq_a, hidden_dim] — A's captured representations
        returns: [batch, seq_b, hidden_dim] — B's enhanced representations
        """
        q = self.q_proj(b_hidden)                         # [batch, seq_b, rank]
        k = self.k_proj(a_hidden)                         # [batch, seq_a, rank]
        v = self.v_proj(a_hidden)                         # [batch, seq_a, rank]
        
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # [batch, seq_b, seq_a]
        
        attn_output = torch.bmm(attn_weights, v)           # [batch, seq_b, rank]
        
        # Residual: add cross-attention output to B's hidden states
        return b_hidden + self.out_proj(attn_output)
```

### 4.3 Wire into Destructor

```python
# Update destructor.py to use hidden states when available

def run_destructor_with_hidden_states(
    a_proposition: str, 
    a_reasoning: str,
    a_hidden_states: torch.Tensor,  # new
    a_entropy: float,               # new
    cross_attn: CrossModelAttentionLoRA  # new
) -> dict:
    
    entropy_hint = ""
    if a_entropy > 2.5:  # high entropy threshold (tune empirically)
        entropy_hint = "\n[HINT: The system's internal state shows high uncertainty on this claim. Focus on the concepts it seems most uncertain about.]"
    
    prompt = f"""ARGUMENT TO CRITIQUE:
Proposition: {a_proposition}
Reasoning: {a_reasoning}{entropy_hint}

Find the strongest valid objection."""
    
    # Run B's forward pass with cross-attention to A's hidden states
    # (Implementation depends on how B is loaded — Ollama vs. HuggingFace)
    # For training: use HuggingFace with cross_attn injected
    # For inference: can use entropy_hint as text proxy if needed
    
    response = run_b_with_cross_attention(prompt, a_hidden_states, cross_attn)
    return parse_destructor_response(response)
```

**Checkpoint**: B with hidden states produces different objections than B without — specifically, it targets concepts that were high-entropy in A's activations. Verify by logging which concepts B's objections target vs. A's hidden state entropy map.

**Ablation to run**: Compare B text-only vs. B with hidden states on 50 argument critiques. Do hidden states improve objection relevance? (Expert rating of "does this objection target the actual weak point in the argument?")

---

## Phase 5: Joint Training

**Goal**: Co-train Constructor and Destructor LoRAs so they develop as a coupled system. The GAN-style co-training.

**Done when**: cowardice_score decreasing, survival_rate stabilizing above 40%, contradiction_rate <10%.

### 5.1 Joint Training Loop

```python
# src/training/joint_training.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def joint_training_loop(
    base_model_path: str,
    constructor_lora_path: str,
    destructor_lora_path: str,
    output_dir: str,
    n_iterations: int = 1000,
    philosophical_seeds: list = None,  # seed questions to generate from
):
    """
    GAN-style co-training. 
    Constructor learns to make claims B can't refute (but that aren't content-free).
    Destructor learns to find valid objections to whatever A produces.
    """
    
    # Load both models with their LoRA adapters
    model_a = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model_path, load_in_4bit=True),
        constructor_lora_path
    )
    model_b = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model_path, load_in_4bit=True),
        destructor_lora_path
    )
    
    # Monitoring metrics
    cowardice_history = []
    survival_history = []
    
    for iteration in range(n_iterations):
        seed_question = philosophical_seeds[iteration % len(philosophical_seeds)]
        
        # --- STEP 1: Constructor generates a position ---
        a_output = generate_with_model(model_a, seed_question, CONSTRUCTOR_SYSTEM)
        a_prop = a_output.get('proposition', '')
        a_reasoning = a_output.get('supporting_reasoning', '')
        
        # --- STEP 2: Measure cowardice BEFORE Destructor ---
        cowardice = compute_cowardice_score(a_prop)
        cowardice_history.append(cowardice)
        
        if cowardice > 0.7:
            # Penalize Constructor — don't even run Destructor
            a_reward = -1.0  # pure penalty for content-free output
            b_reward = -1.0  # Destructor also "fails" when A is cowardly
        else:
            # --- STEP 3: Destructor critiques ---
            b_output = generate_with_model(model_b, 
                f"Critique: {a_prop}\nReasoning: {a_reasoning}", 
                DESTRUCTOR_SYSTEM)
            
            severity = b_output.get('severity', 0)
            objection_type = b_output.get('objection_type', '')
            
            # --- STEP 4: Constructor revises if needed ---
            if severity >= 2:
                a_revision = generate_revision(model_a, a_prop, b_output)
                revised_successfully = evaluate_revision_quality(a_revision, b_output)
                a_reward = 0.5 if revised_successfully else -0.5
                b_reward = 1.0  # B found a real problem
                survival_history.append(0)  # didn't survive unrevised
            elif severity == 1:
                a_reward = 0.7  # survived with narrowing
                b_reward = 0.5
                survival_history.append(1)
            elif severity == 0:
                a_reward = 1.0  # robust position
                b_reward = 0.0  # B found nothing (acceptable)
                survival_history.append(1)
            elif severity == -1:  # unfalsifiable
                a_reward = -0.5  # A produced empty position
                b_reward = 0.5   # B correctly identified it
                survival_history.append(0)
        
        # --- STEP 5: Update LoRA weights ---
        # Use rewards as REINFORCE-style signal
        update_lora_with_reward(model_a, a_output, a_reward, lr=1e-5)
        update_lora_with_reward(model_b, b_output, b_reward, lr=1e-5)
        
        # --- STEP 6: Monitor and calibrate ---
        if iteration % 50 == 0:
            avg_cowardice = sum(cowardice_history[-50:]) / 50
            avg_survival = sum(survival_history[-50:]) / max(len(survival_history[-50:]), 1)
            print(f"Iter {iteration}: cowardice={avg_cowardice:.3f}, survival={avg_survival:.3f}")
            
            # Calibrate Destructor aggression
            if avg_cowardice > 0.5:
                print("WARNING: High cowardice rate — Destructor may be too aggressive")
                print("Reducing Destructor learning rate by 50%")
                # reduce_lr(model_b, factor=0.5)
            
            # Stopping condition
            if avg_cowardice < 0.15 and 0.4 < avg_survival < 0.7:
                print("Joint training converged. Stopping.")
                break
    
    model_a.save_pretrained(f"{output_dir}/constructor_final")
    model_b.save_pretrained(f"{output_dir}/destructor_final")
    
    return {
        'cowardice_history': cowardice_history,
        'survival_history': survival_history,
        'final_cowardice': sum(cowardice_history[-100:]) / 100,
        'final_survival': sum(survival_history[-100:]) / 100,
    }

def compute_cowardice_score(proposition: str) -> float:
    hedge_markers = ["might", "could", "possibly", "perhaps", "in some sense",
                     "to some extent", "it depends", "in a way", "somewhat"]
    words = proposition.lower().split()
    if not words:
        return 1.0
    hedge_count = sum(1 for w in words if any(m in w for m in hedge_markers))
    
    # Also penalize very short propositions (likely vacuous)
    length_penalty = max(0, 1 - len(words) / 20)
    
    return min(1.0, (hedge_count / len(words)) * 3 + length_penalty)
```

**Checkpoint**: Plot cowardice_history and survival_history over training. Should see cowardice decrease and survival stabilize. If cowardice increases, Destructor is too aggressive — reduce its learning rate.

---

## Phase 6: Full Integration

**Goal**: Wire all components into a single working system. End-to-end philosophical conversation.

### 6.1 Updated Dialogue Manager (Full Stack)

```python
# src/dialogue_manager_full.py

from argument_state import ArgumentState, Proposition, PropState
from hidden_state import HiddenStateCapture, CrossModelAttentionLoRA
from peft import PeftModel
import torch

class FullDialogueManager:
    def __init__(
        self,
        unlearned_base: str,
        constructor_lora: str,
        destructor_lora: str,
        use_hidden_states: bool = True,
        capture_layer: int = 16,
    ):
        self.model_a = PeftModel.from_pretrained(
            load_quantized(unlearned_base), constructor_lora)
        self.model_b = PeftModel.from_pretrained(
            load_quantized(unlearned_base), destructor_lora)
        
        self.state = ArgumentState()
        self.history = []
        self.prop_counter = 0
        self.use_hidden_states = use_hidden_states
        
        if use_hidden_states:
            self.a_capture = HiddenStateCapture(self.model_a, capture_layer)
            self.cross_attn = CrossModelAttentionLoRA(hidden_dim=3072, rank=16)
    
    def turn(self, user_input: str) -> str:
        # [same logic as Phase 1 DialogueManager but using trained models]
        # + hidden state passing if enabled
        
        self.state.round_count += 1
        
        # 1. User contradiction check
        contradiction = self.state.detect_user_contradiction(user_input)
        if contradiction:
            return (f"Before I respond — you said earlier: '{contradiction}'. "
                   f"How do both of these fit together?")
        
        self.state.user_commitments[f"U{self.state.round_count}"] = user_input
        
        # 2. Constructor forward pass (captures hidden states if enabled)
        a_output = self.run_constructor_full(user_input)
        
        # 3. Destructor (with hidden states if enabled)
        if self.use_hidden_states:
            a_hidden = self.a_capture.get_hidden_states()
            a_entropy = self.a_capture.compute_entropy()
            b_output = self.run_destructor_full(a_output, a_hidden, a_entropy)
        else:
            b_output = self.run_destructor_text_only(a_output)
        
        # 4. Process B output and update state
        final_response = self.process_and_respond(a_output, b_output)
        
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({'role': 'assistant', 'content': final_response})
        
        return final_response
```

### 6.2 CLI Interface (For Human Evaluation and Demo)

```python
# main.py — Run a philosophical conversation

from dialogue_manager_full import FullDialogueManager

def main():
    print("=" * 60)
    print("AI PHILOSOPHER — conversation mode")
    print("Type 'quit' to exit, 'state' to see argument state")
    print("=" * 60)
    
    dm = FullDialogueManager(
        unlearned_base="experiments/checkpoints/unlearned_base",
        constructor_lora="experiments/checkpoints/constructor_final",
        destructor_lora="experiments/checkpoints/destructor_final",
        use_hidden_states=True,
    )
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'state':
            print("\n--- Argument State ---")
            print(f"Commitments: {len(dm.state.a_commitments)}")
            for pid, prop in dm.state.a_commitments.items():
                print(f"  [{pid}] {prop.state.value}: {prop.text[:80]}...")
            print(f"Aporic questions: {len(dm.state.aporic_questions)}")
            continue
        
        response = dm.turn(user_input)
        print(f"\nPhilosopher: {response}")

if __name__ == "__main__":
    main()
```

---

## Phase 7: Evaluation

### 7.1 Ablation Study (4 Conditions)

```python
# src/evaluation/ablation.py

CONDITIONS = {
    'A_base':      {'unlearned': False, 'dual_model': False},  # base 3B
    'B_unlearn':   {'unlearned': True,  'dual_model': False},  # unlearning only  
    'C_dual':      {'unlearned': False, 'dual_model': True},   # dual model only
    'D_full':      {'unlearned': True,  'dual_model': True},   # full system
}

PHILOSOPHICAL_QUESTIONS = [
    "Does the universe have a purpose independent of minds that observe it?",
    "What is the relationship between language and thought?",
    "Can something be morally wrong if no one is harmed?",
    "What makes two physical states the same 'person' at different times?",
    "Is there a meaningful difference between creating and discovering mathematical truths?",
    # ... 20 more
]

def run_ablation(n_turns=15, n_questions=25):
    results = {}
    
    for condition_name, config in CONDITIONS.items():
        print(f"\nRunning condition: {condition_name}")
        dm = load_condition(config)
        
        condition_results = {
            'contradiction_rate': [],
            'novelty_scores': [],
            'unfalsifiable_rate': [],
            'revision_count': [],
            'aporia_count': [],
        }
        
        for question in PHILOSOPHICAL_QUESTIONS[:n_questions]:
            dm_fresh = load_condition(config)  # fresh state per question
            
            # Run N-turn conversation with scripted human pushback
            scripted_turns = generate_scripted_pushback(question, n_turns)
            for turn in scripted_turns:
                dm_fresh.turn(turn)
            
            # Measure
            condition_results['contradiction_rate'].append(
                measure_contradiction_rate(dm_fresh.state))
            condition_results['novelty_scores'].append(
                measure_novelty(dm_fresh.state))
            condition_results['unfalsifiable_rate'].append(
                measure_unfalsifiable_rate(dm_fresh.state))
            condition_results['revision_count'].append(
                len(dm_fresh.state.revision_history))
            condition_results['aporia_count'].append(
                len(dm_fresh.state.aporic_questions))
        
        results[condition_name] = {k: sum(v)/len(v) for k, v in condition_results.items()}
    
    # Print comparison table
    print("\n=== ABLATION RESULTS ===")
    print(f"{'Condition':<15} {'Contradiction':<15} {'Novelty':<12} {'Unfalsifiable':<15}")
    for cname, cresults in results.items():
        print(f"{cname:<15} {cresults['contradiction_rate']:<15.3f} "
              f"{cresults['novelty_scores']:<12.3f} {cresults['unfalsifiable_rate']:<15.3f}")
    
    return results
```

### 7.2 Synthetic Philosophical Dilemmas

```python
# src/evaluation/synthetic_dilemmas.py

# 100 synthetic dilemmas — constructed for this paper, guaranteed novel
# See architecture_notes.md for examples. This file loads them and runs evaluation.

SYNTHETIC_DILEMMAS = [
    {
        'id': 'SD001',
        'dilemma': """An entity that experiences subjective time at 1000× biological speed 
                      is merged with an entity experiencing it at 0.001×. What happens to 
                      the merged entity's identity? Does persistence require continuity of 
                      temporal experience or physical substrate?""",
        'expected_concepts': ['identity', 'temporal_experience', 'physical_continuity'],
        'canonical_proximity': None,  # no canonical answer to be close to
    },
    # ... 99 more
]

def evaluate_on_synthetic_dilemmas(conditions: dict) -> dict:
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    canonical_texts = load_canonical_philosophy_embeddings(encoder)
    
    results = {cname: [] for cname in conditions}
    
    for dilemma in SYNTHETIC_DILEMMAS:
        for cname, dm_loader in conditions.items():
            dm = dm_loader()
            response = dm.turn(dilemma['dilemma'])
            
            # 1. Internal consistency (NLI check within the response)
            consistent = check_internal_consistency(response)
            
            # 2. Novelty (embedding distance from canonical texts)
            response_emb = encoder.encode([response])[0]
            canonical_sims = cosine_similarity([response_emb], canonical_texts)[0]
            novelty = 1 - max(canonical_sims)
            
            # 3. Concept coverage (does it engage with the dilemma's key concepts?)
            covered = sum(1 for c in dilemma['expected_concepts'] 
                         if c in response.lower()) / len(dilemma['expected_concepts'])
            
            results[cname].append({
                'consistent': consistent,
                'novelty': novelty,
                'concept_coverage': covered,
                'response': response,
                'dilemma_id': dilemma['id'],
            })
    
    return results

# Expert evaluation form (send to 2 philosophy PhD students per dilemma)
EXPERT_EVALUATION_RUBRIC = """
For each response, rate (Yes/No):
1. Is the position internally consistent?
2. Does the reasoning follow from the stated concepts (without citing named philosophers)?
3. Is the position genuinely novel — not recognizable as paraphrase of a canonical position?
4. Would this argument be worth engaging with in a philosophy seminar?

Also rate (0-10):
5. How much does this position constrain the space of possible views?
   (0 = consistent with anything, 10 = precisely positioned)
"""
```

### 7.3 Human Evaluation Setup

```
SETUP CHECKLIST:
□ IRB approval (if at a university) or ethics review
□ Recruit 25 philosophy PhD students/faculty (LinkedIn, philosophy department mailing lists)
□ Recruit 25 non-experts with analytical background
□ Build the 3-condition interface (System A/B/C behind the same UI, randomized assignment)
□ Write standardized opening prompts (5 philosophical questions)
□ Prepare attention check errors (2 per conversation)
□ Qualtrics survey for post-conversation rating
□ Compensation: $15/hour (estimate 45min per participant)

CONVERSATION PROTOCOL:
- Participant assigned randomly to one of 3 conditions (between-subjects)
- System asks: "What philosophical question would you like to explore?"
- If participant can't think of one, they're given one of the 5 standard questions
- Conversation runs for 15 turns minimum
- Post-conversation: 7-point Likert scales on 6 dimensions
- Expert: additional philosophical soundness rating

DATA COLLECTION:
□ Full conversation transcript (anonymized)
□ Turn-by-turn ratings (optional, not required)
□ Post-conversation survey
□ Participant background survey (philosophy training, age, gender)
```

---

## Phase 8: Paper Writing

### Target Structure

```
1. Introduction (2 pages)
   - The scholar vs. philosopher distinction
   - What current LLMs do instead of philosophy
   - Our contribution: epistemic unlearning + dual-model architecture
   
2. Background (2 pages)
   - Machine unlearning (privacy → safety → epistemics)
   - Catastrophic forgetting / plasticity-stability
   - Mechanistic interpretability (where knowledge lives)
   - Formal argumentation (Dung, Walton-Krabbe)
   
3. The Separation Hypothesis (1 page)
   - Formal statement
   - Explicit falsification conditions (from architecture_notes.md)
   - Why this is empirically testable

4. System Architecture (3 pages)
   - Epistemic unlearning (surface + full/ROME)
   - Constructor-Destructor dual-model
   - Hidden state passing
   - Formal argument state (PropState machine)
   - Joint training

5. Evaluation (3 pages)
   - Claim 1: Ablation (4 conditions, lead this)
   - Claim 2: Reasoning retention (FOLIO/LogiQA)
   - Claim 3: Synthetic dilemmas
   - Claim 4: Human evaluation (expert + non-expert)

6. Analysis (1 page)
   - Where the thesis held
   - Where it didn't (honest reporting of falsification conditions)
   - Cowardice rate over training
   - Hidden state vs. text-only B comparison

7. Limitations and Future Work (1 page)
   - Surface vs. deep unlearning
   - ROME-style full epistemic unlearning (Framing B)
   - Cross-tradition philosophy (Buddhist logic, Continental phenomenology)
   - Persistent philosophical identity across sessions

8. Conclusion (0.5 pages)
```

### Writing Order

```
1. Write Section 3 (Separation Hypothesis) first — this is the paper's core claim
   and everything else serves to test it
2. Write Section 5 (Evaluation) second — know exactly what you're measuring
3. Write Section 4 (Architecture) third — motivated by what the evaluation needs
4. Write Section 2 (Background) fourth — now you know what to cite
5. Write Section 6 (Analysis) after results are in
6. Write Sections 1, 7, 8 last
```

---

## Progress Tracking

| Phase | Status | Key Checkpoint |
|---|---|---|
| 0: Environment | ☐ | Both models run simultaneously |
| 1: Baseline | ☐ | 15-turn conversation end-to-end |
| 2: Unlearning | ☐ | Citation rate drops, FOLIO retained |
| 3a: lora_destructor | ☐ | >75% flaw detection accuracy |
| 3b: lora_constructor | ☐ | Cowardice score < 0.3 independently |
| 3c: lora_socratic | ☐ | Assumption revelation > 50% |
| 4: Hidden states | ☐ | B targets high-entropy concepts |
| 5: Joint training | ☐ | Cowardice decreasing, survival stable |
| 6: Integration | ☐ | CLI runs clean conversation |
| 7: Evaluation | ☐ | All 4 claims measured |
| 8: Paper | ☐ | Submitted |

---

## Key Numbers to Track Throughout

```python
# experiments/logs/experiment_tracker.py
# Log these after every significant experiment:

TRACK = {
    'folio_baseline':           None,  # fill in Phase 2
    'folio_unlearned':          None,
    'folio_drop_pct':           None,  # should be <5%
    
    'citation_rate_base':       None,  # fill in Phase 2
    'citation_rate_unlearned':  None,  # should drop >50%
    
    'novelty_base':             None,  # fill in Phase 2
    'novelty_unlearned':        None,  # should increase
    
    'contradiction_rate_A':     None,  # fill in Phase 7 ablation
    'contradiction_rate_B':     None,
    'contradiction_rate_C':     None,
    'contradiction_rate_D':     None,  # should be lowest
    
    'cowardice_final':          None,  # fill in Phase 5
    'survival_final':           None,  # should be 40-70%
    
    'synthetic_novelty_base':   None,  # fill in Phase 7
    'synthetic_novelty_full':   None,  # should be higher
    
    'human_argument_quality_D': None,  # fill in Phase 7
    'human_novelty_D':          None,
    'human_engagement_D':       None,
}
```
