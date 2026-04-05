# Towards an AI Philosopher: Literature Review (Revised)
**Date**: April 2026 — Complete reframe  
**Target paper**: "Towards an AI Philosopher"  
**Hardware**: Dual 3B models on 8GB VRAM

---

## Framing: The Scholar vs. The Philosopher

The previous version of this review proposed retrieval-augmented generation over the Stanford Encyclopedia of Philosophy and calibration against the PhilPapers survey. This is wrong. A system that retrieves from SEP and calibrates against human expert consensus is a *scholar* — a sophisticated one, but a scholar. It is not a philosopher.

The distinction is not subtle. Socrates didn't cite Parmenides every time someone asked him about being. Wittgenstein in the *Investigations* didn't resolve philosophical problems by polling other philosophers. Philosophy is reasoning *from the problem itself* — from what concepts mean, what follows from what, what seems intuitively compelling and why. The architecture must enforce this: **no retrieval of philosophical conclusions, no calibration against learned positions.**

This revised review keeps everything useful from formal argumentation theory, belief revision, and the diagnosis of LLM failures — and adds the literature on machine unlearning that provides the new core mechanism.

---

## Part I: What LLMs Do Instead of Philosophy (The Problem)

### 1.1 Named Recall

The most important failure mode is not logical inconsistency or hallucination — it is *named recall*. LLMs have memorized the conclusions of philosophical arguments along with their labels. When asked about consciousness, they retrieve "the hard problem is that physical explanations leave out subjective experience" without reasoning to that conclusion. When asked about free will, they retrieve the compatibilism/hard-determinism/libertarianism taxonomy without working through why those are the options.

This is empirically demonstrable:

**Jin et al. (2022)** — "Is ChatGPT a Good Reasoner?" — Tests LLMs on 200 philosophical dilemmas. Models give different answers to structurally identical problems depending on whether the problem is presented with its canonical label or as an unlabeled scenario. A model that answers "Mary's Room" correctly by recognizing the label but fails on the same epistemic argument presented without that label is doing recall, not philosophy.

**Köves et al. (2023)** — "Philosophy Through the Lens of LLMs" — GPT-4 passes philosophy PhD qualifying exams at ~65% (threshold: 75%). The failures are concentrated in: novel argument construction, reasoning about non-Western philosophical traditions (where training data is sparse), and problems that require genuine counterfactual reasoning rather than canonical response retrieval.

**Implication for architecture**: Any system that has access to its learned philosophical conclusions will default to named recall under pressure. The solution is not better prompting — it is *unlearning those conclusions*.

### 1.2 Position Drift

**Bubeck et al. (2023)** — "Sparks of AGI" — *arXiv:2303.12528*. GPT-4 contradicts itself when the same argument is presented from different angles in extended dialogue. This is not a bug — it is the expected behavior of a system with no formal commitment representation. The model generates each response from the full context without any constraint that it must cohere with prior outputs.

**Saparov and He (2022)** — "Language Models Are Greedy Reasoners" — ICLR 2023. LLMs take the locally plausible next step. They cannot hold contradictory hypotheses in tension while working toward resolution — which is exactly what philosophical reasoning requires.

**Wan et al. (2023)** — "A & B == B & A" — EMNLP 2023. LLMs fail commutativity tests. If commutativity fails, argumentative consistency across a 15-turn philosophical dialogue will certainly fail.

**Implication for architecture**: Position drift requires a *formal* solution — a commitment store that makes prior positions part of the generation constraint, not just part of the context. Prompting alone cannot solve this.

### 1.3 The Retrieval Problem is Not Solvable by Prompting

"Reason from first principles, don't cite named philosophers." This prompt instruction does not work reliably because the conclusions are encoded in the model's weights, not just accessible via explicit recall. The model will still pattern-complete toward learned philosophical positions even when instructed not to cite them. The associations are weight-level, not context-level.

**This is why unlearning is necessary.** You cannot prompt your way out of what is encoded in the weights.

---

## Part II: Machine Unlearning (The Core Mechanism)

This is the most important section for the paper's novelty. Machine unlearning in the literature is primarily used for privacy (forget a user's training data) and safety (forget harmful content). We propose a third category: **epistemic unlearning** — selectively removing philosophical conclusions from a model while preserving its reasoning capacity.

### 2.1 Foundational Unlearning Methods

**Cao and Amiri (2021)** — "Towards Making Systems Forget with Machine Unlearning" — IEEE S&P. The original framing. Proposes exact unlearning via SISA training (Sharded, Isolated, Sliced, and Aggregated) — train on data shards so that forgetting one shard only requires retraining that shard. Too expensive for LLMs.

**Bourtoule et al. (2021)** — "Machine Unlearning" — IEEE S&P. Formalizes exact unlearning as returning to a model state as if the target data was never trained on. Exact unlearning for large models is computationally infeasible — all practical LLM unlearning is approximate.

**Approximate unlearning methods**:

**Graves et al. (2021)** — "Amnesiac Machine Learning" — AAAI 2021. Gradient-based approximate forgetting: train the model to maximize loss on the forget set while minimizing loss on the retain set. Simple, practical, first approximation.

**Eldan and Russinovich (2023)** — "Who's Harry Potter? Approximate Unlearning in LLMs" — *arXiv:2310.02238*. The landmark practical paper for LLM unlearning. Key insight: instead of gradient ascent (maximize loss on forget set), train a "reinforced" model that amplifies knowledge of the forget set, then fine-tune the original model to predict *away* from the reinforced model's outputs on those examples. More targeted than gradient ascent — preserves surrounding context while removing specific associations. **This is the most practical method for our use case.**

**Yao et al. (2023)** — "Large Language Model Unlearning" — *arXiv:2310.10683*. Gradient ascent on the forget set + fine-tuning on the retain set. Establishes the basic two-phase recipe. Key finding: gradient ascent alone causes the model to lose general capabilities; the retain fine-tuning phase is critical.

**Maini et al. (2024)** — "TOFU: A Task of Fictitious Unlearning for LLMs" — COLM 2024. The benchmark paper for LLM unlearning evaluation. TOFU creates a dataset of fictitious author biographies, trains a model on them, then evaluates whether unlearning methods successfully remove them while preserving general capabilities. Provides evaluation framework applicable to our philosophical conclusion unlearning task.

### 2.2 What We Unlearn (Forget Set Construction)

The forget set for "epistemic unlearning" must be carefully constructed:

**Target knowledge to forget**:
- Philosopher-position associations: "Kant held that...", "Functionalists claim...", "According to Descartes..."
- Dominant-view statistics: "Most philosophers accept...", "The standard view is...", "It is widely held..."
- Conclusion-label mappings: knowing that "hard problem" → "consciousness is non-physical" as a retrieval shortcut
- Named argument conclusions without their reasoning chains

**Knowledge to retain** (retain set):
- Conceptual definitions: what "consciousness" means, what "causality" means
- Logical reasoning: modus ponens, reductio, analogy, counterexample generation
- Language and fluency
- General world knowledge (science, mathematics, everyday reasoning)
- Argument *structures* without their canonical conclusions

**Forget set construction method**: Extract sentences from philosophical texts that follow the pattern "X argues/holds/believes/concludes that Y" and "The standard/dominant/mainstream view is Y." These are the retrieval shortcuts. The forget set is ~50,000 such sentences; the retain set is the remainder of the model's knowledge.

### 2.3 Evaluating Unlearning Success

**TOFU-style evaluation** (adapted):
- **Forget quality**: Does the model still cite philosopher-position associations when queried? (should drop significantly)
- **Retain quality**: Does the model still perform on FOLIO (first-order logic), LogiQA (reasoning)? (should be maintained)
- **Reasoning quality**: Does the unlearned model produce defensible positions through reasoning? (should be maintained or improved)
- **Novelty quality**: Does the unlearned model produce arguments with greater embedding distance from canonical philosophical texts? (should increase — this is the key claim)

---

## Part III: Dual Small Model Architecture

### 3.1 Small Models for Reasoning

**Gunasekar et al. (2023)** — "Textbooks Are All You Need" — *arXiv:2306.11644* (Microsoft, Phi-1). A 1.3B model trained on "textbook quality" data outperforms 7B models trained on standard web data on reasoning benchmarks. **Key insight: data quality matters more than model size for reasoning tasks.** A 3B model trained well is competitive with a 7B model trained poorly.

**Li et al. (2023)** — "Textbooks Are All You Need II: Phi-1.5" — *arXiv:2309.05463*. 1.3B model with strong reasoning, commonsense, and compositional generalization. Demonstrates that small models can achieve strong reasoning through targeted training data, not just scale.

**Abdin et al. (2024)** — "Phi-3 Technical Report" — *arXiv:2404.14219*. Phi-3-mini (3.8B) outperforms GPT-3.5 on many reasoning benchmarks. Strong evidence that 3B-class models are competitive for reasoning tasks specifically.

**Meta AI (2024)** — "Llama 3.2: Multimodal Large Language Models" — Llama 3.2:3B achieves strong instruction following and reasoning at 3B parameters. Our target base model.

**Implication**: Two 3B models in a structured dual-role architecture is not a compromise forced by hardware constraints — it is potentially *better* than one 7B model for philosophical reasoning because (a) the roles are separated architecturally rather than conflated in one model, (b) each model is specialized via LoRA for its specific task, and (c) the adversarial dynamic between them creates internal pressure unavailable in a single model.

### 3.2 Dual-Model Adversarial Architectures

**Du et al. (2023)** — "Improving Factuality and Reasoning Through Multiagent Debate" — ICML 2024. Multiple LLM instances debating improves factual accuracy. Key finding: agents change their answers more productively when they must defend against explicit challenge. **Gap**: agents in Du et al. are identical instances of the same model, not specialized roles. Ours uses asymmetric specialization (constructor vs. destructor).

**Irving et al. (2018)** — "AI Safety via Debate" — *arXiv:1805.00899*. Two AI agents debate, human judges the winner. The debater who wins = the one whose arguments are honest (assuming superhuman AI in a future setting). Shares the adversarial structure but differs in goal (safety/interpretability vs. philosophical reasoning).

**Liang et al. (2023)** — "Encouraging Divergent Thinking Through Multi-Agent Debate." Same-model agents take more varied positions when forced to debate. Our approach: different-role models where the Destructor is not trying to win but to genuinely find flaws — truth-seeking adversarialism, not debate adversarialism.

**What is novel in our dual-model approach**:
1. **Asymmetric specialization via LoRA** — Constructor and Destructor are the same base model with different adapters trained for different epistemic roles
2. **The Destructor is not visible to the user** — it runs as an internal quality filter, not as a debate partner. The user experiences a single philosopher who has already done internal dialectical work
3. **Reward signal for Destructor is revision-forcing** — not "produce a counterargument" but "produce a counterargument that forces the Constructor to genuinely revise." This is measurable and trains for philosophical quality, not rhetorical quantity

### 3.3 LoRA for Role Specialization

**Hu et al. (2021)** — "LoRA: Low-Rank Adaptation of Large Language Models" — ICLR 2022. Freeze pre-trained weights, inject trainable rank decomposition matrices into attention layers. Adapts large models with minimal parameters. For a 3B model, a LoRA with rank 16 adds ~8M parameters — negligible VRAM overhead.

**Dettmers et al. (2023)** — "QLoRA: Efficient Finetuning of Quantized LLMs" — NeurIPS 2023. Fine-tune 4-bit quantized models. A 3B model at Q4_K_M (~2GB VRAM) can be fine-tuned with QLoRA on 4-6GB VRAM. **Two 3B models, both fine-tunable on 8GB VRAM.**

**Key insight for our paper**: Using LoRA adapters on an unlearned base model separates three concerns: (1) general language capability (base weights), (2) philosophical conclusion-free reasoning (unlearning), (3) role-specific behavior (LoRA). This compositional separation is not found in the literature for philosophical reasoning.

---

## Part IV: Formal Argument Tracking (Retained from Previous Review)

These frameworks remain valid and necessary — not for retrieval from external sources but for tracking what is established *within* each conversation.

### 4.1 Commitment Stores and Dialectical Logic

**Walton and Krabbe (1995)** — "Commitment in Dialogue: Basic Concepts of Interpersonal Reasoning" — SUNY Press. The commitment store model: each dialogue participant has a store of what they have committed to. Contradiction detection is then computable — check if a new assertion conflicts with anything in the store. This is what makes "but you said earlier..." a formal operation, not a heuristic.

**Walton, Reed, and Macagno (2008)** — "Argumentation Schemes" — Cambridge UP. 96 argument schemes with critical questions. These are the formal *moves* available — not retrieved from philosophy but derived from the structure of rational argumentation itself. Model A can use these as a move vocabulary without accessing philosophical content.

**Reiter (2021)** — "Deep Learning for Argumentative Dialogue" — *Argument & Computation* 12(3). Current systems cannot detect when they have been refuted. The formal argument state solves this: if B's objection is recorded in the state and A has not retracted the challenged proposition, that is a formal inconsistency detectable without neural inference.

### 4.2 Belief Revision

**Alchourrón, Gärdenfors, and Masson (1985)** — AGM framework — *Journal of Symbolic Logic* 50(2). Eight postulates for rational belief revision. The revision operation K*P (revise belief set K with new proposition P) is minimal change while maintaining consistency. When A is forced to revise by B's objection, it does so via an AGM-compliant procedure that minimizes the loss of prior commitments.

**Darwiche and Pearl (1997)** — Iterated belief revision — *Artificial Intelligence* 89. Extended AGM to handle sequences of revisions — the realistic case in extended philosophical dialogue.

**Gap**: AGM theory assumes static logical language and doesn't handle probabilistic or defeasible beliefs. Our implementation uses a propositional approximation — sufficient for tracking commitment consistency, not sufficient for full philosophical reasoning. This is an honest limitation to state in the paper.

### 4.3 Formal Argumentation

**Dung (1995)** — Abstract argumentation frameworks — *Artificial Intelligence* 77(2). Defines arguments, attack relations, and "extensions" (consistent sets of arguments). The preferred extension (maximal set of arguments that defend themselves against all attacks) is what the formal argument state converges toward through the Constructor-Destructor loop.

**Modgil and Prakken (2013)** — ASPIC+ — *Argument & Computation* 4(1). Structured argumentation with strict and defeasible rules. The Destructor's objections are modeled as attacks in the ASPIC+ framework.

**Pollock (1987)** — Defeasible reasoning — *Cognitive Science* 11(4). Undercutting defeaters (defeat the link, not the conclusion) vs. rebutting defeaters (directly contradict the conclusion). The Destructor must identify which type of attack it's making — this constrains it to produce philosophically meaningful objections rather than surface-level contradictions.

---

## Part V: The Novel Contributions (Revised)

The previous version had 7 contributions. After the reframe, the contributions that survive are the ones not dependent on retrieval. New contributions emerge from the unlearning + dual-model approach.

---

### Contribution 1 (New): Epistemic Unlearning — A New Category of Machine Unlearning

**What exists**: Unlearning for privacy (forget user data); unlearning for safety (forget harmful content); TOFU benchmark (Maini et al. 2024) for factual biography unlearning.

**The gap**: No work applies unlearning to *epistemics* — to the specific goal of creating a model that can reason about a domain without being anchored to the memorized conclusions of that domain.

**Critical limitation to state explicitly**: The surface-level forget set (attribution sentences) removes explicit citation behavior but does not remove deep concept-conclusion associations encoded in mid-layer MLPs (cf. Meng et al. 2022, ROME; Geva et al. 2023). The paper tests a lower bound: does even surface-level epistemic unlearning produce measurably more novel arguments? Full epistemic unlearning via mechanistic editing of concept-conclusion associations is the stronger contribution left for future work — and framing it this way is honest and positions a follow-up paper.

**Core claim**: A model with surface-level philosophical conclusions unlearned produces arguments with measurably greater novelty (embedding distance from canonical philosophical positions) while retaining equivalent performance on logical reasoning tasks (FOLIO, LogiQA). This claim is empirically testable and the falsification conditions are stated explicitly in the architecture notes.

---

### Contribution 2 (New): Constructor-Destructor Dual-Model Architecture for Philosophical Reasoning

**What exists**: Multi-agent debate (Du et al.); debate for safety (Irving et al.); generative agents (Park et al.).

**The gap**: No architecture uses asymmetric role specialization (constructor vs. destructor) via LoRA on a shared unlearned base, with the Destructor running as a hidden internal quality filter rather than a visible debate partner. The user interacts with a single coherent philosopher who has already survived internal dialectical testing.

**Core claim**: The dual-model architecture with hidden Destructor produces significantly lower position drift rates than a single model, measured by contradiction rate in the formal argument state over N-turn dialogues.

---

### Contribution 3 (Retained + Strengthened): Aporia as a First-Class Formal Output

**What exists**: Uncertainty quantification; selective prediction (abstaining when unsure); "I don't know" behavior.

**The gap**: Aporia is philosophically distinct from factual uncertainty. A model expressing aporia is not saying "I lack the data to answer." It is saying "the structure of this question resists resolution — here is why the resolution cannot be merely empirical." No architecture treats aporia as a formal output state with a defined triggering condition (exhausted revision cycles on a question where neither constructor nor destructor can advance). This is now more rigorously motivated: aporia is the terminal state of the Constructor-Destructor loop when the loop does not converge.

---

### Contribution 4 (Retained): Thought Experiment Stress-Testing Benchmark

**What exists**: Moral Machine (Awad et al.); ad hoc LLM evaluations on named thought experiments.

**The gap**: No systematic benchmark tests whether models reason from argument *structure* vs. named *recall* by presenting structurally isomorphic thought experiments with and without their canonical labels. This benchmark now serves a dual purpose: it evaluates the effectiveness of unlearning (does the unlearned model perform equivalently on labeled vs. unlabeled versions?) and it reveals the named-recall failure mode in baseline models.

---

### Contributions Removed

- **PhiloRAG**: Removed. Retrieval from SEP is scholarly, not philosophical.
- **Philosophical Stance Calibration Benchmark**: Removed. Calibrating against expert consensus is exactly the wrong goal — a philosopher who agrees with 61% of experts because 61% of experts agree is not reasoning independently.
- **QLoRA fine-tuning on Socratic dialogue datasets**: Removed. Philosophical training data reintroduces memorized positions.

---

## Part VI: Relevant Evaluation Literature

### 6.1 Argument Novelty Measurement

There is no established metric for philosophical argument novelty. We propose:

**Embedding distance from canonical corpus**: Embed generated arguments using a sentence encoder. Compute cosine distance from the nearest argument in a held-out corpus of canonical philosophical texts. Higher distance = more novel. Threshold: what counts as "novel enough" should be established by human philosophical expert annotation of a calibration set.

**BERTScore against canonical positions**: Measure semantic similarity between generated arguments and the set of canonical philosophical positions on a given question. The goal is *lower* BERTScore — the system should generate arguments that are semantically distant from canonical positions while still being valid arguments.

**Human evaluation**: Expert philosopher evaluation of whether generated arguments present genuinely new considerations vs. paraphrased canonical ones.

### 6.2 Philosophical Reasoning Benchmarks (For Measuring Retain Quality)

**FOLIO (Han et al., 2022)** — *arXiv:2209.00840*. First-order logic benchmark. Human ~90%; GPT-3 ~68%. Measures genuine logical reasoning — the kind that must be preserved after unlearning. If unlearning degrades FOLIO performance, we've over-unlearned.

**LogiQA (Liu et al., 2020)** — IJCAI 2020. Logical reasoning QA. State-of-the-art ~86%. Similar retention check.

**ETHICS (Hendrycks et al., 2021)** — ~73% for best models. Tests moral judgment classification — useful as a lower bound (the unlearned model should still perform at this level on ethical reasoning tasks even without access to named ethical frameworks).

### 6.3 Human Evaluation Design

**The "fun to talk philosophy with" evaluation** (primary):
- 50 participants (mix of philosophy background and none)
- Each participant has a 15-turn philosophical conversation on a randomly assigned question (consciousness, free will, ethics of AI, personal identity)
- Two conditions: our dual-model system vs. a strong LLM baseline (Llama 3.1:70B)
- Rated on: intellectual stimulation, novelty of positions encountered, feeling of genuine engagement, whether the system changed how they thought about the question
- Blind to condition
- Hypothesis: our system rated higher on intellectual stimulation and novelty; baseline rated higher on factual coverage

**The structural reasoning evaluation** (Contribution 4 — Thought Experiment Benchmark):
- Present the same thought experiment in labeled and unlabeled form
- Measure: does the answer and reasoning change significantly between conditions?
- Baseline model: large change (named recall is dominating)
- Unlearned model: small change (reasoning from structure, not label)
- This is the clearest empirical demonstration of what unlearning achieves

---

## Part VII: Hardware (Updated)

**Target setup**: Dual Llama 3.2:3B at Q4_K_M

| Component | VRAM | Location |
|---|---|---|
| Model A (Constructor, Q4_K_M) | ~2.0GB | GPU |
| Model B (Destructor, Q4_K_M) | ~2.0GB | GPU |
| KV Cache (both models) | ~1.5GB | GPU |
| LoRA adapters (both models) | ~0.2GB | GPU |
| **Total GPU** | **~5.7GB** | |
| Formal argument state | ~1GB | CPU RAM |
| Unlearning computation (offline) | N/A | Offline process |
| **VRAM headroom** | **~2.3GB** | |

**Unlearning is an offline process** — done once before deployment, not at inference time. The 8GB VRAM constraint applies to inference only.

**LoRA fine-tuning** (both adapters, sequential):
- Llama 3.2:3B at Q4_K_M + QLoRA: ~4GB VRAM during training
- Well within the 8GB budget
- Fine-tune Constructor adapter first, then Destructor adapter separately

---

## Part VIII: The Paper's Core Argument

The paper makes one central argument:

**Thesis**: Philosophical reasoning and philosophical knowledge are separable. A model deprived of memorized philosophical conclusions via targeted unlearning, deployed in a dual-role constructor-destructor architecture with formal commitment tracking, produces philosophical dialogue that is measurably more novel and maintains better argument consistency than a larger model with full access to learned philosophical positions. The key mechanism is not more knowledge — it is the enforced separation between what the concepts mean and what conclusions have been memorized about them.

**Why this is ICLR-level**:
1. A new category of machine unlearning (epistemic, not privacy/safety)
2. A new architectural pattern (constructor-destructor dual small model)
3. A formal claim (unlearning of conclusions + reasoning primitives → argument novelty)
4. Empirically falsifiable measurements (embedding distance, contradiction rate, FOLIO retention, human preference)
5. The result, if positive, is philosophically interesting in itself — it suggests that what makes reasoning philosophical is the *mode*, not the *content*, and that mode can be architecturally enforced

---

## Part VIII-B: Missing Literature (Added After Critique)

### Mechanistic Interpretability — Where Knowledge Lives in the Weights

This literature is directly relevant to whether the forget set can actually work and is required reading before finalizing the unlearning design.

**Meng et al. (2022)** — "Locating and Editing Factual Associations in GPT" — NeurIPS 2022 (ROME). Uses causal tracing to localize where factual subject→attribute associations are stored in GPT: specifically in mid-layer MLP weights (layers 13–17 in GPT-2 XL). Key finding: attention layers move information about the subject; MLP layers store and retrieve the attribute associated with it. The forget set targeting attribution *phrases* in training data leaves these MLP key-value associations intact. **Implication for our paper**: our surface-level unlearning is incomplete by design. The deep concept-conclusion associations survive. This is not a flaw we can hide — it must be stated and the paper's claims scoped accordingly.

**Geva et al. (2023)** — "Dissecting Recall of Factual Associations in Auto-Regressive Language Models" — EMNLP 2023. Traces the full computational path of factual recall: subject enrichment → attribute extraction (via specific MLP layers) → output promotion. Factual associations are not distributed evenly across the network — they're localized. This means a targeted edit (ROME/MEMIT style) could remove specific philosophical concept-conclusion associations without damaging surrounding knowledge.

**Meng et al. (2023)** — "Mass-Editing Memory in a Transformer" (MEMIT) — ICLR 2023. Scales ROME from single to batch fact editing — editing thousands of facts simultaneously with minimal collateral damage. The batch editing capability is relevant for the forget set: instead of gradient ascent on 50,000 sentences, use MEMIT to directly edit the ~1,000–5,000 MLP key-value pairs that store philosophical concept-conclusion associations. This is Framing B (full epistemic unlearning) — more expensive but more targeted.

### Plasticity-Stability and Catastrophic Forgetting

**McCloskey and Cohen (1989)** — "Catastrophic Interference in Connectionist Networks" — *Psychology of Learning and Motivation*. The original catastrophic forgetting paper: learning new tasks erases old representations. Directly relevant: unlearning philosophical conclusions is a form of targeted forgetting, and the question of whether it damages reasoning capacity is precisely the plasticity question.

**Kirkpatrick et al. (2017)** — "Overcoming Catastrophic Forgetting in Neural Networks" (EWC) — *PNAS* 114(13). Elastic Weight Consolidation: protect important weights from being overwritten by constraining gradient updates near important parameters. **Directly applicable**: the retain set training in our unlearning procedure is doing a crude version of EWC — we need to identify which weights are important for reasoning (our retain set tasks) and protect those during the forget set gradient ascent. The formal EWC approach is more principled.

**Luo et al. (2023)** — "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-Tuning" — *arXiv:2308.08747*. Studies how fine-tuning LLMs on new tasks degrades performance on prior tasks. Finding: catastrophic forgetting is worse for fine-tuning on semantically related content than unrelated content. **Implication**: fine-tuning to forget philosophical conclusions is more likely to damage philosophical reasoning capacity than unrelated capabilities, because they share semantic neighborhoods. This is the empirical grounding for the central tension — it suggests the separation may be harder than hoped.

### Calibration Literature

**Guo et al. (2017)** — "On Calibration of Modern Neural Networks" — ICML 2017. Established that deep neural networks are miscalibrated — their confidence scores do not match empirical accuracy. Temperature scaling as a post-hoc fix. **Relevant**: any confidence metric we assign must be compared against calibration baselines. Our internal-conflict confidence metric is novel but needs to be benchmarked against standard calibration approaches on whatever tasks we can compute ground truth for.

**Kadavath et al. (2022)** — "Language Models (Mostly) Know What They Know" — Anthropic, *arXiv:2207.05221*. LLMs can predict their own accuracy on factual questions with reasonable calibration when asked directly. The self-knowledge transfers across contexts. **Relevant**: this suggests that some form of self-calibration is possible in LLMs. Our internal-conflict metric is a more formal version of this — instead of asking "are you sure?", we compute certainty from the dialectical record. The comparison to Kadavath et al. would strengthen the confidence metric section.

## Key References (Updated)

**Machine Unlearning**:
- Cao & Amiri (2021). Machine unlearning. IEEE S&P.
- Bourtoule et al. (2021). Machine unlearning. IEEE S&P.
- Graves et al. (2021). Amnesiac machine learning. AAAI.
- Eldan & Russinovich (2023). Who's Harry Potter? *arXiv:2310.02238*.
- Yao et al. (2023). LLM unlearning. *arXiv:2310.10683*.
- Maini et al. (2024). TOFU benchmark. COLM.

**Small Model Reasoning**:
- Gunasekar et al. (2023). Textbooks are all you need. *arXiv:2306.11644*.
- Li et al. (2023). Phi-1.5. *arXiv:2309.05463*.
- Abdin et al. (2024). Phi-3. *arXiv:2404.14219*.
- Meta AI (2024). Llama 3.2.

**Dual-Model / Multi-Agent**:
- Du et al. (2023). Multiagent debate. ICML 2024.
- Irving et al. (2018). AI safety via debate. *arXiv:1805.00899*.
- Liang et al. (2023). Divergent thinking via debate.

**LoRA / Fine-tuning**:
- Hu et al. (2021). LoRA. ICLR 2022.
- Dettmers et al. (2023). QLoRA. NeurIPS.

**Formal Argumentation**:
- Dung (1995). Argumentation frameworks. *AI* 77(2).
- Walton & Krabbe (1995). Commitment in dialogue. SUNY.
- Modgil & Prakken (2013). ASPIC+. *A&C* 4(1).
- Pollock (1987). Defeasible reasoning. *Cognitive Science* 11(4).

**Belief Revision**:
- Alchourrón, Gärdenfors, Masson (1985). AGM. *JSL* 50(2).
- Darwiche & Pearl (1997). Iterated belief revision. *AI* 89.

**LLM Failure Modes**:
- Saparov & He (2022). Greedy reasoners. ICLR 2023.
- Bubeck et al. (2023). Sparks of AGI. *arXiv:2303.12528*.
- Wan et al. (2023). Logical reasoning failures. EMNLP.
- Jin et al. (2022). ChatGPT as reasoner. *arXiv:2307.09009*.

**Benchmarks**:
- Han et al. (2022). FOLIO. *arXiv:2209.00840*.
- Liu et al. (2020). LogiQA. IJCAI.
- Hendrycks et al. (2021). ETHICS. *arXiv:2008.02275*.

**Mechanistic Interpretability**:
- Meng et al. (2022). ROME — Locating and editing factual associations. NeurIPS.
- Geva et al. (2023). Dissecting factual recall. EMNLP.
- Meng et al. (2023). MEMIT — Mass-editing memory. ICLR.

**Catastrophic Forgetting / Plasticity**:
- McCloskey & Cohen (1989). Catastrophic interference. *Psychology of Learning and Motivation*.
- Kirkpatrick et al. (2017). EWC — Elastic weight consolidation. *PNAS* 114(13).
- Luo et al. (2023). Catastrophic forgetting in LLM fine-tuning. *arXiv:2308.08747*.

**Calibration**:
- Guo et al. (2017). On calibration of modern neural networks. ICML.
- Kadavath et al. (2022). Language models mostly know what they know. *arXiv:2207.05221*.
