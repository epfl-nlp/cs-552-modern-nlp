# Week 7: Advanced LLM Training and Inference Techniques

This week covers three progressively advanced techniques for improving LLM performance — from prompt-level strategies to full RL-based training. Students should complete the exercises in the order listed below.

---

## Exercise Order

### 1. In-Context Learning, Chain-of-Thought & Self-Consistency
**Notebooks:** `Week-7-ICL-CoT-SelfConsistency.ipynb`

An introduction to inference-time techniques that elicit better reasoning from LLMs without any training.

**Topics covered:**
- Zero-shot vs. few-shot prompting (In-Context Learning)
- Chain-of-Thought (CoT) prompting — zero-shot and few-shot variants
- Self-Consistency: sampling multiple reasoning paths and aggregating via majority vote

**What you will implement:**
- Zero-shot baseline and zero-shot CoT comparison
- Few-shot CoT with hand-crafted demonstrations
- Self-Consistency with N=15 sampled completions and vote analysis

---

### 2. Supervised Fine-Tuning & Direct Preference Optimization
**Notebooks:** `Week-7-SFT-DPO.ipynb`

Moves from prompting to training — covering the standard SFT pipeline and a streamlined alternative (DPO) that skips explicit reward modeling.

**Topics covered:**
- Supervised Fine-Tuning (SFT) on high-quality demonstration data
- Direct Preference Optimization (DPO): mathematical derivation from the Bradley-Terry preference model

**What you will implement:**
- SFT training loop with causal LM loss (prompt tokens masked)
- DPO loss function — concatenated forward pass, log-probability computation, reward accuracy metrics
- Full DPO training loop and evaluation on the Anthropic HH preference dataset

---

### 3. Reinforcement Learning with Verifiable Rewards & Test-Time Scaling
**Notebooks:** `Week7-RLVR-TTS.ipynb`

The most advanced exercise, covering RL-based training with programmatic reward signals and inference-time compute scaling (i.e., "thinking" models).

**Topics covered:**
- Group Relative Policy Optimization (GRPO): RL without a value/critic network
- Reinforcement Learning with Verifiable Rewards (RLVR): binary rewards from ground-truth checking
- Test-Time Compute Scaling (TTS): thinking vs. non-thinking models, thinking budget control
- Evaluation on AIME 2025 mathematical olympiad problems

**What you will implement:**
- Per-token log probabilities from logits
- Group-normalized advantage estimation
- Importance sampling ratios and clipped surrogate objective
- KL divergence penalty
- Full GRPO loss and mini training step
- Answer extraction and binary reward computation for math problems
- Test-time scaling experiments with DeepSeek-R1-style thinking budgets
