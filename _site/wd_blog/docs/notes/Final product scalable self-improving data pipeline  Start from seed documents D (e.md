1. **Final product: scalable self-improving data pipeline**  
   1. Start from seed documents **D** (e.g. internet).  
   2. Iteratively generate data so that:  
      1. Document quality improves with more compute.  
      2. Pretraining on the next generation of data yields a significantly better base model.

2. **Algorithm A**  
   1. Input: model **X**, data **D**.  
   2. Output: model **X’**, data **D’**.  
   3. Run **A** for many iterations.

3. **Success criterion**  
   1. Self-improvement is persistent across iterations (quality of **D**, capability of **X** keep improving).

4. **Baselines**  
   1. **A1 (pure generation):** Use **X** to generate **R**; set **D’ = D ∪ R**; pretrain **X’** on **D’**.  
   2. **A2 (rewriting):** Use **X** to rewrite **D** into **R**; set **D’ = D ∪ R**; pretrain **X’** on **D’**.  
   3. **A3 (RL pretraining / long CoT):** Condition on prefixes in **D**, predict suffixes with long CoT; collect successful traces as **R**.  
   4. **Status:** **A1/A2** likely help for a single run but not beyond; **A3** is widely tried but improvements are still limited.

5. **Assumptions**  
   1. **X’** is always pretrained on **D’** (how we test the success criterion).  
   2. Improvement in **D’** over **D** is the source of improvement of **X’** over **X**.  
   3. We ignore rigorous validation of **D’** in the short term (to revisit later).

6. **Question 0: is this important/exciting?**  
   1. It is hard to argue we won’t eventually need such an algorithm **A**.

7. **Question 1: is this theoretically doable?**  
   1. A human can perform multiple iterations of **A**, so it is at least conceptually feasible.

8. **Question 2: do we have good enough ideas?**  
   1. Candidate scaffold **A4**:  
      1. **A4.1:** Create synthetic QA pairs.  
      2. **A4.2:** Synthesize QA pairs into knowledge.  
      3. Both instantiated by scaffolding **X** with reasoning + search over **D**.  
      4. Sub-versions of **X**:  
         1. **X1:** question generator.  
         2. **X2:** answer generator.  
         3. **X3:** synthesizer.  
      5. Instantiation:  
         1. Start with prompting; add SFT for a longer-term version.  
         2. Eventually may design rewards.

9. **Question 3: how do we de-risk?**  
   1. Core question: can **A** persistently generate better data?  
   2. By (5.2), improvement in **D’** over **D** must drive improvements in **X**.  
   3. Define partial **A’** that only updates data:  
      1. \((X, D) → (X, D’)\).  
      2. This gives a lower bound on the potential of **A**.  
   4. Loop **A’** for many iterations.  
   5. Metrics:  
      1. **Short-term success (go / no-go):**  
         1. Exists an iteration where **D’** is clearly better than the (rewritten) pretraining corpus and what **X** previously knows (subjective judgment OK).  
         2. If never true, **A** is unlikely to work.  
      2. **Long-term behavior (what we want to improve moving from A’ to A):**  
         1. After some iterations **D’** collapses due to hallucinations.  
         2. **D’** plateaus in quality/diversity.  
         3. Neither happens and performance monotonically improves (ideal, but too good to expect from **A’**).  
   6. Materialization:  
      1. Work on a fixed domain.  
      2. Use scaffolding only (no training) for fast iteration.  
      3. Use strong models initially (e.g. **X = Gemini 3 Pro / Kimi K2 Thinking**), with scaling down in mind.  
         1. Short-term success may require a strong **X**.  
         2. Be careful about distillation: we may just be extracting what **X** already knows.  
      4. Judge **D’** with humans + LLMs using:  
         1. Information density.  
         2. Novelty (especially relative to **X**).  
         3. Clarity.  
      5. Use domains we know well so humans can judge:  
         1. Deep learning.  
         2. Theory.

10. **Question 4: long-term planning**  
    1. If we gain conviction in some **A’**, run it with models we can actually pretrain for multiple iterations.  
    2. Design rewards to improve long-term behavior: extend the effective horizon, reduce hallucination, and maintain/increase diversity.