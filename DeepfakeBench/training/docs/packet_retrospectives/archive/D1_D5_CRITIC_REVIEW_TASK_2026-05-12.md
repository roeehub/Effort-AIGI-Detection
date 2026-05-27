# D1-D5 critic-review task — 2026-05-12

> **Status**: self-contained task spec for an independent agent. Authored by the planning agent who wrote `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md`. The user wants a SECOND OPINION on that synthesis from an agent who reads the FACTS without inheriting the planning agent's framing.
>
> **The task is to CHALLENGE, not to validate.** Deferential agreement is failure mode. Performative disagreement (disagreeing for sport when the evidence doesn't support it) is also failure mode. The deliverable is a sharper read of the data, with disagreements pinned to specific evidence.

---

## 0. Your role

You are an independent reviewer. The planning agent has spent a session synthesizing 5 CPU diagnostics (D1-D5) into:

1. A "deeper story" paragraph (D1_D5_OPINIONS_2026-05-12.md §1) describing the mechanism of how P8A, T5C, T3 each fail and what the path to a single-model deployment candidate looks like.
2. A 4-slot GPU plan (D1_D5_OPINIONS_2026-05-12.md §3) with explicit hypotheses, falsifiers, and priority ordering.
3. A self-correction log (§4) listing intra-session retractions.
4. A what-I-might-be-wrong list (§5).

Your job: form an INDEPENDENT view of the same data, then explicitly compare your view to the planning agent's. Specifically the user wants to hear "your version of the story" vs "the planning agent's deeper story" side-by-side.

---

## 1. Mission constraints

- **Read FACTS before OPINIONS.** Do not open D1_D5_OPINIONS_2026-05-12.md until you have read all 5 FACTS docs and formed your own initial reading. This is the AGENTS.md "Reading order for forming an independent view" protocol.
- **No deferential agreement.** If you agree with the planning agent, say "agree because [specific evidence]," not "the planning agent's reading is reasonable." Generic affirmations contribute nothing.
- **No performative disagreement.** If you genuinely agree on a point, agree. The user cares about CORRECTNESS, not contrarianism.
- **Cite evidence at the file:section level.** Every load-bearing claim should reference a specific FACTS doc section. "D3 says X" without §-citation is too loose.
- **Stay inside MODEL_GOALS.** Single model. NO ENSEMBLE. B16-only. Three pillars (recall, FPR<5%, axis-robustness).
- **Forbidden words apply in FACTS-style sections but not OPINION sections.** Your deliverable is an OPINION doc; you may use words like "shortcut", "memorization", "lucky", "confirmed", "refuted" — but be precise.

---

## 2. Reading order

### Pass 1 — FACTS only (form your own view BEFORE reading any opinion)

1. [`analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md)
2. [`analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md)
3. [`analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md)
4. [`analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md)
5. [`analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md)

For each FACTS doc, write yourself a one-sentence "what this measurement actually shows" note. Don't synthesize across docs yet — just internalize each one.

### Pass 2 — Form your own synthesis BEFORE reading the OPINIONS doc

Now write your own "deeper story" paragraph: given the 5 FACTS docs, what is your reading of (a) why P8A handles chronic-6 differently than T5C/T3/E2B; (b) where the dev-lockbox transfer gap structurally lives; (c) what intervention class is most likely to produce a single-model candidate that meets the MODEL_GOALS three pillars. Aim for ~150 words. Keep it before you read the planning agent's version.

### Pass 3 — Read other context that the planning agent USED (so your view is informed of the same priors)

These are the upstream context docs the planning agent had in working memory when authoring the OPINIONS. Read them before reading the OPINIONS doc; they're FACTS (or near-facts) and they shape interpretation:

6. [`docs/packet_retrospectives/MODEL_GOALS.md`](MODEL_GOALS.md) — three pillars + NO ENSEMBLE + B16 + promotion bar.
7. [`docs/packet_retrospectives/AGENT_GUIDE.md`](AGENT_GUIDE.md) — 6-rule contract.
8. [`docs/packet_retrospectives/STATE.md`](STATE.md) — current packet state (read the top three "where we stand" paragraphs).
9. **Upstream sibling diagnostics** (older, still active):
   - [`analysis/cpu_diagnostics_2026-05-12_stage_a/STAGE_A_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_stage_a/STAGE_A_FACTS_2026-05-12.md)
   - [`analysis/cpu_diagnostics_2026-05-12_stage_a/JOB1_L11_DISTANCE_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_stage_a/JOB1_L11_DISTANCE_FACTS_2026-05-12.md)
   - [`analysis/cpu_diagnostics_2026-05-12_stage_a/JOB2_OVERFIRE_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_stage_a/JOB2_OVERFIRE_FACTS_2026-05-12.md)
   - [`analysis/cpu_diagnostics_2026-05-11_a1_cross_substrate/CROSS_SUBSTRATE_FACTS_2026-05-11.md`](../../analysis/cpu_diagnostics_2026-05-11_a1_cross_substrate/CROSS_SUBSTRATE_FACTS_2026-05-11.md)
   - [`analysis/cpu_diagnostics_2026-05-11_a2_extension/A2_EXTENSION_FACTS_2026-05-11.md`](../../analysis/cpu_diagnostics_2026-05-11_a2_extension/A2_EXTENSION_FACTS_2026-05-11.md)
   - [`analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md`](../../analysis/cpu_diagnostics_2026-05-11_a3_atlas_composition/ATLAS_COMPOSITION_FACTS_2026-05-11.md)
   - [`analysis/cpu_diagnostics_2026-05-11_head_retrain/HEAD_RETRAIN_FACTS_2026-05-11.md`](../../analysis/cpu_diagnostics_2026-05-11_head_retrain/HEAD_RETRAIN_FACTS_2026-05-11.md)
10. [`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](threads/iq_shortcut_deconvolution_program_2026-05-08.md) — the parent thread for the IQ-shortcut deconvolution program. Read the "2026-05-12 evening — D1-D5 CPU diagnostic program" section at the END of the file for the planning agent's framing in thread form.
11. Recent packet retros (read tail of each for the verdict):
    - [`docs/packet_retrospectives/packets/T6_T7_T5C.md`](packets/T6_T7_T5C.md)
    - [`docs/packet_retrospectives/packets/T3.md`](packets/T3.md)
    - [`docs/packet_retrospectives/packets/P8A.md`](packets/P8A.md)
    - [`docs/packet_retrospectives/packets/PD.md`](packets/PD.md)

You are ENCOURAGED to also look at other threads, memory entries, or analyses you think might be relevant. The user wants you to "look elsewhere under `packet_retrospectives` so you can form your own view on the matter." Don't restrict yourself to the planning agent's references.

### Pass 4 — Read the planning agent's OPINION docs

12. [`analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md) — THE synthesis you are critiquing.
13. [`analysis/cpu_diagnostics_2026-05-12_stage_a/STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_stage_a/STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md) — superseded by D1_D5_OPINIONS but useful context for what the planning agent retracted.

---

## 3. Specific things to check (point-by-point)

The user has explicitly flagged these claims as load-bearing in the planning agent's synthesis. Engage with each one and form a verdict (agree / disagree / partially agree / data insufficient).

### 3.1 The "deeper story" paragraph (§1 of OPINIONS doc)

Quote the paragraph back. Identify each claim with an inline tag. Engage with each tagged claim:

- C1: "The forgery signal is already in raw CLIP B16" — supported by D2 obs 1 (AUC=1.000 on chronic-6). Is this generalizable beyond chronic-6? What does it imply for non-chronic generalization?
- C2: "Standard FT does not add the forgery signal — FT modifies how it's projected" — does this follow from D2 alone, or do we need additional measurements?
- C3: "FT pulls the chronic-6 encoder separation direction 9-13° toward color-axis IQ alignment" — D2 says this. The 87.5° random-orthogonality baseline assumes random unit vectors; trained probe vectors aren't random. Is the baseline empirically justified?
- C4: "P8A learns identity-cluster memorization" — D5 says per-identity R²=0.22 with mean-IQ; D1 says per-identity coefficients sign-flip. Does "identity-cluster memorization" follow from these, or is "structured per-identity IQ usage" a better characterization?
- C5: "T5C step3500's IQ-correlation lives at the head, not the encoder" — D5 R²=0.60 output vs D2 75° encoder angle. Is the reconciliation logically forced, or are there alternative reconciliations (e.g., the encoder direction has high IQ-projection magnitude even though the angle is mostly orthogonal — magnitude vs direction confound)?
- C6: "Every FT'd encoder retains DIFFERENT separations per substrate" — D3 says 0/4 ckpts pass substrate-agnostic. The lockbox slice is n=87. Is this enough to support the "structural data gap" reading, or should Tier 2 escalation be required before accepting the framing?
- C7: "P8A-output and P8A-L11 anchor levers are off the table" — the planning agent's retraction depends on C4 ("identity memorization is what would be transferred"). Is the inference chain valid?

### 3.2 The 4-slot GPU plan (§3 of OPINIONS doc)

For each slot:
- Is the hypothesis grounded in the cited FACTS?
- Is the falsifier sharp (does a clean experimental outcome map to a clean reading)?
- Are the priorities ranked correctly given the evidence?
- Are there better slots NOT proposed? In particular:
  - The planning agent dropped P8A-output-distillation and P8A-L11-anchor — is this the right call?
  - The planning agent kept LoRA at MEDIUM priority despite D3 showing P8A's encoder is the most substrate-specific (DEV→LOCKBOX 0.594). Is this consistent?
  - The planning agent did NOT propose a "head retrain on substrate-diverse pool" slot. D3 says head retraining on dev-features can't bridge the encoder gap, but it didn't test head retraining on a MIXED pool. Should this be a slot?
  - The planning agent did NOT propose a "data ingestion" slot (bring lockbox-style frames into training). Should this be a slot?
- For each slot, is the cost estimate realistic given the codebase complexity?

### 3.3 Self-correction log (§4)

The planning agent retracts 7 framings within the session. Some retractions are partial. Spot-check:
- Is the L11-feature-anchor retraction based on `JOB1_L11_DISTANCE_FACTS_2026-05-12.md` §3 evidence solid, or could the "uniform L11 drift" reading itself be wrong (e.g., the cosdist median collapses cohort-specific structure)?
- The "T5C is axis-shorting" partial retraction depends on the magnitude-vs-direction reconciliation (C5 above). Same scrutiny applies.

### 3.4 What I might be wrong about (§5 of OPINIONS)

The planning agent flagged 8 uncertainties. For each, decide:
- Is the uncertainty actually load-bearing for the conclusion?
- Is there an additional uncertainty the planning agent missed?
- Specifically: the "lockbox is easier than dev at dev-cal τ" finding (D4 vs contract scorecard reading) — does this require investigation before accepting D1-D5 synthesis at face value?

### 3.5 What's in other packet retros / threads that the planning agent's synthesis might be ignoring

The user explicitly invites looking elsewhere. Things to check:
- `packets/P22.md` (if it exists) or memory `project_p22_succeeded_2026-05-02` / `project_p22_cpu_followups_reframe_2026-05-02` — the pipeline_randomization step1k success was characterized as "score variance collapsed 140×." Does this connect to the D1-D5 framing?
- `packets/PD.md` and memory `project_corr_penalty_*` — PD tried 3-axis correlation penalty with mixed result. Is Slot 1 (continuous-axis-GRL on 6 axes) at risk of the same "shortcut shifts to untargeted axes" failure mode PD had?
- `packets/E2B*` or memory `project_e2b_breaks_deeplive_ceiling_2026-05-04` — E2B is B16-scratch+CE. Slot 2 is B16-scratch+anti-shortcut-bundle. Does E2B's outcome bound Slot 2's expectation?
- Memory `project_xinhe_may6_falseflag_2026-05-06` and `project_deployment_is_e2b_2026-05-06` — the production model is E2B, but P8A handles may6 cleanly (0/92). Where does may6 fit in D1-D5's framing?
- Memory `project_v2_substrate_is_dor_diverse_swap` — the v2 substrate is specifically dor + swap-model-diverse. How does this interact with the chronic-6 cohort the planning agent's synthesis centers on?

These are starting points; you are encouraged to find your own connections.

---

## 4. Deliverable

Write a single OPINIONS doc at `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md` with this structure:

### §0 — Status header

Disclaimer per AGENT_GUIDE Rule 5. Note that this is a critic-review, not a primary OPINION. Cite this task spec as the assignment.

### §1 — Your "deeper story" paragraph (BEFORE comparing to the planning agent's)

Write your version, formed in Pass 2. ~150 words. This is the critical artifact — the user wants to compare yours to the planning agent's side-by-side. Do not rewrite this after reading the planning agent's version.

### §2 — Side-by-side comparison

Two-column table. Left: planning agent's §1 claims (tagged C1-C7 as above). Right: your verdict (agree / partial / disagree / data insufficient) with one-sentence justification per row.

### §3 — Each contested claim in detail

For every C1-C7 (or any other planning-agent claim you contest), one subsection. Cite the FACTS evidence. State the alternative interpretation if you have one.

### §4 — The 4-slot plan: your version

If you'd KEEP the planning agent's 4 slots: say so + add comments / refinements.
If you'd MODIFY: list your alternative 4 slots with hypothesis + falsifier + cost + priority.
If you'd DROP some or ADD others: explain why.

The user has budget for 4 slots. Don't artificially constrain to 4; if you think 3 is right or 5 is right, say so. But land on a concrete recommendation.

### §5 — Things the planning agent missed

Anything from `packet_retrospectives` (threads, packets, memories) that should have been in scope but wasn't.

### §6 — Your unresolved questions

What the next CPU diagnostic should be, if any. What you'd want to know before authorizing GPU.

### §7 — Self-criticism

Per AGENT_GUIDE Rule 5: list anything in your own review that might be wrong, and how a future agent could check it.

---

## 5. Boundaries

- **Do NOT write the FACTS docs themselves.** They are factual artifacts produced by the diagnostic subagents. Your job is to interpret, not to re-run.
- **Do NOT propose new CPU diagnostics WITHOUT cost / time estimates.** If you propose D6, say "$0 / ~2h MPS / requires X cached features" or similar.
- **Do NOT make claims about the LoRA code agent's work** — that's a parallel workstream you don't have visibility into.
- **Do NOT update OPEN_LOOPS / STATE / TIMELINE** — those are the planning agent's responsibility. Your output is the single critic-review doc.
- **If you find a critical error** in any FACTS doc (e.g., a number that's clearly wrong, or a methodology mistake), flag it in §5 — but don't try to "fix" the FACTS doc itself.

---

## 6. Failure modes to avoid

- **Deferential summary**: "The planning agent's view is reasonable, and I agree on most points." Useless output.
- **Generic skepticism**: "The sample sizes are small, the metrics are noisy, more data is needed." True but useless without specifics.
- **Re-arguing the user's original critique**: The user already pushed back on P8A-anchor levers. The planning agent retracted them. You don't need to re-defeat them; you need to evaluate the NEW synthesis.
- **Ignoring the side-by-side request**: The user explicitly wants YOUR "deeper story" paragraph next to the planning agent's. Don't skip §1.
- **Drift into implementation details**: This is a strategic-synthesis review, not a code review. Don't get bogged down in LoRA implementation choices.

---

## 7. Length / time

- Target length: ~1500-2500 words for the deliverable doc. The synthesis OPINIONS doc you're critiquing is ~3000 words; your critique can be shorter.
- Target time: 2-4 hours of reading + 1-2 hours of writing.
- If you find yourself spending > 6h on this, scope down: write §1 (your "deeper story") + §2 (side-by-side table) + §4 (slot recommendations) + §7 (self-criticism). Drop §3 (detailed contest of each claim) — the table can carry the key disagreements.

---

## 8. Hand-back

When you're done:
- The deliverable doc lives at `analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md`.
- Append a single TIMELINE.md entry (~1 line) noting your critic-review completed.
- Don't update STATE.md or memory; the planning agent will adjudicate and propagate.
- Send a short summary back (~200 words): the headline "your story vs planning agent's story" disagreement (if any), the 4-slot recommendation, the one thing you'd most want investigated next.

The user reads your summary first, then the deliverable, then asks the planning agent to respond. That's the loop.
