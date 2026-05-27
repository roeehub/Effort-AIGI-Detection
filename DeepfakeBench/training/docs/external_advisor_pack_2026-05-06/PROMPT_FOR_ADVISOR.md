# Prompt to the external advisor

> Paste the body below as the user message to the external LLM (or hand to a human reviewer). The advisor has access only to the files in this pack — they have NOT been shown the team's existing ranked list or synthesis docs.

---

You are an independent reviewer brought in for a second opinion on a face-swap detection project. The team has been working on this for ~3 weeks; you are seeing the project for the first time.

You have been given a self-contained pack of three documents and a `data/` folder of CSVs/JSONs:

- `ARCHITECTURE.md` — the model architecture (CLIP-B/16 + SVD residuals on attention/MLP), how SVD is applied, the original paper reference (https://arxiv.org/pdf/2411.15633), and the three checkpoint variants (P8A, E2B, PA_3800) that appear in the diagnostic results.
- `FACTS.md` — the factual snapshot of the project's state: what the user wants, what's been tried, today's 7 CPU probes' raw results, and the open structural questions. **Bias-stripped: every claim cites a file.**
- `data/` — raw CSV / JSON outputs of the 7 probes referenced in `FACTS.md`, plus the 14,626-row consolidated identity-browser dataset.

## Your task

Read `ARCHITECTURE.md` → `FACTS.md`, then sample whichever CSVs in `data/` you want to inspect for the questions you have. Then form **your own independent recommendation** for what the team should do next.

The user goal is **a single deployed model** that simultaneously holds:
1. fake recall ≥ 90% on target methods (deeplive, visomaster, teams)
2. real-side FPR ≤ 5%
3. robustness across capture conditions (lighting / camera / codec / color)

All three pillars are load-bearing. Per-mode τ at inference is NOT deployable (Teams does not surface capture mode at inference). Ensembling is NOT an option — the user requires a single model. Hard-negative mining has been explicitly de-prioritized as "a patch, not a structural fix" per the user.

## What the user specifically wants from you

1. **Your ranked list** of the next 1-3 packets to run, with reasoning grounded in the data in §3-4 of FACTS.md and the prior R13 outcomes ledger in §5.
2. **Confidence interval** on whether each proposal would actually produce a model that meets the three-pillar goal (be honest about uncertainty).
3. **Cheap pre-validation probes** you'd want to run before committing GPU spend on each proposed packet.
4. **What information you'd want that isn't in FACTS.md** — so the user can prioritize a probe to gather it before launching anything.
5. **Hindsight pass on §5 (prior R13 attempts)**: anything you think the team mis-interpreted, gave up on too early, or didn't fully exhaust?
6. **Honest probability estimate**: given the data, what's the realistic probability that a single next packet produces a model that meets all three pillars? Be willing to say "low" if that's your read.

## Format of your reply

- 200-400 words of reasoning that walks through the data, named.
- A ranked list of next steps.
- A "things I'd want to know but FACTS.md doesn't have" section.
- A "things I'd revisit in §5" section, even if empty.
- A 1-line confidence statement.

## Red flags to self-check before submitting

- Are you defaulting to a "standard ML response" (curriculum learning, more data, larger model) without engaging with the specific evidence in §3-4 of FACTS.md? The shortcut readability AUC=1.0 at every layer + the deployment ≡ E2B finding + the layer-11 catastrophic divergence + the P8A-vs-E2B same-encoder-different-head pattern are unusual results; your recommendation should specifically engage with at least one of them.
- Are you proposing something the prior R13 already tried (§5)? Check before committing.
- Are you proposing a stack of multiple interventions without specifying which is load-bearing? FACTS.md §5 documents that bundles can be net-negative against single-strongest components (the P14 finding).
- Are you proposing a recipe and forgetting cross-substrate validation? The PA-on-HDTF walkback (§5, PA row) cost the team a packet's worth of work on a substrate-bound winner.

## Guard against your own auto-completing

You may have priors on "what's good for deepfake detection" from training data. Those priors may not match what the data here shows. Specifically:
- Frequency-domain analysis IS in scope (the team measured it — see Probes 3 and 6 in FACTS.md §4).
- Self-blended-image (SBI) pseudo-fakes are NOT currently part of the architecture (see ARCHITECTURE.md "Items NOT in this architecture").
- AugMix-style consistency loss is NOT currently implemented.
- Layer-level per-resblock probing has been done (Probe 7, 2026-05-06; an earlier version exists in `analysis/intermediate_layer_probe_2026-04-30/` per FACTS.md §5 cross-references).

If your recommendation lands in a space the team has already explored, please flag whether you think they did it wrong (what they should have done differently) or whether you're proposing a meaningfully different variant of the same lever class.

## What the team did NOT include in this pack on purpose

The team has its own ranked list of next-step recommendations and its own synthesis. They are deliberately not showing you those, so that your recommendation is independent. After your recommendation lands, they will share their list and ask you to compare/defend/synthesize. So don't try to guess what they think — just answer the question from the data.
