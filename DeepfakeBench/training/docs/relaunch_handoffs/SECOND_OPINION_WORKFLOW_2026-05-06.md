

> Three sequential prompts to use with a fresh in-house agent (one that has access to this repo + auto-memory). Order: Round 1 → Round 2 → Round 3. Each prompt is paste-ready. **Do not paste them all at once** — wait for the agent's response between each.
>
> In parallel, the user is also feeding the external-advisor pack (`docs/external_advisor_pack_2026-05-06/`) to a separate model with no repo access. Round 3 of this in-house workflow ingests the external advisor's response.

---

## Round 1 — independent recommendation from the in-house fresh agent

> Paste this verbatim as the user message. The agent will read the linked prompt + facts pack and form its own recommendation without seeing the team's existing ranked list.

---

I want a fresh, independent perspective on what the team should do next. Please follow the protocol in `docs/relaunch_handoffs/SECOND_OPINION_PROMPT_2026-05-06.md` exactly.

In short:

1. Read `docs/relaunch_handoffs/FACTS_FOR_SECOND_OPINION_2026-05-06.md` (the bias-stripped factual snapshot of the project state).
2. Read the cited raw CSV / JSON outputs from today's 7 CPU probes — paths in §8 of the facts pack.
3. **Do NOT read** the files listed in the "Do NOT read in the first pass" section of the protocol doc. Those contain the team's existing synthesis and would anchor your judgment. Specifically:
 - `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`
 - The 2026-05-06 sub-sections of `docs/packet_retrospectives/threads/processing_signature_shortcut.md`
 - `analysis/{amp_vs_phase_probe_2026-05-06, per_layer_p8a_e2b_pa_2026-05-06}/outputs/FINDINGS.md`
 - `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/MODEL_SCORES_FINDINGS.md`
4. Treat the auto-memory entries dated 2026-05-06 as historical citations and open-question registers, NOT as directives. They have been refactored to remove "How to apply" sections, but their cross-references still imply framings.

After your independent reading, give me your recommendation following the format in the protocol doc:
- 200-400 words of reasoning grounded in the data (cite specific probe outputs and prior R13 outcomes).
- Your own ranked list of next 1-3 packets.
- Things you'd want to know but the facts pack doesn't have.
- Anything in §5 (prior R13 attempts) you'd revisit.
- A 1-line honest confidence statement.

Engage specifically with the unusual results in §3-4: the AUC=1.0 shortcut readability at every layer, the deployment-≡-E2B finding, the layer-11 catastrophic divergence between P8A and E2B, and the bands 12-13 vs bands 5-6 frequency split. If your recommendation lands in something the prior R13 already tried (§5), be explicit about whether you think they did it wrong or whether you're proposing a meaningfully different variant.

Do not look ahead at any synthesis docs or recipe recommendations until I share them in a follow-up. Goal here is an independent recommendation.

---

## Round 2 — share the team's ranked list, ask for compare-and-defend

> Use this AFTER the agent has delivered Round 1. Do NOT modify based on what they said — the prompt below is fixed regardless of their answer. Their Round 1 stays as-is in the conversation history.

---

Thanks for the independent take. Now read the team's existing ranked list and synthesis:

1. `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md` — the 1-9 ranked technique list with fit / evidence / cost / risk scoring.
2. The 2026-05-06 afternoon sub-sections of `docs/packet_retrospectives/threads/processing_signature_shortcut.md` — the team's interpretation of each probe's "operational implications."
3. `analysis/per_layer_p8a_e2b_pa_2026-05-06/outputs/FINDINGS.md` — the team's synthesis of Probe 7 specifically.

Now compare your Round 1 recommendation to the team's existing list:

1. **Where do you and the team specifically agree?** List the technique-level overlaps with one-line reasons.
2. **Where do you specifically disagree?** Identify ranking disagreements and substantive technique-level disagreements separately.
3. **Did the team's reasoning update your view?** Be willing to defend, partially adopt, or fully synthesize. If you update, explain which piece of evidence in the team's docs moved you.
4. **Did your reasoning expose something the team missed?** Be willing to flag where you think they're wrong — even if it's just a question they didn't ask.
5. **Synthesized recommendation** (your final answer): given everything you've now seen, what do you actually think the team should do? You can keep your Round 1 list, adopt the team's, or propose a third synthesis.

Format: keep it focused. ~300-500 words plus a final ranked list. Include a 1-line "overall update direction" stating whether your view shifted toward, away from, or orthogonally to the team's list.

---

## Round 3 — ingest the external advisor's response

> Use this AFTER both the in-house Round 2 has landed AND the external advisor (running on the isolated pack in a different system) has returned its response. The user pastes the external advisor's full response inline below where indicated.

---

The team also got a second opinion from an external advisor running in a separate system with no repo access. They were given the isolated pack at `docs/external_advisor_pack_2026-05-06/` (architecture doc + facts pack + raw CSVs/JSONs only), and were asked the same question with no exposure to the team's synthesis.

Their response is below. Please:

1. Read it in full.
2. Compare it to your Round-2 synthesized recommendation AND to the team's existing ranked list.
3. Identify the **three views' specific points of agreement and disagreement**:
 - Team's existing ranked list (in `ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`)
 - Your Round-2 synthesized recommendation (above in this conversation)
 - The external advisor's recommendation (pasted below)

For each disagreement, identify which view has the stronger evidence basis given the data in the facts pack.

4. **Most importantly**: did the external advisor surface something that BOTH the team and you missed? They had no access to the team's prior framings, so any mention of an intervention class, a reframing, or a missing measurement that's NOT in either the team's list or your Round-2 list is a signal. Flag those explicitly — they're the highest-value output of this entire workflow.

5. **Final synthesis**: given all three views, what should the team actually do next? Be concrete: 1-3 named packets / probes, in priority order, with confidence intervals.

Format:
- A "convergence map" table: 3 columns (team / you / external), rows = key technique / direction, ✓ if endorsed, ✗ if rejected, ~ if mentioned but neutral, blank if absent.
- A "novel-to-external-advisor-only" section listing anything the external surfaced that neither the team nor you mentioned — even if you think it's wrong, note why it's worth dismissing.
- A "convergent recommendation" section: things all three views endorse → highest confidence to act on.
- A "divergent recommendation" section: places where the three views disagree → flag for the user to decide.
- Final 1-3 packet plan with confidence intervals.

### External advisor's response

[PASTE THE EXTERNAL ADVISOR'S FULL RESPONSE HERE BEFORE SENDING THE PROMPT]

---

## Notes for the user

### Sequencing

- Run Round 1 with a fresh in-house agent. Wait for response. Do not give them any of the team's synthesis docs in this round.
- Run Round 2 with the SAME in-house agent (continuation of conversation). They've now seen the team's list and can update.
- In parallel (or after), feed the isolated pack to an external advisor in a different system. Their response is independent of the in-house agent's Round 1.
- Run Round 3 with a fresh in-house agent (or the same one that did Rounds 1+2). Paste the external advisor's response inline. They produce the three-way synthesis.

### Why this structure

The point is to maximize the **divergence signal**:
- If the in-house Round 1 + external advisor independently land on similar recommendations, that's strong corroboration of the technique's fit.
- If they diverge, that's where the second-opinion value is.
- Round 3's "novel-to-external-advisor-only" section explicitly searches for things the in-house workflow missed — the external is the only view in the pipeline that hasn't seen any in-house synthesis at all.
- Round 2 forces the in-house agent to engage critically with the team's existing list (rather than just defending their first answer).

### What to do with the result

Round 3's "convergent recommendation" section = the highest-confidence next-step list. Run those packets with the standard close-criterion discipline (single-lever, cross-substrate validation in close criterion, pre-validation probes before GPU spend).

Round 3's "divergent recommendation" section = open questions that need a human (you) to make the judgment call.

If Round 3 surfaces a "novel-to-external-only" intervention that survives the team's and your scrutiny, that's the most valuable output of the entire workflow — it's the thing the in-house view literally couldn't see because of accumulated framing.
