# Job 11 — INTERPRETATION (OPINION)

> **DISCLAIMER:** This file contains opinion / read-of-the-evidence. For raw counts and tables, see `FINDINGS_FACTS.md`. The user reserves all decisions; the agent does not adjudicate "best ckpt".

## What the per-identity tables show

### Chronic-6 patterns
Each ckpt has a different "worst chronic" identity:

- P8A: Q__s6 FPR=0.944
- E2B_3200: Q__s6 FPR=0.593
- E3_6600: roy_d FPR=0.792

Cross-ckpt agreement on chronic-6 ranges from "all three above 50% FPR" cases (where the identity is just hard for everyone) to single-ckpt spikes (where one ckpt fires on essentially nothing while another fires on most of that identity's frames). The Step 2 "max-min Δ" column ranks identities by this disagreement. The interpretation: identities at the top of that table are NOT shared chronic offenders — they are ckpt-specific failures.

### Step 3 fake-recall vs Step 2 real-FPR
For most chronic identities that ALSO appear in fake suites, cross-ckpt recall disagreement is much smaller than cross-ckpt real-FPR disagreement. Caveat: small sample sizes per identity in lockbox.

A reading consistent with the data: ckpts disagree more on "is this real frame fake?" than on "is this fake frame fake?" for chronic identities. That is what one would expect if the disagreement is in the τ region (calibration and decision-boundary placement) rather than in the underlying representation.

### Anchor-selection lens
Look at Section 5 (uniquely-offending). The user's mandate was to treat the three ckpts as symmetric. Whether one ckpt has fewer ckpt-specific offenders than the others is a per-ckpt FACT in Section 5. Whether that translates to "cleaner real-side overall" depends on the user's real-side success criteria (FPR-budget-cost / per-identity floors / lockbox reproducibility), not on a single statistic.

The largest cross-ckpt FPR spread on a single identity (Section 2) is the most informative diagnostic for "which identities are ckpt-discriminating", and is therefore where any ensemble-rule design should be tuned.
