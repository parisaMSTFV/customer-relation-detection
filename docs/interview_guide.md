# Interview discussion guide

## What the project demonstrates

The project separates temporal normalization, bounded multi-pass candidate generation, truth-free pair scoring, evaluation-label attachment, validation-only graph-policy selection, held-out building evaluation, and operational pseudonymized review output. Exact normalized address matching is the conservative baseline.

## Decisions to explain

- Why buildings, rather than individual orders, define the data split boundary.
- Why Persian and other non-Latin letters must survive normalization and why building/unit parsing is field-specific.
- Why one complete observed address tuple is safer than independently selecting a modal value for every field.
- Why candidate recall must be measured before classifier metrics.
- Why multi-pass blocking recovers single-field errors and why oversized blocks are deferred before enumeration.
- Why missing postcode evidence is excluded from the denominator instead of counted as disagreement.
- Why city and building agreement are excluded from score weights when they often define a candidate block.
- Why family-token agreement has low weight and cannot prove a relationship.
- Why pairwise F1, component merge/split errors, ARI, and B-cubed metrics answer different questions.
- How one incorrect edge can merge otherwise separate groups through transitivity.
- Why threshold and component cap are selected together with a precision floor and asymmetric false-merge cost.
- Why the held-out precision falling slightly below the validation floor is evidence against treating validation policy as a guarantee.
- Why strongest-edge-first processing and a visible component cap defer risky expansion instead of silently accepting it.
- Why a frozen, versioned policy is used for unlabeled data rather than tuning on the input being scored.
- Why HMAC output is pseudonymous rather than anonymous and still requires retention and access controls.
- Why every output is a review signal and not a fraud or household determination.
