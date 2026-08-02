# Interview discussion guide

## What the project demonstrates

The project separates candidate generation, pair scoring, validation-only threshold selection, held-out building evaluation, and graph clustering. It treats exact normalized address matching as a baseline and measures whether fuzzy, missing-aware evidence improves pair and component recovery.

## Decisions to explain

- Why buildings, rather than individual orders, define the data split boundary.
- Why candidate recall must be measured before classifier metrics.
- Why missing postcode evidence is excluded from the denominator instead of counted as disagreement.
- Why family-token agreement has low weight and cannot prove a relationship.
- Why pairwise F1 and connected-component ARI answer different questions.
- How one incorrect edge can merge otherwise separate groups through transitivity.
- Why every output is a review signal and not a fraud or household determination.
