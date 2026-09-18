# Human review feedback audit

The `audit-review-feedback` command turns adjudicated human review outcomes into aggregate quality telemetry. It is intentionally downstream of `analyze`: reviewers work with the pseudonymous edge export, and the audit never needs raw account IDs or raw addresses.

## Inputs

Use the `reports/component_guardrail.csv` created by `relation-detection analyze`.

Provide a second UTF-8 CSV with exactly one row per reviewed pair and these required columns:

| Column | Contract |
|---|---|
| `account_ref_a` | Pseudonymous reference from the guardrail export |
| `account_ref_b` | Pseudonymous reference from the guardrail export |
| `review_outcome` | `match`, `not_match`, or `uncertain` |

Pair order does not matter. A pair may appear only once after canonicalization. Feedback for a pair that does not exist in the supplied guardrail export is rejected.

The command accepts only references in the operational `REF-<20 hex>` format. Do not add raw identifiers, addresses, names, phone numbers, reviewer identities, free-text notes, or downstream decision labels to this file.

## Outcome meanings

- `match`: the reviewer found sufficient approved evidence to confirm that the edge belongs together for the defined review purpose.
- `not_match`: the reviewer found sufficient approved evidence that the edge should not be linked for the defined review purpose.
- `uncertain`: the reviewer could not reach a defensible decision from the approved evidence.

These outcomes are purpose-specific review labels. They must not be interpreted as proof of family relationship, identity, residence, fraud, eligibility, or legal association.

## Run the audit

```bash
uv run relation-detection audit-review-feedback \
  --guardrail path/to/review-output/reports/component_guardrail.csv \
  --feedback path/to/review-feedback.csv \
  --output-root path/to/review-audit
```

The command writes one file:

`reports/review_feedback_metrics.json`

The JSON contains aggregate review coverage, decisive and uncertain counts, observed accepted-edge precision, observed accepted-edge false-merge rate, deferred-edge match rate, breakdowns by guardrail decision, a fingerprint, and the evaluation boundary.

It does not export pair references.

## Interpretation

Observed precision and false-merge rate are conditional on the reviewed accepted edges. They are not estimates of recall, candidate coverage, full component quality, fairness, or business impact.

Review sampling matters. If reviewers inspect only easy, suspicious, high-value, or escalated cases, the observed rates describe that selected sample rather than all system output. Sampling design and any confidence-interval methodology should be defined before using these metrics for a production decision.
