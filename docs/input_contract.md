# Operational input and output contract

The `analyze` command accepts an unlabeled UTF-8 CSV. It validates the complete file before normalization and fails closed on malformed identifiers, dates, nested values, label leakage, empty inputs, or rows without location evidence.

## Required columns

| Column | Contract |
|---|---|
| `order_id` | Non-null, non-blank, unique scalar identifier |
| `account_id` | Non-null, non-blank scalar identifier; repeated orders are allowed |
| `city` | Scalar text; may be blank when another location field is present |
| `street` | Scalar text; may be blank when another location field is present |
| `building` | Scalar text; may contain a field label and Unicode digits |
| `unit` | Scalar text; may be blank |
| `postcode` | Scalar text; may be blank |
| `family_token` | Optional supporting value represented by a column; values may be blank |
| `order_date` | Valid, non-null date or timestamp |

At least one of `city`, `street`, `building`, or `postcode` must be present on every row. Address text fields are limited to 512 characters.

The observation file must not contain evaluation or decision columns including `true_group_id`, `true_relation_type`, `building_id`, `unit_id`, `split`, `noise_level`, or `is_related`.

## Snapshot and history behavior

`--snapshot-date` excludes observations after the requested point in time. For each account, the resolver then considers observations inside the configured history window relative to that account's latest eligible observation. The default is 365 days.

The selected signature is one complete address tuple found in the eligible observations. Fields from different addresses are never combined.

## Secret and identifier handling

The command reads its HMAC salt from `RELATION_ID_SALT` by default. The value must have at least 16 characters and must be managed as an application secret. It is never written to an artifact.

Raw account IDs are replaced with deterministic run-level references of the form `REF-…`. HMAC prevents simple dictionary recovery when the salt remains secret, but the result is still linkable pseudonymous data rather than anonymous data.

## Output files

| File | Contents |
|---|---|
| `reports/predicted_group_assignments.csv` | Pseudonymous account references and predicted group IDs |
| `reports/component_guardrail.csv` | Pseudonymous accepted and deferred high-score edge decisions |
| `reports/review_groups.csv` | Aggregate group size, evidence strength, signal label, and review flag |
| `reports/candidate_block_audit.csv` | Blocking strategy, hashed block key, size, pair count, and decision |
| `reports/analysis_metadata.json` | Input counts and dates, policy version, control metrics, privacy boundary, and fingerprint |

No input copy, raw address, normalized address, raw account ID, pair-level negative evidence, or evaluation label is exported.

## Operational controls

- Restrict input and output access to authorized staff with an approved review purpose.
- Store salts outside source control and rotate them according to the owning system's policy.
- Define a retention period for both source observations and review outputs; delete expired artifacts.
- Monitor missingness, deferred oversized blocks, review volume, component sizes, drift, and reviewed false-merge rates.
- Never treat a link or component as proof of identity, household, family, residence, fraud, or eligibility.
- Keep human review, correction, and appeal paths available for any downstream use.
