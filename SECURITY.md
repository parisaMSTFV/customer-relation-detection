# Security, privacy, and use boundary

This repository contains no real person, address, account, order, credential, connector, private URL, or production configuration. The pipeline runs offline. CI scans tracked and non-ignored workspace text for common credentials and private-infrastructure patterns, uses a locked dependency graph, enforces test coverage, and smoke-tests the built wheel.

The operational `analyze` command validates unlabeled observations before processing. It does not export raw addresses, normalized signatures, or raw account identifiers. Account references use HMAC-SHA256 with an externally supplied salt; this is pseudonymization, not anonymization. Inputs and review artifacts still require an approved purpose, least-privilege access, a retention schedule, deletion, and audit logging in the system that hosts them.

Oversized candidate blocks and component-expanding edges are deferred with explicit audit records. They must be reviewed rather than silently accepted or interpreted as negative evidence.

Predicted groups are shared-address signals that require review. They must not be treated as proof of family relationship, fraud, identity, eligibility, residence, or legal association. Do not use a similar workflow for adverse decisions without consent, governance, bias assessment, access controls, retention limits, and an appeal process.
