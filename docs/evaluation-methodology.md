# Evaluation Methodology

**Status:** Design approved; implementation and empirical validation pending.

- Compare Always-Large, Rule-Based, kNN and Cascade on matched held-out cases.
- Separate development/calibration data from golden references.
- Combine deterministic checks, rubric-based judge and human review.
- Report financial accuracy, completeness, unsupported claims and critical safety errors.
- Compare cost and latency only after documented quality/safety gates; if thresholds
  are unvalidated, report descriptive results without certification claims.
- Preserve failures, retries, call coverage and missing metric values.
- Distinguish observed, estimated, simulated and unknown cost/latency provenance.

See [decisions.md](decisions.md), section 8 for adopted policy.
